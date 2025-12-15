import sys
import os
import glob
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import soundfile as sf

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QSpacerItem,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QComboBox,
)

# ============================================================
# Decent Sampler Wavetable Autosampler (3 wavetables / session)
# ============================================================
# - Elige 3 wavetables al azar (una sola vez por sesión)
# - Genera sustains con looping (loopStart/loopEnd)
# - Capas por velocidad (velocity layers) REALISTAS (timbre + volumen)
# - Round robin (seqMode/seqPosition/seqLength)
# - True legato / portamento (transiciones) + LEGATO-TAILS (sustain post transición)
# - Looping “invisible”: crossfade horneado + recorte de WAVs
#
# NOTA: Este generador es "synth interno simple" (wavetable osc + env + filtro)
# y está diseñado para correr en CI/nube sin dependencias extra.
#
# Output:
#   OUT/
#     Samples/*.wav
#     <InstrumentName>.dspreset
#     session.json
#
# ============================================================

AUDIO_EXTS = (".wav", ".aif", ".aiff")
DEFAULT_WT_DIR = r"D:\WAVETABLE"  # útil en Windows; en nube probablemente no exista

# -----------------------------
# Audio / Wavetable
# -----------------------------
SR_DEFAULT = 44100
WT_FRAME_SIZE = 2048
WT_MIP_LEVELS = 8

# -----------------------------
# Sampling Defaults
# -----------------------------
LOW_MIDI_DEFAULT = 36   # C2
HIGH_MIDI_DEFAULT = 96  # C7
NOTE_STEP_DEFAULT = 3   # cada 3 semitonos

VEL_LAYERS_DEFAULT = 6
RR_DEFAULT = 3

SUS_ATTACK_DEFAULT = 0.025
SUS_HOLD_DEFAULT = 6.0
SUS_RELEASE_DEFAULT = 0.35

# Looping
LOOP_XFADE_DEFAULT = 1024
LOOP_MIN_SEC_DEFAULT = 0.35
LOOP_MAX_SEC_DEFAULT = 1.20

# Velocity layers realistas (volumen)
VEL_AMP_MIN_DEFAULT = 0.22     # nivel mínimo para vel baja (0..1)
VEL_AMP_CURVE_DEFAULT = 1.35   # curva >1 = más diferencia entre capas

# Legato / Portamento
LEGATO_ENABLE_DEFAULT = True
LEGATO_SET = {
    "Minimal (±2, ±7, ±12)": [-12, -7, -2, 2, 7, 12],
    "Normal (±1..±5, ±7, ±12)": [-12, -7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 12],
    "Full (±1..±12)": list(range(-12, 0)) + list(range(1, 13)),
}

LEGATO_SPEEDS = {
    "Fast": 0.80,
    "Medium": 1.00,
    "Slow": 1.25,
}
LEGATO_SPEED_DEFAULT = "Medium"

# Legato transitions WAV envelope (rendering)
LEGATO_WAV_ATTACK_DEFAULT = 0.005
LEGATO_WAV_RELEASE_DEFAULT = 0.20

# Patch "feel" (ajustable)
DETUNE_CENTS_RR_DEFAULT = 6.0
UNISON_VOICES_DEFAULT = 2
UNISON_SPREAD_CENTS_DEFAULT = 7.0
SAT_AMOUNT_DEFAULT = 0.35
NOISE_LEVEL_DEFAULT = 0.0025
VEL_BRIGHTNESS_DEFAULT = 1.2
TARGET_PEAK_DEFAULT = 0.95

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ============================================================
# Helpers (music, naming, file listing)
# ============================================================
def midi_to_hz(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def midi_to_name(m: int) -> str:
    n = NOTE_NAMES[m % 12]
    o = (m // 12) - 1
    return f"{n}{o}"


def list_wavetable_files(folder: str, recursive: bool = True) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    pattern = "**/*" if recursive else "*"
    out = []
    for ext in AUDIO_EXTS:
        out += glob.glob(os.path.join(folder, pattern + ext), recursive=recursive)
        out += glob.glob(os.path.join(folder, pattern + ext.upper()), recursive=recursive)
    out = [p for p in out if os.path.isfile(p)]
    out.sort(key=lambda p: p.lower())
    return out


def vel_splits_equal(n_layers: int) -> List[Tuple[int, int, float]]:
    """
    Devuelve capas por velocidad en rangos uniformes.
    Output: (loVel, hiVel, vel_norm 0..1)
    """
    n_layers = int(np.clip(n_layers, 2, 12))
    edges = np.linspace(1, 128, n_layers + 1, dtype=int)
    layers = []
    for i in range(n_layers):
        lo = int(edges[i])
        hi = int(edges[i + 1] - 1)
        mid = (lo + hi) / 2.0
        vel_norm = float(np.clip(mid / 127.0, 0.0, 1.0))
        layers.append((lo, hi, vel_norm))
    # corrige último hi
    layers[-1] = (layers[-1][0], 127, layers[-1][2])
    return layers


def root_notes(low: int, high: int, step: int) -> List[int]:
    step = int(max(1, step))
    notes = list(range(int(low), int(high) + 1, step))
    if notes[-1] != int(high):
        notes.append(int(high))
    return notes


def zone_bounds(roots: List[int]) -> List[Tuple[int, int, int]]:
    """
    Para cada root midi: (loNote, hiNote, rootNote) usando midpoints.
    """
    out = []
    for i, r in enumerate(roots):
        if i == 0:
            lo = 0
        else:
            lo = int(math.floor((roots[i - 1] + r) / 2.0)) + 1
        if i == len(roots) - 1:
            hi = 127
        else:
            hi = int(math.floor((r + roots[i + 1]) / 2.0))
        out.append((lo, hi, r))
    return out


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# Wavetable load + mipmaps (reuso de lógica tipo "mip")
# ============================================================
def _to_mono_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return x.astype(np.float32, copy=False)


def _fade_edges(frame: np.ndarray, fade: int = 8) -> np.ndarray:
    if fade <= 0 or 2 * fade >= len(frame):
        return frame
    w = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out = frame.copy()
    out[:fade] *= w
    out[-fade:] *= w[::-1]
    return out


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(frame)) + 1e-12
    return (frame / m).astype(np.float32, copy=False)


def _linear_resample(x: np.ndarray, new_len: int) -> np.ndarray:
    n = len(x)
    if new_len == n:
        return x.astype(np.float32, copy=False)
    src = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, new_len, endpoint=False, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32, copy=False)


def load_wavetable_wav(
    path: str,
    frame_size: int = WT_FRAME_SIZE,
    normalize_each_frame: bool = True,
    edge_fade: int = 8,
) -> np.ndarray:
    """
    Carga wavetable (mono). Si es largo y múltiplo de frame_size, lo interpreta como multi-frame.
    Si no, lo fuerza a 1 frame.
    Devuelve frames: [n_frames, frame_size]
    """
    audio, _sr = sf.read(path, always_2d=False)
    audio = _to_mono_float(audio)
    if len(audio) < frame_size:
        audio = np.pad(audio, (0, frame_size - len(audio)))
    n_frames = max(1, len(audio) // frame_size)
    use_len = n_frames * frame_size
    audio = audio[:use_len]
    frames = audio.reshape(n_frames, frame_size).copy()
    for i in range(n_frames):
        f = frames[i]
        f = f - np.mean(f)
        f = _fade_edges(f, edge_fade)
        if normalize_each_frame:
            f = _normalize_frame(f)
        frames[i] = f
    return frames.astype(np.float32, copy=False)


def build_wavetable_mipmaps(frames: np.ndarray, levels: int = WT_MIP_LEVELS) -> List[np.ndarray]:
    """
    Lista de niveles, cada uno: [n_frames, table_len_level]
    """
    frames = np.asarray(frames, dtype=np.float32)
    n_frames, frame_size = frames.shape
    mipmaps = []
    cur = frames
    cur_size = frame_size
    for _lvl in range(int(max(1, levels))):
        mipmaps.append(cur)
        next_size = max(32, cur_size // 2)
        if next_size == cur_size:
            break
        nxt = np.zeros((n_frames, next_size), dtype=np.float32)
        for fi in range(n_frames):
            nxt[fi] = _linear_resample(cur[fi], next_size)
        cur = nxt
        cur_size = next_size
    return mipmaps


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * float(t)


def _table_read_linear(table_1d: np.ndarray, phase: np.ndarray) -> np.ndarray:
    n = len(table_1d)
    idx = phase * n
    i0 = np.floor(idx).astype(np.int32)
    frac = (idx - i0).astype(np.float32)
    i1 = (i0 + 1) % n
    return (1.0 - frac) * table_1d[i0] + frac * table_1d[i1]


def render_wavetable_osc(
    f0_hz: np.ndarray,
    sr: int,
    mipmaps: List[np.ndarray],
    position: float,
    phase0: float,
    mip_strength: float,
) -> np.ndarray:
    """
    Osc wavetable con mipmaps:
    - position: interp entre frames (0..1)
    - mip_strength: 0..1 => selecciona mip level más agresivo en agudos
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    n_samples = len(f0_hz)
    n_levels = len(mipmaps)
    base_frames = mipmaps[0]
    n_frames = base_frames.shape[0]

    pos = float(np.clip(position, 0.0, 1.0))
    fidx = pos * (n_frames - 1) if n_frames > 1 else 0.0
    f0i = int(np.floor(fidx))
    ft = float(fidx - f0i)
    f1i = min(f0i + 1, n_frames - 1)

    phase = np.empty(n_samples, dtype=np.float32)
    ph = float(phase0 % 1.0)
    for i in range(n_samples):
        phase[i] = ph
        ph += float(f0_hz[i]) / float(sr)
        ph -= math.floor(ph)

    # mip selector simple por frecuencia
    f_ref = 55.0
    ratio = np.maximum(f0_hz / f_ref, 1e-6)
    lvl_float = np.log2(ratio) * float(np.clip(mip_strength, 0.0, 1.0))
    lvl = np.clip(np.floor(lvl_float).astype(np.int32), 0, n_levels - 1)

    out = np.zeros(n_samples, dtype=np.float32)
    for L in range(n_levels):
        mask = (lvl == L)
        if not np.any(mask):
            continue
        tables_L = mipmaps[L]
        t0 = tables_L[f0i]
        t1 = tables_L[f1i]
        table = _lerp(t0, t1, ft)
        out[mask] = _table_read_linear(table, phase[mask]).astype(np.float32)
    return out


# ============================================================
# Synthesis (env + filter + saturation)
# ============================================================
def adsr_env(n: int, sr: int, attack: float, release: float) -> np.ndarray:
    a = max(1, int(float(attack) * sr))
    r = max(1, int(float(release) * sr))
    sus = max(0, n - a - r)
    env_a = np.linspace(0.0, 1.0, a, endpoint=False, dtype=np.float32)
    env_s = np.ones((sus,), dtype=np.float32)
    # release exponencial suave
    x = np.linspace(0.0, 1.0, r, endpoint=True, dtype=np.float32)
    env_r = (1.0 - x) ** 2.2
    return np.concatenate([env_a, env_s, env_r], axis=0)[:n].astype(np.float32)


def one_pole_lowpass(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    cutoff_hz = float(np.clip(cutoff_hz, 30.0, sr * 0.45))
    a = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / sr)
    y = np.empty_like(x, dtype=np.float32)
    z = 0.0
    for i in range(len(x)):
        z = z + a * (float(x[i]) - z)
        y[i] = z
    return y


def soft_clip(x: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 1e-6:
        return x.astype(np.float32, copy=False)
    drive = 1.0 + 6.0 * amount
    return np.tanh(x * drive).astype(np.float32)


@dataclass
class Patch:
    wt_paths: List[str]   # 3 paths
    wt_positions: List[float]  # 0..1 por WT (frame interp)
    mix: List[float]      # 3 pesos sum=1
    mip_strength: float
    seed: int


def choose_patch(wt_files: List[str], seed: int) -> Patch:
    if len(wt_files) < 3:
        raise RuntimeError("Necesitas al menos 3 wavetables en la carpeta.")
    # seed=0 => random real
    rng = random.Random(None if seed == 0 else seed)

    picks = rng.sample(wt_files, 3)
    w = np.array([rng.random(), rng.random(), rng.random()], dtype=np.float32)
    w = (w / np.sum(w)).tolist()

    pos = [float(rng.uniform(0.0, 1.0)) for _ in range(3)]
    mip_strength = float(rng.uniform(0.65, 1.0))  # aliasing control (constante sesión)
    return Patch(wt_paths=picks, wt_positions=pos, mix=w, mip_strength=mip_strength, seed=seed)


def glide_time_for_interval(semi: int) -> float:
    a = abs(int(semi))
    if a <= 2:
        return 0.075
    if a <= 5:
        return 0.115
    if a <= 12:
        return 0.155
    return 0.200


def render_synth(
    wt_mips: List[List[np.ndarray]],  # 3 wavetables, cada uno mipmaps list
    patch: Patch,
    midi: int,
    vel_norm: float,
    sr: int,
    seconds: float,
    attack: float,
    release: float,
    rr_index: int,
    rr_count: int,
    detune_cents_rr: float,
    unison_voices: int,
    unison_spread_cents: float,
    sat_amount: float,
    noise_level: float,
    vel_brightness: float,
    target_peak: float,
    vel_amp_min: float,
    vel_amp_curve: float,
    np_rng: np.random.Generator,
    glide_from_hz: Optional[float] = None,
    glide_time: float = 0.0,
) -> np.ndarray:
    n = int(max(1, seconds * sr))
    f_to = midi_to_hz(int(midi))

    if glide_from_hz is None or glide_time <= 1e-6:
        f = np.full((n,), f_to, dtype=np.float32)
    else:
        g = int(max(1, min(n, glide_time * sr)))
        ratio = float(f_to / float(glide_from_hz))
        k = np.arange(g, dtype=np.float32)
        ramp = (float(glide_from_hz) * (ratio ** (k / max(1.0, float(g - 1))))).astype(np.float32)
        f = np.full((n,), f_to, dtype=np.float32)
        f[:g] = ramp

    # Round robin randomización determinista (detune + fase)
    phase0 = float(np_rng.random())
    cents = float(np_rng.uniform(-detune_cents_rr, detune_cents_rr))
    # centrado por rr_index
    if rr_count > 1:
        cents += float(unison_spread_cents * ((rr_index - 1) - (rr_count - 1) / 2.0) / (rr_count - 1))
    detune = 2.0 ** (cents / 1200.0)
    f = (f * detune).astype(np.float32)

    # Brillo por velocity (+ nota)
    note_norm = float(np.clip((midi - LOW_MIDI_DEFAULT) / max(1.0, (HIGH_MIDI_DEFAULT - LOW_MIDI_DEFAULT)), 0.0, 1.0))
    cutoff = 1200.0 + (vel_norm ** float(max(0.1, vel_brightness))) * 12000.0 + note_norm * 2500.0

    # Mezcla de 3 wavetables
    sig = np.zeros((n,), dtype=np.float32)
    for j in range(3):
        mipmaps = wt_mips[j]
        pos = patch.wt_positions[j]
        mix = float(patch.mix[j])

        if unison_voices <= 1:
            sig += mix * render_wavetable_osc(f, sr, mipmaps, pos, phase0 + 0.17 * j, patch.mip_strength)
        else:
            u = np.zeros((n,), dtype=np.float32)
            for v in range(int(unison_voices)):
                cents_u = (v - (unison_voices - 1) / 2.0) * float(unison_spread_cents)
                det_u = 2.0 ** (cents_u / 1200.0)
                u += render_wavetable_osc(f * det_u, sr, mipmaps, pos, phase0 + 0.23 * v + 0.11 * j, patch.mip_strength)
            u /= float(unison_voices)
            sig += mix * u

    # ruido sutil
    sig += float(noise_level) * np_rng.standard_normal(n).astype(np.float32)

    # filtro
    sig = one_pole_lowpass(sig, cutoff_hz=cutoff, sr=sr)

    # envolvente
    env = adsr_env(n, sr, attack=attack, release=release)
    sig = (sig * env).astype(np.float32)

    # saturación
    sig = soft_clip(sig, sat_amount)

    # normalización pico
    peak = float(np.max(np.abs(sig)) + 1e-12)
    if peak > 0:
        sig = (sig * (float(target_peak) / peak)).astype(np.float32)

    # --- Dinámica real por velocity (IMPORTANTE) ---
    # Mantiene cambios de volumen entre capas; amp siempre <= 1 para no clipear.
    vel_norm = float(np.clip(vel_norm, 0.0, 1.0))
    amp = float(vel_amp_min + (1.0 - vel_amp_min) * (vel_norm ** float(max(0.05, vel_amp_curve))))
    amp = float(np.clip(amp, 0.0, 1.0))
    sig = (sig * amp).astype(np.float32)

    return sig


# ============================================================
# Auto-loop detection (estable + cruces por cero)
# ============================================================
@dataclass
class LoopPoints:
    loop_start: int
    loop_end: int
    loop_xfade: int


def _nearest_zero(x: np.ndarray, idx: int, search: int = 512) -> int:
    lo = max(0, idx - search)
    hi = min(len(x) - 1, idx + search)
    seg = x[lo:hi]
    k = int(np.argmin(np.abs(seg)))
    return lo + k


def find_loop_points(
    x: np.ndarray,
    sr: int,
    loop_min_sec: float,
    loop_max_sec: float,
    loop_xfade: int,
    start_after_sec: float = 0.25,
) -> LoopPoints:
    n = len(x)
    start_after = int(np.clip(start_after_sec * sr, 0, max(0, n - 1)))

    # estimación de pitch por autocorrelación (para elegir loop length)
    probe_len = min(int(0.25 * sr), n - start_after)
    probe = x[start_after:start_after + probe_len].astype(np.float32, copy=False)
    if len(probe) < 2048:
        L_sec = 0.6
    else:
        p = probe - float(np.mean(probe))
        fft = np.fft.rfft(p, n=2 * len(p))
        ac = np.fft.irfft(fft * np.conj(fft))[:len(p)]
        ac[0] = 0.0
        min_lag = int(sr / 2000.0)
        max_lag = int(sr / 40.0)
        min_lag = max(1, min_lag)
        max_lag = min(len(ac) - 1, max_lag)
        lag = int(min_lag + np.argmax(ac[min_lag:max_lag]))
        f_est = sr / max(1, lag)
        L_sec = float(np.clip(3.0 / max(40.0, f_est), loop_min_sec, loop_max_sec))

    L = int(L_sec * sr)
    L = max(2048, min(L, n // 2))

    # candidato dentro del sustain (evita final)
    cand_lo = start_after
    cand_hi = min(n - L - 1, n - int(0.40 * sr))
    if cand_hi <= cand_lo:
        cand_hi = cand_lo + 1

    win = 2048
    hop = 256
    best = cand_lo
    best_score = 1e9

    for s in range(cand_lo, cand_hi, hop):
        seg = x[s:s + L]
        if len(seg) < L:
            break
        rms = []
        for k in range(0, L - win, win):
            w = seg[k:k + win]
            rms.append(float(np.sqrt(np.mean(w * w) + 1e-12)))
        if len(rms) < 3:
            continue
        score = float(np.std(rms) / (np.mean(rms) + 1e-12))
        if score < best_score:
            best_score = score
            best = s

    loop_start = _nearest_zero(x, best, search=512)
    loop_end = _nearest_zero(x, loop_start + L, search=512)
    loop_end = max(loop_end, loop_start + 1024)

    loop_start = int(np.clip(loop_start, 0, n - 2))
    loop_end = int(np.clip(loop_end, loop_start + 1, n - 1))

    return LoopPoints(loop_start=loop_start, loop_end=loop_end, loop_xfade=int(max(0, loop_xfade)))


def apply_baked_loop_crossfade(x: np.ndarray, loop_start: int, loop_end: int, xfade: int) -> np.ndarray:
    """Hornear crossfade dentro del audio para que el loop sea inaudible."""
    xfade = int(max(0, xfade))
    if xfade <= 0:
        return x
    if loop_end - loop_start <= xfade * 2:
        return x

    a0 = int(loop_start)
    a1 = int(loop_start + xfade)
    b0 = int(loop_end - xfade)
    b1 = int(loop_end)

    if a1 <= a0 or b1 <= b0:
        return x
    if a0 < 0 or b1 > len(x):
        return x

    fade_in = np.linspace(0.0, 1.0, xfade, dtype=np.float32)
    fade_out = 1.0 - fade_in

    head = x[a0:a1].copy()
    tail = x[b0:b1].copy()

    x[b0:b1] = tail * fade_out + head * fade_in
    return x


def trim_after_loop(x: np.ndarray, sr: int, loop_end: int, keep_sec: float = 0.15) -> np.ndarray:
    """Recorta el archivo para que no sea enorme: conserva hasta loop_end + pequeño colchón."""
    keep = int(max(0.0, float(keep_sec)) * int(sr))
    end = int(min(len(x), int(loop_end) + keep))
    end = max(end, int(loop_end) + 1)
    return x[:end].astype(np.float32, copy=False)


# ============================================================
# Decent Sampler preset writer (XML)
# ============================================================
def _posix_rel(path: str) -> str:
    # DS espera paths con /
    return str(Path(path).as_posix())


def _xml_indent(elem, level=0):
    # compatible con python 3.9+ sin ET.indent
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            _xml_indent(e, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def build_dspreset(
    instrument_name: str,
    sustain_samples: List[Dict[str, str]],
    legato_samples: Optional[List[Dict[str, str]]] = None,
    legato_tail_samples: Optional[List[Dict[str, str]]] = None,
) -> str:
    import xml.etree.ElementTree as ET

    root = ET.Element("DecentSampler", attrib={"pluginVersion": "1"})
    groups = ET.SubElement(root, "groups", attrib={"volume": "0dB"})

    # Sustains: trigger first
    g_sus = ET.SubElement(groups, "group", attrib={
        "tags": "sustain",
        "trigger": "first",
        "silencedByTags": "sustain,legato,legato-tails",
        "silencingMode": "normal",
        "ampVelTrack": "0",
    })
    for s in sustain_samples:
        ET.SubElement(g_sus, "sample", attrib=s)

    if legato_samples and legato_tail_samples:
        # Legato transitions
        g_leg = ET.SubElement(groups, "group", attrib={
            "tags": "legato",
            "trigger": "legato",
            "silencedByTags": "sustain,legato,legato-tails",
            "silencingMode": "normal",
            "attack": "0.002",
            "decay": "0.040",
            "sustain": "0.0",
            "release": "0.160",
        })
        for s in legato_samples:
            ET.SubElement(g_leg, "sample", attrib=s)

        # Legato tails: sustains sin ataque (start=loopStart)
        g_tail = ET.SubElement(groups, "group", attrib={
            "tags": "legato-tails",
            "trigger": "legato",
            "silencedByTags": "sustain,legato,legato-tails",
            "silencingMode": "normal",
            "attack": "0.010",   # fade-in corto para empalmar con transición
            "release": "0.250",
            "ampVelTrack": "0",
            "volume": "-3dB",
        })
        for s in legato_tail_samples:
            ET.SubElement(g_tail, "sample", attrib=s)

        tags = ET.SubElement(root, "tags")
        ET.SubElement(tags, "tag", attrib={"name": "legato", "polyphony": "1"})
        ET.SubElement(tags, "tag", attrib={"name": "legato-tails", "polyphony": "1"})

    _xml_indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")


# ============================================================
# Worker (thread) — genera librería completa
# ============================================================
class BuildWorker(QObject):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        out_dir: str,
        wt_dir: str,
        instrument_name: str,
        seed: int,
        sample_rate: int,
        low_midi: int,
        high_midi: int,
        note_step: int,
        vel_layers: int,
        rr_count: int,
        sus_attack: float,
        sus_hold: float,
        sus_release: float,
        loop_xfade: int,
        loop_min_sec: float,
        loop_max_sec: float,
        detune_cents_rr: float,
        unison_voices: int,
        unison_spread_cents: float,
        sat_amount: float,
        noise_level: float,
        vel_brightness: float,
        target_peak: float,
        make_legato: bool,
        legato_preset_name: str,
        legato_speed_factor: float,
        legato_wav_attack: float,
        legato_wav_release: float,
        keep_oneshots: bool,
        oneshot_seconds: float,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.wt_dir = wt_dir
        self.instrument_name = instrument_name
        self.seed = int(seed)
        self.sr = int(sample_rate)
        self.low_midi = int(low_midi)
        self.high_midi = int(high_midi)
        self.note_step = int(note_step)
        self.vel_layers = int(vel_layers)
        self.rr_count = int(rr_count)

        self.sus_attack = float(sus_attack)
        self.sus_hold = float(sus_hold)
        self.sus_release = float(sus_release)

        self.loop_xfade = int(loop_xfade)
        self.loop_min_sec = float(loop_min_sec)
        self.loop_max_sec = float(loop_max_sec)

        self.detune_cents_rr = float(detune_cents_rr)
        self.unison_voices = int(unison_voices)
        self.unison_spread_cents = float(unison_spread_cents)
        self.sat_amount = float(sat_amount)
        self.noise_level = float(noise_level)
        self.vel_brightness = float(vel_brightness)
        self.target_peak = float(target_peak)

        self.make_legato = bool(make_legato)
        self.legato_preset_name = legato_preset_name
        self.legato_speed_factor = float(legato_speed_factor)

        self.legato_wav_attack = float(legato_wav_attack)
        self.legato_wav_release = float(legato_wav_release)

        self.keep_oneshots = bool(keep_oneshots)
        self.oneshot_seconds = float(oneshot_seconds)

        self._mip_cache: Dict[str, List[np.ndarray]] = {}

    def _load_mips(self, wt_path: str) -> List[np.ndarray]:
        wt_path = os.path.abspath(wt_path)
        mm = self._mip_cache.get(wt_path)
        if mm is not None:
            return mm
        frames = load_wavetable_wav(wt_path, frame_size=WT_FRAME_SIZE)
        mm = build_wavetable_mipmaps(frames, levels=WT_MIP_LEVELS)
        self._mip_cache[wt_path] = mm
        return mm

    def _write_wav(self, path: str, audio: np.ndarray):
        sf.write(path, audio, self.sr, subtype="PCM_16")

    def run(self):
        try:
            wt_files = list_wavetable_files(self.wt_dir, recursive=True)
            if len(wt_files) < 3:
                raise RuntimeError("No se encontraron al menos 3 wavetables (.wav/.aif/.aiff) en la carpeta indicada.")

            safe_mkdir(self.out_dir)
            samples_dir = os.path.join(self.out_dir, "Samples")
            safe_mkdir(samples_dir)

            # Patch fijo sesión
            patch = choose_patch(wt_files, self.seed)

            self.log.emit("=== Sesión / Patch ===")
            self.log.emit(f"Instrument: {self.instrument_name}")
            self.log.emit(f"Seed: {'RANDOM' if self.seed == 0 else self.seed}")
            for i, p in enumerate(patch.wt_paths, start=1):
                self.log.emit(f"WT{i}: {os.path.basename(p)} | pos={patch.wt_positions[i-1]:.2f} | mix={patch.mix[i-1]:.2f}")
            self.log.emit(f"Mip strength: {patch.mip_strength:.2f}")
            if self.make_legato:
                self.log.emit(f"Legato speed factor: {self.legato_speed_factor:.2f}")
                self.log.emit(f"Legato WAV env: A={self.legato_wav_attack:.3f}s R={self.legato_wav_release:.3f}s")

            # Carga mipmaps (cache)
            wt_mips = [self._load_mips(p) for p in patch.wt_paths]

            # Plan
            roots = root_notes(self.low_midi, self.high_midi, self.note_step)
            bounds = zone_bounds(roots)
            layers = vel_splits_equal(self.vel_layers)

            legato_intervals = LEGATO_SET.get(self.legato_preset_name, LEGATO_SET["Normal (±1..±5, ±7, ±12)"])

            # Estimación de cantidad total para progreso
            total_sus = len(bounds) * len(layers) * self.rr_count
            total_leg = 0
            if self.make_legato:
                total_leg = len(bounds) * len(layers) * self.rr_count * len(legato_intervals)
            total_one = len(bounds) * len(layers) * self.rr_count if self.keep_oneshots else 0
            total = max(1, total_sus + total_leg + total_one)

            done = 0
            sustain_nodes: List[Dict[str, str]] = []
            legato_nodes: List[Dict[str, str]] = []

            # session json (metadata)
            session = {
                "instrument": self.instrument_name,
                "sample_rate": self.sr,
                "patch": asdict(patch),
                "sampling": {
                    "low_midi": self.low_midi,
                    "high_midi": self.high_midi,
                    "note_step": self.note_step,
                    "vel_layers": self.vel_layers,
                    "rr_count": self.rr_count,
                    "sustain": {"attack": self.sus_attack, "hold": self.sus_hold, "release": self.sus_release},
                    "loop": {"xfade": self.loop_xfade, "min_sec": self.loop_min_sec, "max_sec": self.loop_max_sec},
                    "legato": {
                        "enabled": self.make_legato,
                        "preset": self.legato_preset_name,
                        "intervals": legato_intervals,
                        "speed_factor": self.legato_speed_factor,
                                            "wav_attack": self.legato_wav_attack,
                        "wav_release": self.legato_wav_release,
                    },
                },
                "synth": {
                    "detune_cents_rr": self.detune_cents_rr,
                    "unison_voices": self.unison_voices,
                    "unison_spread_cents": self.unison_spread_cents,
                    "sat_amount": self.sat_amount,
                    "noise_level": self.noise_level,
                    "vel_brightness": self.vel_brightness,
                    "target_peak": self.target_peak,
                    "vel_amp_min": VEL_AMP_MIN_DEFAULT,
                    "vel_amp_curve": VEL_AMP_CURVE_DEFAULT,
                },
            }
            with open(os.path.join(self.out_dir, "session.json"), "w", encoding="utf-8") as f:
                json.dump(session, f, indent=2)

            # Render sustains (+ loop)
            self.log.emit("\n=== Render sustains (looped) ===")
            for loNote, hiNote, rootNote in bounds:
                for layer_idx, (loVel, hiVel, vel_norm) in enumerate(layers, start=1):
                    for rr in range(1, self.rr_count + 1):
                        # rng determinista por sample:
                        key = (rootNote * 1000 + layer_idx * 100 + rr) & 0xFFFFFFFF
                        base_seed = (1234567 if self.seed == 0 else self.seed) ^ key
                        np_rng = np.random.default_rng(base_seed)

                        seconds = self.sus_attack + self.sus_hold + self.sus_release
                        audio = render_synth(
                            wt_mips=wt_mips,
                            patch=patch,
                            midi=rootNote,
                            vel_norm=vel_norm,
                            sr=self.sr,
                            seconds=seconds,
                            attack=self.sus_attack,
                            release=self.sus_release,
                            rr_index=rr,
                            rr_count=self.rr_count,
                            detune_cents_rr=self.detune_cents_rr,
                            unison_voices=self.unison_voices,
                            unison_spread_cents=self.unison_spread_cents,
                            sat_amount=self.sat_amount,
                            noise_level=self.noise_level,
                            vel_brightness=self.vel_brightness,
                            target_peak=self.target_peak,
                            vel_amp_min=VEL_AMP_MIN_DEFAULT,
                            vel_amp_curve=VEL_AMP_CURVE_DEFAULT,
                            np_rng=np_rng,
                        )

                        lp = find_loop_points(
                            audio,
                            sr=self.sr,
                            loop_min_sec=self.loop_min_sec,
                            loop_max_sec=self.loop_max_sec,
                            loop_xfade=self.loop_xfade,
                            start_after_sec=max(0.12, self.sus_attack * 2.0),
                        )

                        # 1) hornea crossfade del loop en el audio
                        audio = apply_baked_loop_crossfade(audio, lp.loop_start, lp.loop_end, lp.loop_xfade)

                        # 2) recorta para que no sea enorme
                        audio = trim_after_loop(audio, sr=self.sr, loop_end=lp.loop_end, keep_sec=0.20)

                        fname = f"sus_{midi_to_name(rootNote)}_v{layer_idx}_rr{rr}.wav"
                        out_path = os.path.join(samples_dir, fname)
                        self._write_wav(out_path, audio)

                        sustain_nodes.append({
                            "path": _posix_rel(f"Samples/{fname}"),
                            "rootNote": str(rootNote),
                            "loNote": str(loNote),
                            "hiNote": str(hiNote),
                            "loVel": str(loVel),
                            "hiVel": str(hiVel),
                            "seqMode": "round_robin",
                            "seqPosition": str(rr),
                            "seqLength": str(self.rr_count),
                            "loopEnabled": "true",
                            "loopStart": str(lp.loop_start),
                            "loopEnd": str(lp.loop_end),
                            "loopCrossfade": "0",  # ya horneado
                            "loopCrossfadeMode": "equal_power",
                        })

                        done += 1
                        if done % 5 == 0:
                            self.progress.emit(int(100 * done / total))

            # Legato tails: usa los mismos sustains pero arrancando desde loopStart
            legato_tail_nodes: List[Dict[str, str]] = []
            if self.make_legato:
                for s in sustain_nodes:
                    tail = dict(s)
                    tail["start"] = tail.get("loopStart", "0")
                    # Asegura que el tail solo aplique a la nota destino (no a toda la zona)
                    tail["loNote"] = tail.get("rootNote", tail.get("loNote", "0"))
                    tail["hiNote"] = tail.get("rootNote", tail.get("hiNote", "127"))
                    legato_tail_nodes.append(tail)

            # One-shots (opcionales, no mapeados en preset)
            if self.keep_oneshots:
                self.log.emit("\n=== Render one-shots (extras) ===")
                ones_dir = os.path.join(self.out_dir, "OneShots")
                safe_mkdir(ones_dir)

                for _loNote, _hiNote, rootNote in bounds:
                    for layer_idx, (_loVel, _hiVel, vel_norm) in enumerate(layers, start=1):
                        for rr in range(1, self.rr_count + 1):
                            key = (rootNote * 2000 + layer_idx * 200 + rr + 77) & 0xFFFFFFFF
                            base_seed = (7654321 if self.seed == 0 else self.seed) ^ key
                            np_rng = np.random.default_rng(base_seed)

                            audio = render_synth(
                                wt_mips=wt_mips,
                                patch=patch,
                                midi=rootNote,
                                vel_norm=vel_norm,
                                sr=self.sr,
                                seconds=self.oneshot_seconds,
                                attack=min(self.sus_attack, 0.02),
                                release=min(0.25, self.sus_release),
                                rr_index=rr,
                                rr_count=self.rr_count,
                                detune_cents_rr=self.detune_cents_rr,
                                unison_voices=self.unison_voices,
                                unison_spread_cents=self.unison_spread_cents,
                                sat_amount=self.sat_amount,
                                noise_level=self.noise_level,
                                vel_brightness=self.vel_brightness,
                                target_peak=self.target_peak,
                                vel_amp_min=VEL_AMP_MIN_DEFAULT,
                                vel_amp_curve=VEL_AMP_CURVE_DEFAULT,
                                np_rng=np_rng,
                            )

                            fname = f"oneshot_{midi_to_name(rootNote)}_v{layer_idx}_rr{rr}.wav"
                            out_path = os.path.join(ones_dir, fname)
                            self._write_wav(out_path, audio)

                            done += 1
                            if done % 10 == 0:
                                self.progress.emit(int(100 * done / total))

            # Legato transitions
            if self.make_legato:
                self.log.emit("\n=== Render legato transitions (true legato) ===")
                for _loNote, _hiNote, rootNote in bounds:
                    for layer_idx, (loVel, hiVel, vel_norm) in enumerate(layers, start=1):
                        for rr in range(1, self.rr_count + 1):
                            for semi in legato_intervals:
                                src = int(rootNote - int(semi))
                                if src < self.low_midi or src > self.high_midi:
                                    done += 1
                                    continue

                                glide_t = glide_time_for_interval(int(semi)) * float(self.legato_speed_factor)
                                seconds = float(glide_t + max(0.12, self.legato_wav_release))

                                key = (rootNote * 3000 + layer_idx * 300 + rr * 10 + (semi + 24)) & 0xFFFFFFFF
                                base_seed = (2222222 if self.seed == 0 else self.seed) ^ key
                                np_rng = np.random.default_rng(base_seed)

                                audio = render_synth(
                                    wt_mips=wt_mips,
                                    patch=patch,
                                    midi=rootNote,
                                    vel_norm=vel_norm,
                                    sr=self.sr,
                                    seconds=seconds,
                                    attack=float(self.legato_wav_attack),
                                    release=float(self.legato_wav_release),
                                    rr_index=rr,
                                    rr_count=self.rr_count,
                                    detune_cents_rr=self.detune_cents_rr,
                                    unison_voices=max(1, self.unison_voices),
                                    unison_spread_cents=self.unison_spread_cents,
                                    sat_amount=self.sat_amount,
                                    noise_level=self.noise_level,
                                    vel_brightness=self.vel_brightness,
                                    target_peak=self.target_peak,
                                    vel_amp_min=VEL_AMP_MIN_DEFAULT,
                                    vel_amp_curve=VEL_AMP_CURVE_DEFAULT,
                                    np_rng=np_rng,
                                    glide_from_hz=midi_to_hz(src),
                                    glide_time=glide_t,
                                )

                                fname = f"leg_{midi_to_name(src)}_to_{midi_to_name(rootNote)}_v{layer_idx}_rr{rr}_i{semi:+d}.wav"
                                out_path = os.path.join(samples_dir, fname)
                                self._write_wav(out_path, audio)

                                legato_nodes.append({
                                    "path": _posix_rel(f"Samples/{fname}"),
                                    "rootNote": str(rootNote),
                                    "loNote": str(rootNote),
                                    "hiNote": str(rootNote),
                                    "loVel": str(loVel),
                                    "hiVel": str(hiVel),
                                    "seqMode": "round_robin",
                                    "seqPosition": str(rr),
                                    "seqLength": str(self.rr_count),
                                    "legatoInterval": str(int(semi)),
                                })

                                done += 1
                                if done % 20 == 0:
                                    self.progress.emit(int(100 * done / total))

            # Preset final
            dsp = build_dspreset(
                instrument_name=self.instrument_name,
                sustain_samples=sustain_nodes,
                legato_samples=(legato_nodes if legato_nodes else None),
                legato_tail_samples=(legato_tail_nodes if (self.make_legato and legato_tail_nodes) else None),
            )
            preset_path = os.path.join(self.out_dir, f"{self.instrument_name}.dspreset")
            with open(preset_path, "w", encoding="utf-8") as f:
                f.write(dsp)

            self.progress.emit(100)
            self.log.emit("\n✅ Listo!")
            self.log.emit(f"Preset: {preset_path}")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# UI
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WT Autosampler → Decent Sampler (3 wavetables / sesión)")
        self.resize(1040, 780)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # --- Paths ---
        gb_paths = QGroupBox("Rutas")
        p = QVBoxLayout(gb_paths)

        self.wt_dir = QLineEdit(DEFAULT_WT_DIR)
        btn_wt = QPushButton("Elegir wavetables…")
        btn_wt.clicked.connect(self.pick_wt_dir)

        row_wt = QHBoxLayout()
        row_wt.addWidget(QLabel("Carpeta wavetables:"))
        row_wt.addWidget(self.wt_dir, stretch=1)
        row_wt.addWidget(btn_wt)
        p.addLayout(row_wt)

        self.out_dir = QLineEdit("")
        btn_out = QPushButton("Elegir output…")
        btn_out.clicked.connect(self.pick_out_dir)

        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Carpeta output:"))
        row_out.addWidget(self.out_dir, stretch=1)
        row_out.addWidget(btn_out)
        p.addLayout(row_out)

        self.name_edit = QLineEdit("MyWTInstrument")
        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("Nombre instrumento:"))
        row_name.addWidget(self.name_edit, stretch=1)
        p.addLayout(row_name)

        layout.addWidget(gb_paths)

        # --- Session / Sampling ---
        gb_sampling = QGroupBox("Sesión + muestreo")
        s = QVBoxLayout(gb_sampling)

        row0 = QHBoxLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(12345)

        self.sr_spin = QSpinBox()
        self.sr_spin.setRange(8000, 192000)
        self.sr_spin.setValue(SR_DEFAULT)

        row0.addWidget(QLabel("Seed (0 = random total):"))
        row0.addWidget(self.seed_spin)
        row0.addSpacing(12)
        row0.addWidget(QLabel("Sample rate:"))
        row0.addWidget(self.sr_spin)
        row0.addStretch()
        s.addLayout(row0)

        row1 = QHBoxLayout()
        self.low_midi = QSpinBox()
        self.low_midi.setRange(0, 127)
        self.low_midi.setValue(LOW_MIDI_DEFAULT)
        self.high_midi = QSpinBox()
        self.high_midi.setRange(0, 127)
        self.high_midi.setValue(HIGH_MIDI_DEFAULT)
        self.step = QSpinBox()
        self.step.setRange(1, 12)
        self.step.setValue(NOTE_STEP_DEFAULT)

        row1.addWidget(QLabel("MIDI low:"))
        row1.addWidget(self.low_midi)
        row1.addSpacing(8)
        row1.addWidget(QLabel("high:"))
        row1.addWidget(self.high_midi)
        row1.addSpacing(12)
        row1.addWidget(QLabel("Step (semitonos):"))
        row1.addWidget(self.step)
        row1.addStretch()
        s.addLayout(row1)

        row2 = QHBoxLayout()
        self.vel_layers = QSpinBox()
        self.vel_layers.setRange(2, 12)
        self.vel_layers.setValue(VEL_LAYERS_DEFAULT)
        self.rr = QSpinBox()
        self.rr.setRange(1, 8)
        self.rr.setValue(RR_DEFAULT)

        row2.addWidget(QLabel("Velocity layers:"))
        row2.addWidget(self.vel_layers)
        row2.addSpacing(12)
        row2.addWidget(QLabel("Round robins:"))
        row2.addWidget(self.rr)
        row2.addStretch()
        s.addLayout(row2)

        layout.addWidget(gb_sampling)

        # --- Sustain + Loop ---
        gb_sus = QGroupBox("Sustain + Looping")
        su = QVBoxLayout(gb_sus)

        row3 = QHBoxLayout()
        self.sus_attack = QDoubleSpinBox()
        self.sus_attack.setRange(0.001, 1.0)
        self.sus_attack.setSingleStep(0.005)
        self.sus_attack.setValue(SUS_ATTACK_DEFAULT)

        self.sus_hold = QDoubleSpinBox()
        self.sus_hold.setRange(0.2, 30.0)
        self.sus_hold.setSingleStep(0.25)
        self.sus_hold.setValue(SUS_HOLD_DEFAULT)

        self.sus_release = QDoubleSpinBox()
        self.sus_release.setRange(0.01, 5.0)
        self.sus_release.setSingleStep(0.05)
        self.sus_release.setValue(SUS_RELEASE_DEFAULT)

        row3.addWidget(QLabel("Attack (s):"))
        row3.addWidget(self.sus_attack)
        row3.addSpacing(10)
        row3.addWidget(QLabel("Hold (s):"))
        row3.addWidget(self.sus_hold)
        row3.addSpacing(10)
        row3.addWidget(QLabel("Release (s):"))
        row3.addWidget(self.sus_release)
        row3.addStretch()
        su.addLayout(row3)

        row4 = QHBoxLayout()
        self.loop_xfade = QSpinBox()
        self.loop_xfade.setRange(0, 20000)
        self.loop_xfade.setValue(LOOP_XFADE_DEFAULT)

        self.loop_min = QDoubleSpinBox()
        self.loop_min.setRange(0.05, 5.0)
        self.loop_min.setSingleStep(0.05)
        self.loop_min.setValue(LOOP_MIN_SEC_DEFAULT)

        self.loop_max = QDoubleSpinBox()
        self.loop_max.setRange(0.10, 10.0)
        self.loop_max.setSingleStep(0.10)
        self.loop_max.setValue(LOOP_MAX_SEC_DEFAULT)

        row4.addWidget(QLabel("Loop crossfade (samples):"))
        row4.addWidget(self.loop_xfade)
        row4.addSpacing(12)
        row4.addWidget(QLabel("Loop min (s):"))
        row4.addWidget(self.loop_min)
        row4.addSpacing(8)
        row4.addWidget(QLabel("max (s):"))
        row4.addWidget(self.loop_max)
        row4.addStretch()
        su.addLayout(row4)

        layout.addWidget(gb_sus)

        # --- Legato ---
        gb_leg = QGroupBox("True Legato / Portamento")
        lg = QVBoxLayout(gb_leg)

        row5 = QHBoxLayout()
        self.legato_check = QCheckBox("Generar transiciones legato (true legato)")
        self.legato_check.setChecked(LEGATO_ENABLE_DEFAULT)

        self.legato_preset = QComboBox()
        self.legato_preset.addItems(list(LEGATO_SET.keys()))
        self.legato_preset.setCurrentText("Normal (±1..±5, ±7, ±12)")

        self.legato_speed = QComboBox()
        self.legato_speed.addItems(list(LEGATO_SPEEDS.keys()))
        self.legato_speed.setCurrentText(LEGATO_SPEED_DEFAULT)

        row5.addWidget(self.legato_check)
        row5.addSpacing(10)
        row5.addWidget(QLabel("Set intervalos:"))
        row5.addWidget(self.legato_preset, stretch=1)
        row5.addSpacing(10)
        row5.addWidget(QLabel("Speed:"))
        row5.addWidget(self.legato_speed)
        row5.addStretch()
        lg.addLayout(row5)

        # Legato WAV envelope (attack/release) for rendered transitions
        row5b = QHBoxLayout()
        self.legato_wav_attack = QDoubleSpinBox()
        self.legato_wav_attack.setRange(0.001, 1.0)
        self.legato_wav_attack.setSingleStep(0.001)
        self.legato_wav_attack.setDecimals(4)
        self.legato_wav_attack.setValue(LEGATO_WAV_ATTACK_DEFAULT)

        self.legato_wav_release = QDoubleSpinBox()
        self.legato_wav_release.setRange(0.02, 2.5)
        self.legato_wav_release.setSingleStep(0.01)
        self.legato_wav_release.setDecimals(3)
        self.legato_wav_release.setValue(LEGATO_WAV_RELEASE_DEFAULT)

        row5b.addWidget(QLabel("Legato WAV Attack (s):"))
        row5b.addWidget(self.legato_wav_attack)
        row5b.addSpacing(12)
        row5b.addWidget(QLabel("Legato WAV Release (s):"))
        row5b.addWidget(self.legato_wav_release)
        row5b.addStretch()
        lg.addLayout(row5b)

        layout.addWidget(gb_leg)

        # --- Extras: oneshots ---
        gb_extra = QGroupBox("Extras")
        ex = QHBoxLayout(gb_extra)

        self.oneshot_check = QCheckBox("Generar one-shots extra (no mapeados al preset)")
        self.oneshot_check.setChecked(False)

        self.oneshot_len = QDoubleSpinBox()
        self.oneshot_len.setRange(0.1, 5.0)
        self.oneshot_len.setSingleStep(0.05)
        self.oneshot_len.setValue(0.85)

        ex.addWidget(self.oneshot_check)
        ex.addSpacing(10)
        ex.addWidget(QLabel("Duración one-shot (s):"))
        ex.addWidget(self.oneshot_len)
        ex.addStretch()
        layout.addWidget(gb_extra)

        # --- Synth feel ---
        gb_synth = QGroupBox("Carácter del synth (realismo)")
        sy = QVBoxLayout(gb_synth)

        row6 = QHBoxLayout()
        self.detune = QDoubleSpinBox()
        self.detune.setRange(0.0, 50.0)
        self.detune.setSingleStep(0.5)
        self.detune.setValue(DETUNE_CENTS_RR_DEFAULT)

        self.unison = QSpinBox()
        self.unison.setRange(1, 8)
        self.unison.setValue(UNISON_VOICES_DEFAULT)

        self.spread = QDoubleSpinBox()
        self.spread.setRange(0.0, 50.0)
        self.spread.setSingleStep(0.5)
        self.spread.setValue(UNISON_SPREAD_CENTS_DEFAULT)

        row6.addWidget(QLabel("Detune RR (cents):"))
        row6.addWidget(self.detune)
        row6.addSpacing(10)
        row6.addWidget(QLabel("Unison voices:"))
        row6.addWidget(self.unison)
        row6.addSpacing(10)
        row6.addWidget(QLabel("Unison spread (cents):"))
        row6.addWidget(self.spread)
        row6.addStretch()
        sy.addLayout(row6)

        row7 = QHBoxLayout()
        self.sat = QDoubleSpinBox()
        self.sat.setRange(0.0, 1.0)
        self.sat.setSingleStep(0.05)
        self.sat.setValue(SAT_AMOUNT_DEFAULT)

        self.noise = QDoubleSpinBox()
        self.noise.setRange(0.0, 0.05)
        self.noise.setSingleStep(0.0005)
        self.noise.setDecimals(4)
        self.noise.setValue(NOISE_LEVEL_DEFAULT)

        self.vel_bright = QDoubleSpinBox()
        self.vel_bright.setRange(0.2, 3.0)
        self.vel_bright.setSingleStep(0.1)
        self.vel_bright.setValue(VEL_BRIGHTNESS_DEFAULT)

        self.peak = QDoubleSpinBox()
        self.peak.setRange(0.1, 1.0)
        self.peak.setSingleStep(0.05)
        self.peak.setValue(TARGET_PEAK_DEFAULT)

        row7.addWidget(QLabel("Saturation (0..1):"))
        row7.addWidget(self.sat)
        row7.addSpacing(10)
        row7.addWidget(QLabel("Noise:"))
        row7.addWidget(self.noise)
        row7.addSpacing(10)
        row7.addWidget(QLabel("Vel brightness:"))
        row7.addWidget(self.vel_bright)
        row7.addSpacing(10)
        row7.addWidget(QLabel("Target peak:"))
        row7.addWidget(self.peak)
        row7.addStretch()
        sy.addLayout(row7)

        layout.addWidget(gb_synth)

        # --- Run ---
        self.btn_run = QPushButton("Generar librería")
        self.btn_run.clicked.connect(self.start)
        layout.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        layout.addWidget(self.logs, stretch=1)

        layout.addItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        footer = QLabel("Decent Sampler Autosampler (3-wavetable session)")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

        self.thread: Optional[QThread] = None
        self.worker: Optional[BuildWorker] = None

    def log(self, msg: str):
        self.logs.append(msg)

    def pick_wt_dir(self):
        start = self.wt_dir.text().strip() or os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de wavetables", start)
        if folder:
            self.wt_dir.setText(folder)

    def pick_out_dir(self):
        start = self.out_dir.text().strip() or os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de salida", start)
        if folder:
            self.out_dir.setText(folder)

    def start(self):
        wt_dir = self.wt_dir.text().strip()
        out_dir = self.out_dir.text().strip()
        name = self.name_edit.text().strip()

        if not wt_dir or not os.path.isdir(wt_dir):
            QMessageBox.warning(self, "Wavetables", "La carpeta de wavetables no existe.")
            return
        if not out_dir:
            QMessageBox.warning(self, "Output", "Elige una carpeta de salida.")
            return
        if not name:
            QMessageBox.warning(self, "Nombre", "Define un nombre de instrumento.")
            return

        low = int(self.low_midi.value())
        high = int(self.high_midi.value())
        if high <= low:
            QMessageBox.warning(self, "MIDI range", "MIDI high debe ser mayor que MIDI low.")
            return

        loop_min = float(self.loop_min.value())
        loop_max = float(self.loop_max.value())
        if loop_max < loop_min:
            loop_min, loop_max = loop_max, loop_min

        # UI reset
        self.logs.clear()
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)

        self.thread = QThread()
        self.worker = BuildWorker(
            out_dir=out_dir,
            wt_dir=wt_dir,
            instrument_name=name,
            seed=int(self.seed_spin.value()),
            sample_rate=int(self.sr_spin.value()),
            low_midi=low,
            high_midi=high,
            note_step=int(self.step.value()),
            vel_layers=int(self.vel_layers.value()),
            rr_count=int(self.rr.value()),
            sus_attack=float(self.sus_attack.value()),
            sus_hold=float(self.sus_hold.value()),
            sus_release=float(self.sus_release.value()),
            loop_xfade=int(self.loop_xfade.value()),
            loop_min_sec=loop_min,
            loop_max_sec=loop_max,
            detune_cents_rr=float(self.detune.value()),
            unison_voices=int(self.unison.value()),
            unison_spread_cents=float(self.spread.value()),
            sat_amount=float(self.sat.value()),
            noise_level=float(self.noise.value()),
            vel_brightness=float(self.vel_bright.value()),
            target_peak=float(self.peak.value()),
            make_legato=bool(self.legato_check.isChecked()),
            legato_preset_name=str(self.legato_preset.currentText()),
            legato_speed_factor=float(LEGATO_SPEEDS[self.legato_speed.currentText()]),
            legato_wav_attack=float(self.legato_wav_attack.value()),
            legato_wav_release=float(self.legato_wav_release.value()),
            keep_oneshots=bool(self.oneshot_check.isChecked()),
            oneshot_seconds=float(self.oneshot_len.value()),
        )

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_done)
        self.worker.error.connect(self.on_err)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.start()

    def on_done(self):
        self.btn_run.setEnabled(True)
        QMessageBox.information(self, "Listo", "Librería generada.")

    def on_err(self, msg: str):
        self.btn_run.setEnabled(True)
        self.log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg)


def main():
    app = QApplication(sys.argv)
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6"))
    except Exception:
        pass
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
