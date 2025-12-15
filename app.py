import os, glob, math, argparse, sys, traceback
import re, unicodedata
import numpy as np
import soundfile as sf

# =======================
# PyInstaller/--windowed safety:
# evita crash de argparse cuando sys.stderr/sys.stdout son None
# =======================
if getattr(sys, "frozen", False):
    try:
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w")
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")
    except Exception:
        pass

# =======================
# WAVETABLE CORE (offline)
# =======================
WT_FRAME_SIZE = 2048
WT_MIP_LEVELS = 8

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
    audio, _sr = sf.read(path, always_2d=False)
    audio = _to_mono_float(audio)

    if len(audio) < frame_size:
        audio = np.pad(audio, (0, frame_size - len(audio)))
    rem = len(audio) % frame_size
    if rem != 0:
        audio = np.pad(audio, (0, frame_size - rem))

    n_frames = max(1, len(audio) // frame_size)
    audio = audio[: n_frames * frame_size]
    frames = audio.reshape(n_frames, frame_size).copy()

    for i in range(n_frames):
        f = frames[i]
        f = f - np.mean(f)
        f = _fade_edges(f, edge_fade)
        if normalize_each_frame:
            f = _normalize_frame(f)
        frames[i] = f

    return frames.astype(np.float32, copy=False)

def build_wavetable_mipmaps(frames: np.ndarray, levels: int = WT_MIP_LEVELS):
    frames = np.asarray(frames, dtype=np.float32)
    n_frames, frame_size = frames.shape
    mipmaps = []
    cur = frames
    cur_size = frame_size
    for _lvl in range(levels):
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

def _lerp(a, b, t):
    return a + (b - a) * t

def _table_read_linear(table_1d: np.ndarray, phase: np.ndarray) -> np.ndarray:
    n = len(table_1d)
    idx = phase * n
    i0 = np.floor(idx).astype(np.int32)
    frac = idx - i0
    i1 = (i0 + 1) % n
    return (1.0 - frac) * table_1d[i0] + frac * table_1d[i1]

# ==========================================================
# Interpolación mejor: Cubic/Hermite (Catmull-Rom) 4 puntos
# ==========================================================
def _table_read_cubic_hermite(table_1d: np.ndarray, phase: np.ndarray) -> np.ndarray:
    n = len(table_1d)
    idx = phase * n
    i1 = np.floor(idx).astype(np.int32)
    t = (idx - i1).astype(np.float32)

    i0 = (i1 - 1) % n
    i2 = (i1 + 1) % n
    i3 = (i1 + 2) % n

    p0 = table_1d[i0]
    p1 = table_1d[i1]
    p2 = table_1d[i2]
    p3 = table_1d[i3]

    a = (-0.5 * p0) + (1.5 * p1) - (1.5 * p2) + (0.5 * p3)
    b = (p0) - (2.5 * p1) + (2.0 * p2) - (0.5 * p3)
    c = (-0.5 * p0) + (0.5 * p2)
    d = p1

    return (((a * t + b) * t + c) * t + d).astype(np.float32, copy=False)

# ==========================================================
# Derivative mip + smoothing + mip por bloques
# ==========================================================
def render_wavetable_osc_f0_array(
    f0_hz: np.ndarray,
    sr: int,
    mipmaps: list,
    position: float,
    phase0: float,
    mip_strength: float,
    block_size: int = 256,
    mip_smooth_tau: float = 0.030,
):
    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    n = int(len(f0_hz))
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    base_frames = mipmaps[0]
    n_frames = base_frames.shape[0]

    pos = float(np.clip(position, 0.0, 1.0))
    fidx = pos * (n_frames - 1)
    f0i = int(np.floor(fidx))
    ft = float(fidx - f0i)
    f1i = min(f0i + 1, n_frames - 1)

    inc = (f0_hz / float(sr)).astype(np.float32)  # cycles/sample
    c = np.cumsum(inc, dtype=np.float64)
    phase = (float(phase0) + np.concatenate(([0.0], c[:-1].astype(np.float32)))) % 1.0

    n_levels = len(mipmaps)
    out = np.zeros(n, dtype=np.float32)

    inc_ref = max(55.0 / float(sr), 1e-12)
    ms = float(np.clip(mip_strength, 0.0, 1.0))

    bs = int(max(16, block_size))
    nb = int((n + bs - 1) // bs)

    tau = float(max(1e-6, mip_smooth_tau))
    g = 1.0 - math.exp(-float(bs) / (tau * float(sr)))

    lvl_sm = None
    table_cache: dict[int, np.ndarray] = {}

    for bi in range(nb):
        i = bi * bs
        j = min(n, i + bs)

        inc_blk = float(np.max(inc[i:j]) + 1e-12)
        ratio = max(inc_blk / inc_ref, 1e-6)
        lvl = (math.log2(ratio) * ms)

        if lvl_sm is None:
            lvl_sm = lvl
        else:
            lvl_sm = lvl_sm + g * (lvl - lvl_sm)

        L0 = int(np.clip(math.floor(lvl_sm), 0, n_levels - 1))
        frac = float(np.clip(lvl_sm - float(L0), 0.0, 1.0))
        L1 = int(min(L0 + 1, n_levels - 1))

        ph = phase[i:j]

        if L0 not in table_cache:
            tables_L0 = mipmaps[L0]
            table_cache[L0] = _lerp(tables_L0[f0i], tables_L0[f1i], ft).astype(np.float32, copy=False)
        if L1 not in table_cache:
            tables_L1 = mipmaps[L1]
            table_cache[L1] = _lerp(tables_L1[f0i], tables_L1[f1i], ft).astype(np.float32, copy=False)

        y0 = _table_read_cubic_hermite(table_cache[L0], ph)
        if L1 == L0 or frac <= 1e-6:
            out[i:j] = y0
        else:
            y1 = _table_read_cubic_hermite(table_cache[L1], ph)
            out[i:j] = (y0 * (1.0 - frac) + y1 * frac).astype(np.float32, copy=False)

    return out

# =======================
# MUSICAL / FILE UTILS
# =======================
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def note_name_to_hz(note: str) -> float:
    note = note.strip().upper().replace(" ", "")
    if len(note) < 2:
        raise ValueError(f"Nota inválida: {note}")
    if note[1] == "#":
        nn = note[:2]
        oct_str = note[2:]
    else:
        nn = note[:1]
        oct_str = note[1:]
    if nn not in NOTE_NAMES:
        raise ValueError(f"Nota inválida: {note}")
    octv = int(oct_str)
    semitone = NOTE_NAMES.index(nn)
    midi = (octv + 1) * 12 + semitone
    hz = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    return float(hz)

def list_wav_files(folder: str):
    files = glob.glob(os.path.join(folder, "**/*.wav"), recursive=True)
    files += glob.glob(os.path.join(folder, "**/*.WAV"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=lambda p: p.lower())
    return files

def safe_name(s: str) -> str:
    return s.replace("#","s").replace("-","m")

def compact_token_from_filename(path: str) -> str:
    # basename sin extensión
    s = os.path.splitext(os.path.basename(path))[0]
    # quita acentos / caracteres raros → ascii
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    # solo letras/números, todo junto
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))

def exp_decay(t: np.ndarray, tau: float) -> np.ndarray:
    return np.exp(-t / float(tau)).astype(np.float32)

# ==========================================
# Dither TPDF (solo si exportas PCM_*)
# ==========================================
def add_tpdf_dither(y: np.ndarray, bits: int, rng: np.random.Generator):
    lsb = 1.0 / (2 ** (bits - 1))
    d = (rng.random(len(y), dtype=np.float32) - rng.random(len(y), dtype=np.float32)) * lsb
    return (y + d).astype(np.float32, copy=False)

def _bits_for_subtype(subtype: str) -> int | None:
    st = subtype.upper()
    if st == "PCM_16":
        return 16
    if st == "PCM_24":
        return 24
    if st == "PCM_32":
        return 32
    return None

# ==========================================
# Humanize determinístico
# ==========================================
def stable_hash32(s: str) -> int:
    h = 2166136261
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h

def hash_float01(a: int) -> float:
    x = (int(a) * 1103515245 + 12345) & 0x7fffffff
    return x / 0x7fffffff

# ==========================================
# ADSR helpers
# ==========================================
def adsr_sustain_env(n: int, sr: int, v: float) -> np.ndarray:
    v = float(np.clip(v, 0.0, 1.0))
    atk = np.interp(v, [0.0, 1.0], [0.120, 0.004])
    dec = np.interp(v, [0.0, 1.0], [1.40, 0.45])
    sus = np.interp(v, [0.0, 1.0], [0.40, 0.90])

    aN = int(atk * sr)
    dN = int(dec * sr)
    sN = max(0, n - (aN + dN))

    env = np.zeros(n, dtype=np.float32)
    i = 0
    if aN > 0:
        a = np.linspace(0.0, 1.0, aN, endpoint=False, dtype=np.float32)
        env[i:i+aN] = a ** np.interp(v, [0,1], [1.8, 0.7])
        i += aN
    if dN > 0:
        env[i:i+dN] = np.linspace(1.0, sus, dN, endpoint=False, dtype=np.float32)
        i += dN
    if sN > 0:
        env[i:i+sN] = sus
    return env

def release_env(n: int, sr: int, v: float) -> np.ndarray:
    v = float(np.clip(v, 0.0, 1.0))
    tau = float(np.interp(v, [0.0, 1.0], [2.2, 1.2]))
    t = (np.arange(n, dtype=np.float32) / float(sr))
    return exp_decay(t, tau=tau).astype(np.float32, copy=False)

# ==========================================
# Looping con crossfade
# ==========================================
def apply_loop_crossfade(y: np.ndarray, sr: int, loop_start_s: float, loop_end_s: float, xfade_s: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    ls = int(np.clip(loop_start_s * sr, 0, n-1))
    le = int(np.clip(loop_end_s * sr, 0, n))
    xf = int(max(0, xfade_s * sr))
    if xf < 8:
        return y
    if le <= ls + xf + 8:
        return y

    A = y[ls:ls+xf].copy()
    B = y[le-xf:le].copy()

    fade = np.linspace(0.0, 1.0, xf, endpoint=False, dtype=np.float32)
    cross = (B * (1.0 - fade) + A * fade).astype(np.float32)

    y2 = y.copy()
    y2[ls:ls+xf] = cross
    y2[le-xf:le] = cross
    return y2

# ==========================================
# Body EQ barato
# ==========================================
def cheap_body_eq(y: np.ndarray, sr: int, kt: float, v: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    k_fast = int(np.clip(np.interp(kt, [0,1], [9, 5]), 5, 21))
    k_slow = int(np.clip(np.interp(kt, [0,1], [61, 33]), 21, 129))

    w_fast = np.ones(k_fast, dtype=np.float32) / float(k_fast)
    w_slow = np.ones(k_slow, dtype=np.float32) / float(k_slow)

    lp_fast = np.convolve(y, w_fast, mode="same").astype(np.float32, copy=False)
    lp_slow = np.convolve(y, w_slow, mode="same").astype(np.float32, copy=False)

    band = (lp_fast - lp_slow).astype(np.float32, copy=False)

    body_g = float(np.interp(v, [0,1], [0.22, 0.08]))
    y2 = (y + body_g * band).astype(np.float32, copy=False)

    k_hp = int(np.clip(np.interp(kt, [0,1], [15, 33]), 9, 65))
    w_hp = np.ones(k_hp, dtype=np.float32) / float(k_hp)
    low = np.convolve(y2, w_hp, mode="same").astype(np.float32, copy=False)
    high = (y2 - low).astype(np.float32, copy=False)

    shelf = float(np.interp(kt, [0,1], [1.0, 0.78]))
    return (low + shelf * high).astype(np.float32, copy=False)

# ==========================================
# Bandpass noise keytracked
# ==========================================
def bandpass_noise_keytracked(hp_noise: np.ndarray, sr: int, kt: float) -> np.ndarray:
    hp_noise = np.asarray(hp_noise, dtype=np.float32)

    k_fast = int(np.clip(np.interp(kt, [0,1], [13, 5]), 5, 25))
    k_slow = int(np.clip(np.interp(kt, [0,1], [81, 29]), 21, 161))

    w_fast = np.ones(k_fast, dtype=np.float32) / float(k_fast)
    w_slow = np.ones(k_slow, dtype=np.float32) / float(k_slow)

    lp_fast = np.convolve(hp_noise, w_fast, mode="same").astype(np.float32, copy=False)
    lp_slow = np.convolve(hp_noise, w_slow, mode="same").astype(np.float32, copy=False)
    return (lp_fast - lp_slow).astype(np.float32, copy=False)

# =======================
# SYNTH
# =======================
def synth_sustain_one_shot(
    mip_low, mip_mid, mip_high,
    f0_base: float,
    sr: int,
    n: int,
    v: float,
    base_pos, base_mip, base_phase,
    note_keytrack: float,
    note_name: str,
    vel_index: int,
    rr_index: int,
    rng_noise: np.random.Generator,
    mip_block: int,
    mip_smooth_tau: float,
):
    v = float(np.clip(v, 0.0, 1.0))
    kt = float(np.clip(note_keytrack, 0.0, 1.0))
    t = (np.arange(n, dtype=np.float32) / float(sr))

    knee = 0.71
    knee_part = float(np.clip((v - knee) / max(1e-6, (1.0 - knee)), 0.0, 1.0))
    v_curve = v ** 1.45
    amp_db = np.interp(v_curve, [0.0, 0.75, 1.0], [-34.0, -12.5, -9.5])
    amp_db -= 1.5 * knee_part
    amp = db_to_lin(float(amp_db))

    pitch_attack_cents = (2.0 + 10.0*(v**0.9)) * (1.0 + 0.25*kt)
    pitch_env = pitch_attack_cents * exp_decay(t, tau=np.interp(v, [0,1], [0.075, 0.040]))

    vib_rate = np.interp(v, [0,1], [4.2, 5.6])
    vib_depth_cents = np.interp(v, [0,1], [0.2, 1.2]) * (1.0 + 0.15*knee_part)
    h_vib = stable_hash32(f"{note_name}|{vel_index}|RR{rr_index}|VIBPH")
    vib_phase = 2.0 * np.pi * hash_float01(h_vib)
    vibrato = vib_depth_cents * np.sin(2.0*np.pi*vib_rate*t + vib_phase).astype(np.float32)

    cents_total = (pitch_env + vibrato).astype(np.float32)
    f0 = (f0_base * (2.0 ** (cents_total / 1200.0))).astype(np.float32)

    # detune por oscilador
    h = stable_hash32(f"{note_name}|{vel_index}|RR{rr_index}|{int(kt*1000)}|{int(v*10000)}")
    base_det = (hash_float01(h) * 2.0 - 1.0)

    det_low_c  = 0.30 * base_det
    det_mid_c  = 0.90 * base_det + 0.35
    det_high_c = 1.30 * base_det - 0.55

    spread = np.interp(v, [0,1], [0.85, 1.15])
    det_low_c  *= spread
    det_mid_c  *= spread
    det_high_c *= spread

    f0_low  = (f0 * (2.0 ** (det_low_c  / 1200.0))).astype(np.float32)
    f0_mid  = (f0 * (2.0 ** (det_mid_c  / 1200.0))).astype(np.float32)
    f0_high = (f0 * (2.0 ** (det_high_c / 1200.0))).astype(np.float32)

    pos_low  = float(np.clip(base_pos[0] + 0.08*(v-0.5) + 0.05*(kt-0.5), 0.0, 1.0))
    pos_mid  = float(np.clip(base_pos[1] + 0.11*(v-0.5) + 0.06*(kt-0.5), 0.0, 1.0))
    pos_high = float(np.clip(base_pos[2] + 0.16*(v-0.5) + 0.08*(kt-0.5), 0.0, 1.0))

    ph_off0 = hash_float01(stable_hash32(f"{note_name}|{vel_index}|RR{rr_index}|PH0")) * 0.25
    ph_off1 = hash_float01(stable_hash32(f"{note_name}|{vel_index}|RR{rr_index}|PH1")) * 0.25
    ph_off2 = hash_float01(stable_hash32(f"{note_name}|{vel_index}|RR{rr_index}|PH2")) * 0.25
    ph0 = float((base_phase[0] + ph_off0) % 1.0)
    ph1 = float((base_phase[1] + ph_off1) % 1.0)
    ph2 = float((base_phase[2] + ph_off2) % 1.0)

    y0 = render_wavetable_osc_f0_array(
        f0_low * 1.0, sr, mip_low,
        position=pos_low, phase0=ph0, mip_strength=base_mip[0],
        block_size=mip_block, mip_smooth_tau=mip_smooth_tau,
    )
    y1 = render_wavetable_osc_f0_array(
        f0_mid * 2.0, sr, mip_mid,
        position=pos_mid, phase0=ph1, mip_strength=base_mip[1],
        block_size=mip_block, mip_smooth_tau=mip_smooth_tau,
    )
    y2 = render_wavetable_osc_f0_array(
        f0_high * 4.0, sr, mip_high,
        position=pos_high, phase0=ph2, mip_strength=base_mip[2],
        block_size=mip_block, mip_smooth_tau=mip_smooth_tau,
    )

    phase_end_low  = float((ph0 + float(np.sum((f0_low  * 1.0) / float(sr), dtype=np.float64))) % 1.0)
    phase_end_mid  = float((ph1 + float(np.sum((f0_mid  * 2.0) / float(sr), dtype=np.float64))) % 1.0)
    phase_end_high = float((ph2 + float(np.sum((f0_high * 4.0) / float(sr), dtype=np.float64))) % 1.0)

    bright = (v**0.65) * (0.85 + 0.30*kt)
    w_low  = float(np.clip(0.80 - 0.22*bright, 0.40, 0.85))
    w_mid  = float(np.clip(0.28 + 0.40*bright, 0.20, 0.75))
    w_high = float(np.clip(0.02 + 0.70*(bright**1.20), 0.00, 0.70))

    fenv_amt = np.interp(v, [0,1], [0.20, 0.85]) * (0.8 + 0.3*kt)
    fenv_amt *= (1.0 - 0.20*kt)
    fenv_amt *= (1.0 + 0.20*knee_part)
    fenv = (fenv_amt * exp_decay(t, tau=np.interp(v, [0,1], [0.20, 0.08]))).astype(np.float32)

    w_high_t = np.clip(w_high + 0.35 * fenv, 0.0, 0.9).astype(np.float32)
    w_mid_t  = np.clip(w_mid  + 0.10 * fenv, 0.0, 0.9).astype(np.float32)
    w_low_t  = np.clip(w_low  - 0.12 * fenv, 0.0, 0.9).astype(np.float32)

    # micro movimiento
    h2 = stable_hash32(f"{note_name}|{vel_index}|RR{rr_index}|LFO")
    rate = np.interp(hash_float01(h2), [0,1], [0.18, 0.55])
    phase_lfo = 2.0 * np.pi * hash_float01(h2 ^ 0xA5A5A5A5)
    lfo = np.sin(2.0*np.pi*rate*t + phase_lfo).astype(np.float32)

    depth = np.interp(v, [0,1], [0.06, 0.015]) * (0.85 + 0.25*kt)
    env_sus = adsr_sustain_env(n, sr, v)
    mod = (1.0 + depth * lfo * (env_sus**0.55)).astype(np.float32)

    y1 = (y1 * mod).astype(np.float32, copy=False)
    y2 = (y2 * (1.0 + 0.5*depth*lfo*(env_sus**0.55))).astype(np.float32, copy=False)

    y = (w_low_t*y0 + w_mid_t*y1 + w_high_t*y2).astype(np.float32, copy=False)

    # transients
    noise = rng_noise.standard_normal(n).astype(np.float32)
    hp_noise = (noise - np.concatenate(([0.0], noise[:-1]))).astype(np.float32)
    bp_noise = bandpass_noise_keytracked(hp_noise, sr=sr, kt=kt)

    trans_len = int(np.interp(v, [0,1], [0.010, 0.030]) * sr)
    trans = np.zeros(n, dtype=np.float32)
    if trans_len > 8:
        trans_env = exp_decay(t[:trans_len], tau=np.interp(v, [0,1], [0.010, 0.018]))
        trans[:trans_len] = bp_noise[:trans_len] * trans_env * (0.7 + 0.9*fenv[:trans_len])
    trans_gain = np.interp(v, [0,1], [0.010, 0.060]) * (0.9 + 0.3*kt)
    y = (y + trans_gain * trans).astype(np.float32, copy=False)

    click_len = int(np.interp(v, [0,1], [0.0015, 0.0030]) * sr)
    if click_len > 8:
        click_env = exp_decay(t[:click_len], tau=np.interp(v, [0,1], [0.0008, 0.0016]))
        click = np.zeros(n, dtype=np.float32)
        click[:click_len] = click_env
        click_hp = click - np.concatenate(([0.0], click[:-1]))

        k = int(np.clip(np.interp(kt, [0,1], [13, 7]), 5, 21))
        click_lp = np.convolve(click_hp, np.ones(k, dtype=np.float32)/k, mode="same").astype(np.float32)

        click_gain = np.interp(v, [0,1], [0.020, 0.090]) * (0.8 + 0.35*kt)
        y = (y + click_gain * click_lp * (0.7 + 0.9*fenv)).astype(np.float32, copy=False)

    y = (y * env_sus).astype(np.float32, copy=False)

    # drive
    drive_base = np.interp(v, [0,1], [1.10, 2.70])
    drive_base *= (1.05 - 0.25*kt)
    drive_base *= (0.95 + 0.15*kt)
    drive_base *= (1.0 + 0.12*knee_part)
    drive_env = (drive_base * (1.0 + 0.90 * fenv)).astype(np.float32)
    y = np.tanh(drive_env * y).astype(np.float32, copy=False)

    floor = np.interp(v, [0,1], [0.006, 0.0015])
    y = (y + floor * bp_noise * (env_sus**0.7)).astype(np.float32, copy=False)

    y = cheap_body_eq(y, sr=sr, kt=kt, v=v)

    # RMS staging
    rms = float(np.sqrt(np.mean(y*y) + 1e-12))
    target_rms = db_to_lin(np.interp(v**1.35, [0,1], [-28.0, -10.0]))
    corr = float(np.clip(target_rms / (rms + 1e-12), 0.5, 1.8))
    y = (y * corr).astype(np.float32, copy=False)

    # final
    y = (y * amp * 0.85).astype(np.float32, copy=False)

    return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False), phase_end_low, phase_end_mid, phase_end_high

def synth_release_tail(
    mip_low, mip_mid, mip_high,
    f0_base: float,
    sr: int,
    n: int,
    v: float,
    base_pos, base_mip,
    note_keytrack: float,
    note_name: str,
    vel_index: int,
    rr_index: int,
    rng_noise: np.random.Generator,
    phase0_low: float,
    phase0_mid: float,
    phase0_high: float,
    mip_block: int,
    mip_smooth_tau: float,
):
    v = float(np.clip(v, 0.0, 1.0))
    kt = float(np.clip(note_keytrack, 0.0, 1.0))
    t = (np.arange(n, dtype=np.float32) / float(sr))

    vib_rate = np.interp(v, [0,1], [3.8, 5.0])
    vib_depth_cents = np.interp(v, [0,1], [0.10, 0.55])
    h_vib = stable_hash32(f"{note_name}|{vel_index}|RR{rr_index}|REL|VIBPH")
    vib_phase = 2.0 * np.pi * hash_float01(h_vib)
    vibrato = vib_depth_cents * np.sin(2.0*np.pi*vib_rate*t + vib_phase).astype(np.float32)
    f0 = (f0_base * (2.0 ** (vibrato / 1200.0))).astype(np.float32)

    h = stable_hash32(f"{note_name}|{vel_index}|RR{rr_index}|REL|{int(kt*1000)}|{int(v*10000)}")
    base_det = (hash_float01(h) * 2.0 - 1.0)

    det_low_c  = 0.30 * base_det
    det_mid_c  = 0.90 * base_det + 0.35
    det_high_c = 1.30 * base_det - 0.55

    spread = np.interp(v, [0,1], [0.85, 1.10])
    det_low_c  *= spread
    det_mid_c  *= spread
    det_high_c *= spread

    f0_low  = (f0 * (2.0 ** (det_low_c  / 1200.0))).astype(np.float32)
    f0_mid  = (f0 * (2.0 ** (det_mid_c  / 1200.0))).astype(np.float32)
    f0_high = (f0 * (2.0 ** (det_high_c / 1200.0))).astype(np.float32)

    pos_low  = float(np.clip(base_pos[0] + 0.08*(v-0.5) + 0.05*(kt-0.5), 0.0, 1.0))
    pos_mid  = float(np.clip(base_pos[1] + 0.11*(v-0.5) + 0.06*(kt-0.5), 0.0, 1.0))
    pos_high = float(np.clip(base_pos[2] + 0.16*(v-0.5) + 0.08*(kt-0.5), 0.0, 1.0))

    bright = (v**0.65) * (0.85 + 0.30*kt)
    w_low  = float(np.clip(0.82 - 0.18*bright, 0.45, 0.88))
    w_mid  = float(np.clip(0.28 + 0.28*bright, 0.20, 0.70))
    w_high = float(np.clip(0.02 + 0.35*(bright**1.10), 0.00, 0.40))

    fenv_amt = np.interp(v, [0,1], [0.08, 0.22]) * (0.9 - 0.2*kt)
    fenv = (fenv_amt * exp_decay(t, tau=np.interp(v, [0,1], [0.35, 0.18]))).astype(np.float32)

    w_high_t = np.clip(w_high + 0.12 * fenv, 0.0, 0.55).astype(np.float32)
    w_mid_t  = np.clip(w_mid  + 0.05 * fenv, 0.0, 0.80).astype(np.float32)
    w_low_t  = np.clip(w_low  - 0.05 * fenv, 0.0, 0.95).astype(np.float32)

    y0 = render_wavetable_osc_f0_array(
        f0_low * 1.0, sr, mip_low, position=pos_low,
        phase0=float(phase0_low), mip_strength=base_mip[0],
        block_size=mip_block, mip_smooth_tau=mip_smooth_tau,
    )
    y1 = render_wavetable_osc_f0_array(
        f0_mid * 2.0, sr, mip_mid, position=pos_mid,
        phase0=float(phase0_mid), mip_strength=base_mip[1],
        block_size=mip_block, mip_smooth_tau=mip_smooth_tau,
    )
    y2 = render_wavetable_osc_f0_array(
        f0_high * 4.0, sr, mip_high, position=pos_high,
        phase0=float(phase0_high), mip_strength=base_mip[2],
        block_size=mip_block, mip_smooth_tau=mip_smooth_tau,
    )

    y = (w_low_t*y0 + w_mid_t*y1 + w_high_t*y2).astype(np.float32, copy=False)

    env_rel = release_env(n, sr, v)
    y = (y * env_rel).astype(np.float32, copy=False)

    noise = rng_noise.standard_normal(n).astype(np.float32)
    hp_noise = (noise - np.concatenate(([0.0], noise[:-1]))).astype(np.float32)
    bp_noise = bandpass_noise_keytracked(hp_noise, sr=sr, kt=kt)
    floor = np.interp(v, [0,1], [0.0045, 0.0012])
    y = (y + floor * bp_noise * (env_rel**0.85)).astype(np.float32, copy=False)

    drive_base = np.interp(v, [0,1], [1.05, 1.85])
    drive_base *= (1.05 - 0.30*kt)
    drive_env = (drive_base * (1.0 + 0.35 * fenv)).astype(np.float32)
    y = np.tanh(drive_env * y).astype(np.float32, copy=False)

    y = cheap_body_eq(y, sr=sr, kt=kt, v=v)
    return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)

# =======================
# RENDER PIPELINE (para GUI/CLI)
# =======================
def _default_output_dir() -> str:
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
    return os.path.join(base, "Output")

def render_instrument(
    wt_dir: str,
    out_dir: str | None,
    seed: int,
    sr: int,
    duration: float,
    release_time: float,
    loop_start: float,
    loop_end: float,
    loop_xfade: float,
    bitdepth: str,
    velocity_midi: int,
    start_oct: int,
    end_oct: int,
    mip_block: int,
    mip_smooth_tau: float,
    round_robins: int,
    log_fn=None,
    progress_fn=None,
    cancel_check=None,
):
    if out_dir is None:
        out_dir = _default_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    def log(s: str):
        if log_fn:
            log_fn(s)
        else:
            print(s)

    vel_midi = int(np.clip(velocity_midi, 1, 127))
    v = float(vel_midi / 127.0)
    vel_index = 1

    rng = np.random.default_rng(None if seed == 0 else seed)

    wt_files = list_wav_files(wt_dir)
    if len(wt_files) < 3:
        raise RuntimeError(f"Necesitas al menos 3 wavetables .wav en {wt_dir}")

    wt_low, wt_mid, wt_high = rng.choice(wt_files, size=3, replace=False).tolist()
    log("Wavetables fijos para toda la sesión (low/mid/high):")
    log(f"  LOW : {os.path.basename(wt_low)}")
    log(f"  MID : {os.path.basename(wt_mid)}")
    log(f"  HIGH: {os.path.basename(wt_high)}")

    # Prefijo con los nombres de las wavetables (compacto, sin espacios, minusculas)
    prefix = (
        compact_token_from_filename(wt_low)
        + compact_token_from_filename(wt_mid)
        + compact_token_from_filename(wt_high)
    )
    if not prefix:
        prefix = "wavetables"

    mip_low  = build_wavetable_mipmaps(load_wavetable_wav(wt_low),  levels=WT_MIP_LEVELS)
    mip_mid  = build_wavetable_mipmaps(load_wavetable_wav(wt_mid),  levels=WT_MIP_LEVELS)
    mip_high = build_wavetable_mipmaps(load_wavetable_wav(wt_high), levels=WT_MIP_LEVELS)

    base_pos   = [float(rng.uniform(0.0, 1.0)) for _ in range(3)]
    base_mip   = [float(rng.uniform(0.65, 1.0)) for _ in range(3)]
    base_phase = [float(rng.uniform(0.0, 1.0)) for _ in range(3)]

    # 22 roots: C y F# por octava
    roots = []
    for o in range(int(start_oct), int(end_oct) + 1):
        roots.append(f"C{o}")
        roots.append(f"F#{o}")

    rrN = 1  # exacto 22 archivos

    n_sus = int(int(sr) * float(duration))
    n_rel = 0  # fuerza release off para no duplicar archivos

    samples_dir = os.path.join(out_dir, "Samples")
    os.makedirs(samples_dir, exist_ok=True)

    bits = _bits_for_subtype(bitdepth)

    kt_den = max(1.0, float(end_oct - start_oct))

    session_tag = (
        f"{os.path.basename(wt_low)}|{os.path.basename(wt_mid)}|{os.path.basename(wt_high)}|"
        f"{base_pos[0]:.4f},{base_pos[1]:.4f},{base_pos[2]:.4f}|"
        f"{base_mip[0]:.4f},{base_mip[1]:.4f},{base_mip[2]:.4f}|"
        f"{base_phase[0]:.4f},{base_phase[1]:.4f},{base_phase[2]:.4f}|VEL{vel_midi}"
    )

    total = len(roots) * rrN
    done = 0

    log(f"Generando roots (C y F# por octava): {len(roots)} WAVs (C{start_oct}..C{end_oct} y F#{start_oct}..F#{end_oct})")
    log(f"Velocity fija: {vel_midi} (v={v:.3f}) | SR={sr} | DUR={duration}s | BIT={bitdepth}")
    log(f"OUT: {out_dir}")
    log(f"File prefix: {prefix}")

    for note in roots:
        if cancel_check and cancel_check():
            log("Cancelado por el usuario.")
            break

        f0_base = note_name_to_hz(note)

        # keytrack por octava (C-2 y F#-2 comparten octava -2)
        if "F#" in note:
            octv = int(note[2:])
        else:
            octv = int(note[1:])
        kt = float((octv - start_oct) / kt_den)

        for rri in range(1, rrN + 1):
            if cancel_check and cancel_check():
                log("Cancelado por el usuario.")
                break

            seed_noise = stable_hash32(f"{session_tag}|{note}|VEL{vel_midi}|RR{rri}|noise") & 0x7FFFFFFF
            rng_noise = np.random.default_rng(int(seed_noise))

            y, _ph_end0, _ph_end1, _ph_end2 = synth_sustain_one_shot(
                mip_low, mip_mid, mip_high,
                f0_base=f0_base,
                sr=int(sr),
                n=n_sus,
                v=v,
                base_pos=base_pos,
                base_mip=base_mip,
                base_phase=base_phase,
                note_keytrack=kt,
                note_name=note,
                vel_index=vel_index,
                rr_index=rri,
                rng_noise=rng_noise,
                mip_block=int(mip_block),
                mip_smooth_tau=float(mip_smooth_tau),
            )

            y = apply_loop_crossfade(
                y, sr=int(sr),
                loop_start_s=float(loop_start),
                loop_end_s=float(loop_end),
                xfade_s=float(loop_xfade),
            )

            if bits is not None:
                y = add_tpdf_dither(y, bits=bits, rng=rng_noise)

            # todo a la misma carpeta (samples_dir)
            fn = f"{prefix}_{note}.wav"   # ej: nombreejemplo_C3.wav / nombreejemplo_F#3.wav
            out_path = os.path.join(samples_dir, fn)
            sf.write(out_path, y, int(sr), subtype=bitdepth)
            log(f"Wrote {out_path}")

            done += 1
            if progress_fn:
                progress_fn(done, total)

    if progress_fn:
        progress_fn(total, total)
    log("DONE.")
    return out_dir

# =======================
# GUI (PySide6 + qdarkstyle)
# =======================
def launch_gui():
    from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox,
        QComboBox, QTextEdit, QProgressBar, QMessageBox, QGroupBox
    )
    import qdarkstyle

    class Worker(QObject):
        log = Signal(str)
        progress = Signal(int, int)
        finished = Signal(bool, str)  # ok, out_dir
        def __init__(self, cfg: dict):
            super().__init__()
            self.cfg = cfg
            self._cancel = False

        @Slot()
        def run(self):
            try:
                out = render_instrument(
                    wt_dir=self.cfg["wt_dir"],
                    out_dir=self.cfg["out_dir"],
                    seed=self.cfg["seed"],
                    sr=self.cfg["sr"],
                    duration=self.cfg["duration"],
                    release_time=self.cfg["release_time"],
                    loop_start=self.cfg["loop_start"],
                    loop_end=self.cfg["loop_end"],
                    loop_xfade=self.cfg["loop_xfade"],
                    bitdepth=self.cfg["bitdepth"],
                    velocity_midi=self.cfg["velocity"],
                    start_oct=self.cfg["start_oct"],
                    end_oct=self.cfg["end_oct"],
                    mip_block=self.cfg["mip_block"],
                    mip_smooth_tau=self.cfg["mip_smooth_tau"],
                    round_robins=self.cfg["round_robins"],
                    log_fn=lambda s: self.log.emit(s),
                    progress_fn=lambda d, t: self.progress.emit(d, t),
                    cancel_check=lambda: self._cancel,
                )
                if self._cancel:
                    self.finished.emit(False, out)
                else:
                    self.finished.emit(True, out)
            except Exception:
                tb = traceback.format_exc()
                self.log.emit(tb)
                self.finished.emit(False, "")

        def cancel(self):
            self._cancel = True

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("AudioEnvelopeTransferGUI")
            self.setMinimumSize(980, 650)

            self.thread = None
            self.worker = None

            root = QWidget()
            self.setCentralWidget(root)

            main = QVBoxLayout(root)
            main.setSpacing(10)

            # ---- Settings group
            g = QGroupBox("Settings")
            grid = QGridLayout(g)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(8)

            # wt_dir
            grid.addWidget(QLabel("Wavetable folder (wt_dir):"), 0, 0)
            self.wt_dir = QLineEdit(r"D:\WAVETABLE")
            btn_wt = QPushButton("Browse...")
            btn_wt.clicked.connect(self.browse_wt)
            grid.addWidget(self.wt_dir, 0, 1)
            grid.addWidget(btn_wt, 0, 2)

            # out_dir
            grid.addWidget(QLabel("Output folder (out_dir):"), 1, 0)
            self.out_dir = QLineEdit("")
            self.out_dir.setPlaceholderText("Auto (Output next to EXE)")
            btn_out = QPushButton("Browse...")
            btn_out.clicked.connect(self.browse_out)
            grid.addWidget(self.out_dir, 1, 1)
            grid.addWidget(btn_out, 1, 2)

            # seed
            grid.addWidget(QLabel("Seed (0 = random):"), 2, 0)
            self.seed = QSpinBox()
            self.seed.setRange(0, 2_000_000_000)
            self.seed.setValue(0)
            grid.addWidget(self.seed, 2, 1)

            # sr
            grid.addWidget(QLabel("Sample rate (sr):"), 3, 0)
            self.sr = QSpinBox()
            self.sr.setRange(8000, 192000)
            self.sr.setValue(92000)
            grid.addWidget(self.sr, 3, 1)

            # duration
            grid.addWidget(QLabel("Duration (s):"), 4, 0)
            self.duration = QDoubleSpinBox()
            self.duration.setDecimals(2)
            self.duration.setRange(0.5, 60.0)
            self.duration.setValue(15.0)
            grid.addWidget(self.duration, 4, 1)

            # velocity
            grid.addWidget(QLabel("Velocity (1..127):"), 5, 0)
            self.velocity = QSpinBox()
            self.velocity.setRange(1, 127)
            self.velocity.setValue(100)
            grid.addWidget(self.velocity, 5, 1)

            # round robins (fijo a 1)
            grid.addWidget(QLabel("Round Robins (RR):"), 6, 0)
            self.rr = QSpinBox()
            self.rr.setRange(1, 1)
            self.rr.setValue(1)
            self.rr.setEnabled(False)
            grid.addWidget(self.rr, 6, 1)

            # bitdepth
            grid.addWidget(QLabel("Bit depth:"), 7, 0)
            self.bitdepth = QComboBox()
            self.bitdepth.addItems(["FLOAT", "PCM_16", "PCM_24", "PCM_32"])
            self.bitdepth.setCurrentText("FLOAT")
            grid.addWidget(self.bitdepth, 7, 1)

            # release_time (fijo off)
            grid.addWidget(QLabel("Release time (s) (0 = off):"), 8, 0)
            self.release_time = QDoubleSpinBox()
            self.release_time.setDecimals(2)
            self.release_time.setRange(0.0, 20.0)
            self.release_time.setValue(0.0)
            self.release_time.setEnabled(False)
            grid.addWidget(self.release_time, 8, 1)

            # loop start/end/xfade
            grid.addWidget(QLabel("Loop start / end / xfade (s):"), 9, 0)
            row = QHBoxLayout()
            self.loop_start = QDoubleSpinBox(); self.loop_start.setDecimals(3); self.loop_start.setRange(0.0, 60.0); self.loop_start.setValue(1.0)
            self.loop_end   = QDoubleSpinBox(); self.loop_end.setDecimals(3); self.loop_end.setRange(0.0, 60.0); self.loop_end.setValue(4.0)
            self.loop_xfade = QDoubleSpinBox(); self.loop_xfade.setDecimals(3); self.loop_xfade.setRange(0.0, 1.0);  self.loop_xfade.setValue(0.12)
            row.addWidget(self.loop_start); row.addWidget(self.loop_end); row.addWidget(self.loop_xfade)
            wrow = QWidget(); wrow.setLayout(row)
            grid.addWidget(wrow, 9, 1)

            # mip block / smooth
            grid.addWidget(QLabel("Mip block / smooth tau:"), 10, 0)
            row2 = QHBoxLayout()
            self.mip_block = QSpinBox(); self.mip_block.setRange(64, 2048); self.mip_block.setValue(256)
            self.mip_tau   = QDoubleSpinBox(); self.mip_tau.setDecimals(3); self.mip_tau.setRange(0.0, 1.0); self.mip_tau.setValue(0.030)
            row2.addWidget(self.mip_block); row2.addWidget(self.mip_tau)
            wrow2 = QWidget(); wrow2.setLayout(row2)
            grid.addWidget(wrow2, 10, 1)

            # fixed tones label
            grid.addWidget(QLabel("Roots (fixed):"), 11, 0)
            roots_lbl = QLabel("C-2, F#-2 .. C8, F#8 (22 tones)")
            roots_lbl.setStyleSheet("opacity: 0.85;")
            grid.addWidget(roots_lbl, 11, 1)

            main.addWidget(g)

            # ---- Controls
            ctrl = QHBoxLayout()
            self.btn_start = QPushButton("Start Render")
            self.btn_cancel = QPushButton("Cancel")
            self.btn_cancel.setEnabled(False)
            self.btn_start.clicked.connect(self.start_render)
            self.btn_cancel.clicked.connect(self.cancel_render)
            ctrl.addWidget(self.btn_start)
            ctrl.addWidget(self.btn_cancel)
            ctrl.addStretch(1)
            main.addLayout(ctrl)

            # ---- Progress
            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            main.addWidget(self.progress)

            # ---- Logs
            self.logs = QTextEdit()
            self.logs.setReadOnly(True)
            self.logs.setPlaceholderText("Logs...")
            main.addWidget(self.logs, 1)

            # ---- Footer centered
            footer = QLabel("© 2025 Gabriel Golker")
            footer.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            footer.setStyleSheet("opacity: 0.80; padding: 6px;")
            main.addWidget(footer)

        def append_log(self, s: str):
            self.logs.append(s)

        def browse_wt(self):
            d = QFileDialog.getExistingDirectory(self, "Select Wavetable Folder", self.wt_dir.text() or os.getcwd())
            if d:
                self.wt_dir.setText(d)

        def browse_out(self):
            d = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.out_dir.text() or os.getcwd())
            if d:
                self.out_dir.setText(d)

        def _set_running(self, running: bool):
            self.btn_start.setEnabled(not running)
            self.btn_cancel.setEnabled(running)

        def start_render(self):
            wt = self.wt_dir.text().strip()
            if not wt or not os.path.isdir(wt):
                QMessageBox.warning(self, "Invalid wt_dir", "Please select a valid wavetable folder.")
                return

            out = self.out_dir.text().strip()
            out = out if out else None

            cfg = {
                "wt_dir": wt,
                "out_dir": out,
                "seed": int(self.seed.value()),
                "sr": int(self.sr.value()),
                "duration": float(self.duration.value()),
                "release_time": float(self.release_time.value()),
                "loop_start": float(self.loop_start.value()),
                "loop_end": float(self.loop_end.value()),
                "loop_xfade": float(self.loop_xfade.value()),
                "bitdepth": str(self.bitdepth.currentText()),
                "velocity": int(self.velocity.value()),
                "start_oct": -2,
                "end_oct": 8,
                "mip_block": int(self.mip_block.value()),
                "mip_smooth_tau": float(self.mip_tau.value()),
                "round_robins": int(self.rr.value()),
            }

            self.logs.clear()
            self.progress.setValue(0)
            self._set_running(True)

            self.thread = QThread()
            self.worker = Worker(cfg)
            self.worker.moveToThread(self.thread)

            self.thread.started.connect(self.worker.run)
            self.worker.log.connect(self.append_log)
            self.worker.progress.connect(self.on_progress)
            self.worker.finished.connect(self.on_finished)

            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            self.thread.start()

        def cancel_render(self):
            if self.worker is not None:
                self.append_log("Cancel requested...")
                self.worker.cancel()
                self.btn_cancel.setEnabled(False)

        def on_progress(self, done: int, total: int):
            if total <= 0:
                self.progress.setValue(0)
                return
            pct = int(np.clip((done / total) * 100.0, 0.0, 100.0))
            self.progress.setValue(pct)

        def on_finished(self, ok: bool, out_dir: str):
            self._set_running(False)
            if ok:
                self.append_log("")
                self.append_log(f"✅ Finished. Output: {out_dir}")
                QMessageBox.information(self, "Done", f"Render finished.\nOutput:\n{out_dir}")
            else:
                self.append_log("")
                self.append_log("❌ Finished with errors or canceled.")
                QMessageBox.warning(self, "Finished", "Render finished with errors or was canceled.")

    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyside6())
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

# =======================
# CLI (opcional) + default GUI
# =======================
def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--nogui", action="store_true", help="Run in CLI mode (no GUI)")

    ap.add_argument("--wt_dir", default=r"D:\WAVETABLE", help="Carpeta con wavetables .wav (recursivo)")
    ap.add_argument("--out_dir", default=None, help="Carpeta de salida (si None, se auto-elige)")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--sr", type=int, default=92000)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--release_time", type=float, default=0.0)

    ap.add_argument("--loop_start", type=float, default=1.0)
    ap.add_argument("--loop_end", type=float, default=4.0)
    ap.add_argument("--loop_xfade", type=float, default=0.12)

    ap.add_argument("--bitdepth", default="FLOAT", choices=["PCM_16","PCM_24","PCM_32","FLOAT"])
    ap.add_argument("--velocity", type=int, default=100)

    ap.add_argument("--start_oct", type=int, default=-2)
    ap.add_argument("--end_oct", type=int, default=8)

    ap.add_argument("--mip_block", type=int, default=256)
    ap.add_argument("--mip_smooth_tau", type=float, default=0.030)

    ap.add_argument("--round_robins", type=int, default=3)

    args = ap.parse_args()

    if not args.nogui:
        launch_gui()
        return

    out = render_instrument(
        wt_dir=args.wt_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        sr=args.sr,
        duration=args.duration,
        release_time=args.release_time,
        loop_start=args.loop_start,
        loop_end=args.loop_end,
        loop_xfade=args.loop_xfade,
        bitdepth=args.bitdepth,
        velocity_midi=args.velocity,
        start_oct=args.start_oct,
        end_oct=args.end_oct,
        mip_block=args.mip_block,
        mip_smooth_tau=args.mip_smooth_tau,
        round_robins=args.round_robins,
    )
    print("Output:", out)

if __name__ == "__main__":
    main()
