import os, glob, math, argparse, sys
import numpy as np
import soundfile as sf

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

# ==========================================================
# (1) Interpolación mejor: Cubic/Hermite (Catmull-Rom) 4 puntos
# ==========================================================
def _table_read_cubic_hermite(table_1d: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """
    Catmull-Rom cubic (Hermite-like) 4-point interpolation.
    phase: 0..1 (vector)
    table_1d: 1D wavetable (circular)
    """
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
# (2) Anti-aliasing mejor: derivative mip (por inc) + smoothing
# (3) Optimización: mip-blend por bloques
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

    # phase vectorizada
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

# ==========================================
# Velocities perceptuales
# ==========================================
def velocity_bins_perceptual(n_layers: int, curve: float = 2.2):
    x = np.linspace(0.0, 1.0, n_layers + 1)
    edges = 1.0 + 127.0 * (x ** curve)
    bins = []
    prev = 0
    for i in range(n_layers):
        lo = int(math.floor(edges[i]))
        hi = int(math.floor(edges[i+1] - 1e-9))
        lo = max(lo, prev + 1)
        hi = max(hi, lo)
        if i == n_layers - 1:
            hi = 127
        bins.append((lo, hi))
        prev = hi
    return bins

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
# ADSR helpers (sustain-only + release-only)
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

    drive_base = np.interp(v, [0,1], [1.10, 2.70])
    drive_base *= (1.05 - 0.25*kt)
    drive_base *= (0.95 + 0.15*kt)
    drive_base *= (1.0 + 0.12*knee_part)
    drive_env = (drive_base * (1.0 + 0.90 * fenv)).astype(np.float32)
    y = np.tanh(drive_env * y).astype(np.float32, copy=False)

    floor = np.interp(v, [0,1], [0.006, 0.0015])
    y = (y + floor * bp_noise * (env_sus**0.7)).astype(np.float32, copy=False)

    y = cheap_body_eq(y, sr=sr, kt=kt, v=v)

    rms = float(np.sqrt(np.mean(y*y) + 1e-12))
    target_rms = db_to_lin(np.interp(v**1.35, [0,1], [-28.0, -10.0]))
    corr = float(np.clip(target_rms / (rms + 1e-12), 0.5, 1.8))
    y = (y * corr).astype(np.float32, copy=False)

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
# MAIN
# =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt_dir", default=r"D:\WAVETABLE", help="Carpeta con wavetables .wav (recursivo)")

    # ✅ CAMBIO: out_dir ya NO es required (para EXE --windowed/GUI)
    ap.add_argument("--out_dir", default=None, help="Carpeta de salida (si None, se auto-elige)")

    ap.add_argument("--seed", type=int, default=0, help="0 = random real (elige wavetables/patch distinto), otro = reproducible")

    ap.add_argument("--sr", type=int, default=92000)
    ap.add_argument("--duration", type=float, default=15.0, help="Duración del sample principal (sustain/loop)")
    ap.add_argument("--release_time", type=float, default=4.0, help="Duración del sample release (0 desactiva)")

    ap.add_argument("--loop_start", type=float, default=1.0)
    ap.add_argument("--loop_end", type=float, default=4.0)
    ap.add_argument("--loop_xfade", type=float, default=0.12)

    ap.add_argument("--bitdepth", default="FLOAT", choices=["PCM_16","PCM_24","PCM_32","FLOAT"])
    ap.add_argument("--vel_layers", type=int, default=25)
    ap.add_argument("--vel_curve", type=float, default=2.2)

    ap.add_argument("--start_oct", type=int, default=-2)
    ap.add_argument("--end_oct", type=int, default=8)

    ap.add_argument("--mip_block", type=int, default=256, help="Tamaño de bloque para mip-blend (128-512 típico)")
    ap.add_argument("--mip_smooth_tau", type=float, default=0.030, help="Tau smoothing para mip-level (segundos)")

    ap.add_argument("--round_robins", type=int, default=3, help="Cantidad de RR por (nota, vel) (1-4 típico)")

    args = ap.parse_args()

    # ✅ CAMBIO: fallback automático si no pasas --out_dir
    if args.out_dir is None:
        base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
        args.out_dir = os.path.join(base, "Output")

    sr = int(args.sr)
    n_sus = int(sr * float(args.duration))
    n_rel = int(sr * float(max(0.0, args.release_time)))

    rrN = int(max(1, args.round_robins))
    rng = np.random.default_rng(None if args.seed == 0 else args.seed)

    wt_files = list_wav_files(args.wt_dir)
    if len(wt_files) < 3:
        raise RuntimeError(f"Necesitas al menos 3 wavetables .wav en {args.wt_dir}")

    wt_low, wt_mid, wt_high = rng.choice(wt_files, size=3, replace=False).tolist()

    print("Wavetables fijos para toda la sesión (low/mid/high):")
    print("  LOW :", os.path.basename(wt_low))
    print("  MID :", os.path.basename(wt_mid))
    print("  HIGH:", os.path.basename(wt_high))

    mip_low  = build_wavetable_mipmaps(load_wavetable_wav(wt_low),  levels=WT_MIP_LEVELS)
    mip_mid  = build_wavetable_mipmaps(load_wavetable_wav(wt_mid),  levels=WT_MIP_LEVELS)
    mip_high = build_wavetable_mipmaps(load_wavetable_wav(wt_high), levels=WT_MIP_LEVELS)

    base_pos   = [float(rng.uniform(0.0, 1.0)) for _ in range(3)]
    base_mip   = [float(rng.uniform(0.65, 1.0)) for _ in range(3)]
    base_phase = [float(rng.uniform(0.0, 1.0)) for _ in range(3)]

    vel_bins = velocity_bins_perceptual(int(args.vel_layers), curve=float(args.vel_curve))

    roots = [f"C{o}" for o in range(int(args.start_oct), int(args.end_oct) + 1)]
    total = len(roots) * len(vel_bins) * rrN

    samples_dir = os.path.join(args.out_dir, "Samples")
    os.makedirs(samples_dir, exist_ok=True)

    print(f"Generando {len(roots)} roots x {len(vel_bins)} vel x {rrN} RR = {total} WAVs (+ release si aplica)")
    print(f"SR={sr}  SUS={args.duration}s  REL={args.release_time}s  BIT={args.bitdepth}")
    print(f"Loop: start={args.loop_start}s end={args.loop_end}s xfade={args.loop_xfade}s (se aplica al sample principal)")
    print(f"OUT: {args.out_dir}")

    kt_den = max(1.0, float(args.end_oct - args.start_oct))

    session_tag = (
        f"{os.path.basename(wt_low)}|{os.path.basename(wt_mid)}|{os.path.basename(wt_high)}|"
        f"{base_pos[0]:.4f},{base_pos[1]:.4f},{base_pos[2]:.4f}|"
        f"{base_mip[0]:.4f},{base_mip[1]:.4f},{base_mip[2]:.4f}|"
        f"{base_phase[0]:.4f},{base_phase[1]:.4f},{base_phase[2]:.4f}"
    )

    bits = _bits_for_subtype(args.bitdepth)

    for note in roots:
        f0_base = note_name_to_hz(note)
        note_dir = os.path.join(samples_dir, safe_name(note))
        os.makedirs(note_dir, exist_ok=True)

        octv = int(note[1:])
        kt = float((octv - args.start_oct) / kt_den)

        for vi, (vlo, vhi) in enumerate(vel_bins, start=1):
            v_center = 0.5 * (vlo + vhi)
            v = float(np.clip(v_center / 127.0, 0.0, 1.0))

            for rri in range(1, rrN + 1):
                seed_noise = stable_hash32(f"{session_tag}|{note}|{vi}|RR{rri}|noise") & 0x7FFFFFFF
                rng_noise = np.random.default_rng(int(seed_noise))

                y, ph_end0, ph_end1, ph_end2 = synth_sustain_one_shot(
                    mip_low, mip_mid, mip_high,
                    f0_base=f0_base,
                    sr=sr,
                    n=n_sus,
                    v=v,
                    base_pos=base_pos,
                    base_mip=base_mip,
                    base_phase=base_phase,
                    note_keytrack=kt,
                    note_name=note,
                    vel_index=vi,
                    rr_index=rri,
                    rng_noise=rng_noise,
                    mip_block=int(args.mip_block),
                    mip_smooth_tau=float(args.mip_smooth_tau),
                )

                y = apply_loop_crossfade(
                    y, sr=sr,
                    loop_start_s=float(args.loop_start),
                    loop_end_s=float(args.loop_end),
                    xfade_s=float(args.loop_xfade),
                )

                if bits is not None:
                    y = add_tpdf_dither(y, bits=bits, rng=rng_noise)

                fn = f"SYN_{safe_name(note)}_VEL{vi:02d}_RR{rri:02d}_({vlo:03d}-{vhi:03d}).wav"
                out_path = os.path.join(note_dir, fn)
                sf.write(out_path, y, sr, subtype=args.bitdepth)
                print(f"Wrote {out_path}")

                if n_rel > 0:
                    seed_rel = stable_hash32(f"{session_tag}|{note}|{vi}|RR{rri}|release_noise") & 0x7FFFFFFF
                    rng_rel = np.random.default_rng(int(seed_rel))

                    yrel = synth_release_tail(
                        mip_low, mip_mid, mip_high,
                        f0_base=f0_base,
                        sr=sr,
                        n=n_rel,
                        v=v,
                        base_pos=base_pos,
                        base_mip=base_mip,
                        note_keytrack=kt,
                        note_name=note,
                        vel_index=vi,
                        rr_index=rri,
                        rng_noise=rng_rel,
                        phase0_low=ph_end0,
                        phase0_mid=ph_end1,
                        phase0_high=ph_end2,
                        mip_block=int(args.mip_block),
                        mip_smooth_tau=float(args.mip_smooth_tau),
                    )

                    if bits is not None:
                        yrel = add_tpdf_dither(yrel, bits=bits, rng=rng_rel)

                    fnr = f"SYN_{safe_name(note)}_VEL{vi:02d}_RR{rri:02d}_REL_({vlo:03d}-{vhi:03d}).wav"
                    out_path_r = os.path.join(note_dir, fnr)
                    sf.write(out_path_r, yrel, sr, subtype=args.bitdepth)
                    print(f"Wrote {out_path_r}")

    print("DONE.")

if __name__ == "__main__":
    main()
