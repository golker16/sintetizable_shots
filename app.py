import os, glob, math, argparse
import numpy as np
import soundfile as sf

# =======================
# WAVETABLE CORE (reutilizado, sin UI)
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

    # pad a múltiplo de frame (mejor que truncar)
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

def _pick_mip_level(f0_hz: float, mip_strength: float, n_levels: int) -> int:
    f_ref = 55.0
    ratio = max(f0_hz / f_ref, 1e-6)
    lvl_float = math.log2(ratio) * float(np.clip(mip_strength, 0.0, 1.0))
    return int(np.clip(math.floor(lvl_float), 0, n_levels - 1))

def render_wavetable_osc_f0_array(
    f0_hz: np.ndarray,
    sr: int,
    mipmaps: list,
    position: float,
    phase0: float,
    mip_strength: float,
):
    """
    Oscilador wavetable con f0 por-sample (vectorizado).
    Para performance: elegimos mip level una sola vez usando f0 medio.
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    n = len(f0_hz)

    base_frames = mipmaps[0]
    n_frames = base_frames.shape[0]
    pos = float(np.clip(position, 0.0, 1.0))
    fidx = pos * (n_frames - 1)
    f0i = int(np.floor(fidx))
    ft = float(fidx - f0i)
    f1i = min(f0i + 1, n_frames - 1)

    n_levels = len(mipmaps)
    f0_mean = float(np.maximum(np.mean(f0_hz), 1.0))
    L = _pick_mip_level(f0_mean, float(mip_strength), n_levels)
    tables_L = mipmaps[L]
    table = _lerp(tables_L[f0i], tables_L[f1i], ft).astype(np.float32, copy=False)

    # phase: phase[i] = phase0 + sum_{k< i} f0[k]/sr
    inc = (f0_hz / float(sr)).astype(np.float32)
    c = np.cumsum(inc, dtype=np.float64)  # estable numéricamente
    phase = (float(phase0) + np.concatenate(([0.0], c[:-1].astype(np.float32)))) % 1.0

    out = _table_read_linear(table, phase).astype(np.float32, copy=False)
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
    midi = (octv + 1) * 12 + semitone  # C-1 = 0
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

def velocity_bins(n_layers: int):
    edges = np.linspace(1.0, 128.0, n_layers + 1)
    bins = []
    prev_hi = 0
    for i in range(n_layers):
        lo = int(math.floor(edges[i]))
        hi = int(math.floor(edges[i+1] - 1e-9))
        lo = max(lo, prev_hi + 1)
        hi = max(hi, lo)
        if i == n_layers - 1:
            hi = 127
        bins.append((lo, hi))
        prev_hi = hi
    return bins

def sat_tanh(x: np.ndarray, drive: float) -> np.ndarray:
    return np.tanh(float(drive) * x).astype(np.float32, copy=False)

def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))

def exp_decay(t: np.ndarray, tau: float) -> np.ndarray:
    return np.exp(-t / float(tau)).astype(np.float32)

def adsr_env(n: int, sr: int, v: float) -> np.ndarray:
    """
    ADSR más realista por velocity:
    - suave: ataque más lento / release más largo / sustain más bajo
    - fuerte: ataque corto / sustain alto / release algo más corto
    """
    v = float(np.clip(v, 0.0, 1.0))
    atk = np.interp(v, [0.0, 1.0], [0.120, 0.004])
    dec = np.interp(v, [0.0, 1.0], [1.40, 0.45])
    sus = np.interp(v, [0.0, 1.0], [0.40, 0.90])
    rel = np.interp(v, [0.0, 1.0], [3.80, 1.50])

    aN = int(atk * sr)
    dN = int(dec * sr)
    rN = int(rel * sr)
    sN = max(0, n - (aN + dN + rN))

    env = np.zeros(n, dtype=np.float32)
    i = 0
    if aN > 0:
        # ataque curva (suave = curva más lenta)
        a = np.linspace(0.0, 1.0, aN, endpoint=False, dtype=np.float32)
        env[i:i+aN] = a ** np.interp(v, [0,1], [1.8, 0.7])
        i += aN
    if dN > 0:
        env[i:i+dN] = np.linspace(1.0, sus, dN, endpoint=False, dtype=np.float32)
        i += dN
    if sN > 0:
        env[i:i+sN] = sus
        i += sN
    if i < n:
        tail = n - i
        start = env[i-1] if i > 0 else sus
        env[i:] = np.linspace(start, 0.0, tail, endpoint=True, dtype=np.float32)
    return env

# =======================
# SYNTH: 3 OSC fijos (3 wavetables) + performance por velocity
# =======================
def synth_one_shot(
    mip_low, mip_mid, mip_high,
    f0_base: float,
    sr: int,
    n: int,
    v: float,
    base_pos, base_mip, base_phase,
    note_keytrack: float,
    rng_local: np.random.Generator,
):
    """
    v: 0..1 (velocity)
    note_keytrack: 0..1 (más alto => nota más aguda)
    """
    v = float(np.clip(v, 0.0, 1.0))
    kt = float(np.clip(note_keytrack, 0.0, 1.0))

    t = (np.arange(n, dtype=np.float32) / float(sr))

    # ---- dinámica (volumen) en dB: suave muy bajo, fuerte cercano a 0dB ----
    # Esto hace que las capas suaves no parezcan "solo un poco más bajo", sino realmente tocadas suave.
    amp_db = np.interp(v**1.65, [0.0, 1.0], [-32.0, -3.0])
    amp = db_to_lin(amp_db)

    # ---- “interprete”: pitch env + vibrato sutil (más fuerte => más evidente) ----
    pitch_attack_cents = (2.0 + 10.0*(v**0.9)) * (1.0 + 0.25*kt)  # agudas un poco más nerviosas
    pitch_env = pitch_attack_cents * exp_decay(t, tau=np.interp(v, [0,1], [0.075, 0.040]))

    vib_rate = np.interp(v, [0,1], [4.2, 5.6])
    vib_depth_cents = np.interp(v, [0,1], [0.2, 1.2])
    vibrato = vib_depth_cents * np.sin(2.0*np.pi*vib_rate*t).astype(np.float32)

    cents_total = (pitch_env + vibrato).astype(np.float32)
    f0 = (f0_base * (2.0 ** (cents_total / 1200.0))).astype(np.float32)

    # ---- micro-detune fijo por sample (no “chorus”, solo vida) ----
    det_cents = float(rng_local.uniform(-1.5, 1.5))
    det = float(2.0 ** (det_cents / 1200.0))
    f0 = (f0 * det).astype(np.float32)

    # ---- WT scan dependiente de velocity y keytracking ----
    # (no random): consistente y “más brillante” a más velocidad y más nota
    pos_low  = float(np.clip(base_pos[0] + 0.08*(v-0.5) + 0.05*(kt-0.5), 0.0, 1.0))
    pos_mid  = float(np.clip(base_pos[1] + 0.11*(v-0.5) + 0.06*(kt-0.5), 0.0, 1.0))
    pos_high = float(np.clip(base_pos[2] + 0.16*(v-0.5) + 0.08*(kt-0.5), 0.0, 1.0))

    # ---- 3 capas tipo low/mid/high (armónicos) ----
    y0 = render_wavetable_osc_f0_array(f0 * 1.0, sr, mip_low,  position=pos_low,  phase0=base_phase[0], mip_strength=base_mip[0])
    y1 = render_wavetable_osc_f0_array(f0 * 2.0, sr, mip_mid,  position=pos_mid,  phase0=base_phase[1], mip_strength=base_mip[1])
    y2 = render_wavetable_osc_f0_array(f0 * 4.0, sr, mip_high, position=pos_high, phase0=base_phase[2], mip_strength=base_mip[2])

    # ---- mezcla por velocity + keytracking ----
    # suave: más low / menos high; fuerte: aparece high + más mid
    bright = (v**0.65) * (0.85 + 0.30*kt)

    w_low  = np.clip(0.80 - 0.22*bright, 0.40, 0.85)
    w_mid  = np.clip(0.28 + 0.40*bright, 0.20, 0.75)
    w_high = np.clip(0.02 + 0.70*(bright**1.20), 0.00, 0.70)

    y = (w_low*y0 + w_mid*y1 + w_high*y2).astype(np.float32, copy=False)

    # ---- transient realista (ataque) ----
    # burst corto de “aire” (diferenciador = highpass barato y vectorizado)
    noise = rng_local.standard_normal(n).astype(np.float32)
    hp_noise = (noise - np.concatenate(([0.0], noise[:-1]))).astype(np.float32)  # highpass simple
    trans_len = int(np.interp(v, [0,1], [0.010, 0.030]) * sr)
    trans = np.zeros(n, dtype=np.float32)
    if trans_len > 8:
        trans_env = exp_decay(t[:trans_len], tau=np.interp(v, [0,1], [0.010, 0.018]))
        trans[:trans_len] = hp_noise[:trans_len] * trans_env
    trans_gain = np.interp(v, [0,1], [0.010, 0.060]) * (0.9 + 0.3*kt)
    y = (y + trans_gain * trans).astype(np.float32, copy=False)

    # ---- ADSR + saturación (más fuerte => más drive) ----
    env = adsr_env(n, sr, v)
    y = (y * env).astype(np.float32, copy=False)

    drive = np.interp(v, [0,1], [1.10, 2.70]) * (0.95 + 0.25*kt)
    y = sat_tanh(y, drive=drive)

    # ---- “breath/noise floor” (más presente en vel bajas) ----
    floor = np.interp(v, [0,1], [0.006, 0.0015])
    y = (y + floor * hp_noise * (env**0.7)).astype(np.float32, copy=False)

    # ---- aplicar volumen final (NO normalizar por archivo) ----
    y = (y * amp * 0.85).astype(np.float32, copy=False)

    return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)

# =======================
# MAIN
# =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt_dir", default=r"D:\WAVETABLE", help="Carpeta con wavetables .wav (recursivo)")
    ap.add_argument("--out_dir", required=True, help="Carpeta de salida")
    ap.add_argument("--seed", type=int, default=0, help="0 = random real, otro = reproducible")
    ap.add_argument("--sr", type=int, default=92000)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--bitdepth", default="PCM_24", choices=["PCM_16","PCM_24","PCM_32"])
    ap.add_argument("--vel_layers", type=int, default=25)

    # 11 roots: C-2..C8 (inclusive)
    ap.add_argument("--start_oct", type=int, default=-2)
    ap.add_argument("--end_oct", type=int, default=8)

    args = ap.parse_args()

    sr = int(args.sr)
    n = int(sr * float(args.duration))
    rng = np.random.default_rng(None if args.seed == 0 else args.seed)

    wt_files = list_wav_files(args.wt_dir)
    if len(wt_files) < 3:
        raise RuntimeError(f"Necesitas al menos 3 wavetables .wav en {args.wt_dir}")

    # ✅ MISMA SESIÓN = mismos 3 osciladores (3 wavetables) para TODO
    wt_low, wt_mid, wt_high = rng.choice(wt_files, size=3, replace=False).tolist()

    print("Wavetables fijos para toda la sesión (low/mid/high):")
    print("  LOW :", os.path.basename(wt_low))
    print("  MID :", os.path.basename(wt_mid))
    print("  HIGH:", os.path.basename(wt_high))

    mip_low  = build_wavetable_mipmaps(load_wavetable_wav(wt_low),  levels=WT_MIP_LEVELS)
    mip_mid  = build_wavetable_mipmaps(load_wavetable_wav(wt_mid),  levels=WT_MIP_LEVELS)
    mip_high = build_wavetable_mipmaps(load_wavetable_wav(wt_high), levels=WT_MIP_LEVELS)

    # patch base global (constante para todo el instrumento)
    base_pos   = [float(rng.uniform(0.0, 1.0)) for _ in range(3)]
    base_mip   = [float(rng.uniform(0.65, 1.0)) for _ in range(3)]
    base_phase = [float(rng.uniform(0.0, 1.0)) for _ in range(3)]

    vel_bins = velocity_bins(int(args.vel_layers))

    roots = [f"C{o}" for o in range(int(args.start_oct), int(args.end_oct) + 1)]
    total = len(roots) * len(vel_bins)

    samples_dir = os.path.join(args.out_dir, "Samples")
    os.makedirs(samples_dir, exist_ok=True)

    print(f"Generando {len(roots)} roots (C{args.start_oct}..C{args.end_oct}) x {len(vel_bins)} vel = {total} WAVs")
    print(f"SR={sr}  DUR={args.duration}s  BIT={args.bitdepth}")

    # para keytracking (0..1): C-2 = 0, C8 = 1
    kt_den = max(1.0, float(args.end_oct - args.start_oct))

    for note in roots:
        f0_base = note_name_to_hz(note)
        note_dir = os.path.join(samples_dir, safe_name(note))
        os.makedirs(note_dir, exist_ok=True)

        octv = int(note[1:])  # "C-2" -> "-2"
        kt = float((octv - args.start_oct) / kt_den)

        for vi, (vlo, vhi) in enumerate(vel_bins, start=1):
            v_center = 0.5 * (vlo + vhi)
            v = float(np.clip(v_center / 127.0, 0.0, 1.0))

            # RNG local: mantiene coherencia por nota/vel si seed != 0
            local_seed = None if args.seed == 0 else (args.seed * 1000003 + (hash((note, vi)) & 0x7FFFFFFF))
            rng_local = np.random.default_rng(local_seed)

            y = synth_one_shot(
                mip_low, mip_mid, mip_high,
                f0_base=f0_base,
                sr=sr,
                n=n,
                v=v,
                base_pos=base_pos,
                base_mip=base_mip,
                base_phase=base_phase,
                note_keytrack=kt,
                rng_local=rng_local,
            )

            fn = f"SYN_{safe_name(note)}_VEL{vi:02d}_({vlo:03d}-{vhi:03d}).wav"
            out_path = os.path.join(note_dir, fn)
            sf.write(out_path, y, sr, subtype=args.bitdepth)
            print(f"Wrote {out_path}")

    print("DONE.")

if __name__ == "__main__":
    main()


