import os, glob, math, argparse
import numpy as np
import soundfile as sf

# =======================
# Wavetable core (tomado de tu base, sin UI)
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
    # selector simple como el tuyo
    f_ref = 55.0
    ratio = max(f0_hz / f_ref, 1e-6)
    lvl_float = math.log2(ratio) * float(np.clip(mip_strength, 0.0, 1.0))
    lvl = int(np.clip(math.floor(lvl_float), 0, n_levels - 1))
    return lvl

def render_wavetable_osc_constant(
    f0_hz: float,
    sr: int,
    n_samples: int,
    mipmaps: list,
    position: float = 0.0,
    phase0: float = 0.0,
    mip_strength: float = 1.0,
):
    n_levels = len(mipmaps)
    base_frames = mipmaps[0]
    n_frames = base_frames.shape[0]

    pos = float(np.clip(position, 0.0, 1.0))
    fidx = pos * (n_frames - 1)
    f0i = int(np.floor(fidx))
    ft = float(fidx - f0i)
    f1i = min(f0i + 1, n_frames - 1)

    L = _pick_mip_level(float(f0_hz), float(mip_strength), n_levels)
    tables_L = mipmaps[L]
    t0 = tables_L[f0i]
    t1 = tables_L[f1i]
    table = _lerp(t0, t1, ft).astype(np.float32, copy=False)

    # phase vectorizado
    phase = (float(phase0) + (np.arange(n_samples, dtype=np.float32) * (float(f0_hz) / float(sr)))) % 1.0
    out = _table_read_linear(table, phase).astype(np.float32, copy=False)

    # phase final (por si algún día lo necesitas)
    ph_end = float((float(phase0) + (n_samples * (float(f0_hz) / float(sr)))) % 1.0)
    return out, ph_end

# =======================
# Musical utils
# =======================
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def name_c_oct(octv: int) -> str:
    return f"C{octv}"

def note_name_to_hz(note: str) -> float:
    # A4=440, formato "C-2", "C4", "C#3", etc. (12-TET)
    # Convención: C4 = 261.625565...
    # MIDI equivalente: C4 = 60
    note = note.strip().upper().replace(" ", "")
    # parse nombre + octava
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
    octv = int(oct_str)  # acepta negativos
    semitone = NOTE_NAMES.index(nn)
    midi = (octv + 1) * 12 + semitone  # porque C-1=0
    hz = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    return float(hz)

def velocity_bins(n_layers: int):
    # 25 capas => 25 rangos dentro de 1..127
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

def adsr_env(n: int, sr: int, v_norm: float) -> np.ndarray:
    # más fuerte => ataque más corto y sustain más alto
    atk = np.interp(v_norm, [0.0, 1.0], [0.090, 0.006])
    dec = np.interp(v_norm, [0.0, 1.0], [1.20, 0.55])
    sus = np.interp(v_norm, [0.0, 1.0], [0.50, 0.88])
    rel = np.interp(v_norm, [0.0, 1.0], [3.20, 1.70])

    aN = int(atk * sr)
    dN = int(dec * sr)
    rN = int(rel * sr)
    sN = max(0, n - (aN + dN + rN))

    env = np.zeros(n, dtype=np.float32)
    idx = 0
    if aN > 0:
        env[idx:idx+aN] = np.linspace(0.0, 1.0, aN, endpoint=False, dtype=np.float32)
        idx += aN
    if dN > 0:
        env[idx:idx+dN] = np.linspace(1.0, sus, dN, endpoint=False, dtype=np.float32)
        idx += dN
    if sN > 0:
        env[idx:idx+sN] = sus
        idx += sN
    if idx < n:
        tail = n - idx
        start = env[idx-1] if idx > 0 else sus
        env[idx:] = np.linspace(start, 0.0, tail, endpoint=True, dtype=np.float32)
    return env

def sat_tanh(x: np.ndarray, drive: float) -> np.ndarray:
    return np.tanh(float(drive) * x).astype(np.float32, copy=False)

# =======================
# File utils
# =======================
def list_wav_files(folder: str):
    files = glob.glob(os.path.join(folder, "**/*.wav"), recursive=True)
    files += glob.glob(os.path.join(folder, "**/*.WAV"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=lambda p: p.lower())
    return files

def safe_name(s: str) -> str:
    return s.replace("#","s").replace("-","m")

# =======================
# Synth patch (3 capas fijas: low/mid/high)
# =======================
def synth_one_shot(
    mipmaps_low, mipmaps_mid, mipmaps_high,
    note_hz: float,
    sr: int,
    n_samples: int,
    v_norm: float,
    base_pos, base_mip, base_phase,
    rng_local: np.random.Generator,
):
    # dinámica (curva tipo “real”)
    amp = 0.08 + 0.92 * (v_norm ** 1.85)

    # más fuerte => más brillo y un poco más drive
    drive = float(np.interp(v_norm, [0.0, 1.0], [1.15, 2.60]))

    # micro detune sutil (determinístico por (nota,vel) si usas same seed)
    cents = float(rng_local.uniform(-2.0, 2.0))
    detune = 2.0 ** (cents / 1200.0)

    # “patch” consistente:
    # - low toca a f0
    # - mid toca a 2*f0
    # - high toca a 4*f0
    # (esto te da una sensación de capas espectrales SIN filtros pesados)
    f0_low  = note_hz * detune
    f0_mid  = note_hz * 2.0 * detune
    f0_high = note_hz * 4.0 * detune

    # escaneo de wavetable: leve movimiento por velocidad (mantiene coherencia)
    pos_low  = float(np.clip(base_pos[0] + 0.10 * (v_norm - 0.5), 0.0, 1.0))
    pos_mid  = float(np.clip(base_pos[1] + 0.14 * (v_norm - 0.5), 0.0, 1.0))
    pos_high = float(np.clip(base_pos[2] + 0.20 * (v_norm - 0.5), 0.0, 1.0))

    y0, _ = render_wavetable_osc_constant(f0_low,  sr, n_samples, mipmaps_low,  position=pos_low,  phase0=base_phase[0], mip_strength=base_mip[0])
    y1, _ = render_wavetable_osc_constant(f0_mid,  sr, n_samples, mipmaps_mid,  position=pos_mid,  phase0=base_phase[1], mip_strength=base_mip[1])
    y2, _ = render_wavetable_osc_constant(f0_high, sr, n_samples, mipmaps_high, position=pos_high, phase0=base_phase[2], mip_strength=base_mip[2])

    # mezcla por velocidad: high aparece más al tocar fuerte
    w_low  = np.interp(v_norm, [0,1], [0.78, 0.60])
    w_mid  = np.interp(v_norm, [0,1], [0.40, 0.62])
    w_high = np.interp(v_norm, [0,1], [0.06, 0.55])

    y = (w_low*y0 + w_mid*y1 + w_high*y2).astype(np.float32, copy=False)

    # envolvente + saturación
    env = adsr_env(n_samples, sr, v_norm)
    y = (y * env).astype(np.float32, copy=False)
    y = sat_tanh(y, drive=drive)

    # aplicar volumen final (sin normalizar por sample => las vel se sienten reales)
    y = (y * amp * 0.85).astype(np.float32, copy=False)

    # headroom
    return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)

# =======================
# Main generation
# =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt_dir", required=True, help="Carpeta con wavetables .wav (recursivo)")
    ap.add_argument("--out_dir", required=True, help="Carpeta de salida")
    ap.add_argument("--seed", type=int, default=0, help="0 = random real, otro = reproducible")
    ap.add_argument("--sr", type=int, default=92000)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--bitdepth", default="PCM_24", choices=["PCM_16","PCM_24","PCM_32"])
    ap.add_argument("--vel_layers", type=int, default=25)

    # roots: por defecto EXACTAMENTE 10 roots empezando en C-2 => C-2..C7
    ap.add_argument("--start_oct", type=int, default=-2, help="Octava de inicio para C (ej: -2)")
    ap.add_argument("--num_roots", type=int, default=10, help="Cuántos C por octava (ej: 10 => C-2..C7)")

    args = ap.parse_args()

    sr = int(args.sr)
    n_samples = int(sr * float(args.duration))

    rng = np.random.default_rng(None if args.seed == 0 else args.seed)

    wt_files = list_wav_files(args.wt_dir)
    if len(wt_files) < 3:
        raise RuntimeError("Necesitas al menos 3 wavetables .wav en --wt_dir")

    # 3 wavetables fijos (consistentes en todo el set)
    picks = rng.choice(wt_files, size=3, replace=False).tolist()
    wt_low, wt_mid, wt_high = picks[0], picks[1], picks[2]

    print("Wavetables (low/mid/high):")
    print("  LOW :", os.path.basename(wt_low))
    print("  MID :", os.path.basename(wt_mid))
    print("  HIGH:", os.path.basename(wt_high))

    # cargar mipmaps
    mip_low  = build_wavetable_mipmaps(load_wavetable_wav(wt_low),  levels=WT_MIP_LEVELS)
    mip_mid  = build_wavetable_mipmaps(load_wavetable_wav(wt_mid),  levels=WT_MIP_LEVELS)
    mip_high = build_wavetable_mipmaps(load_wavetable_wav(wt_high), levels=WT_MIP_LEVELS)

    # patch base fijo (consistente)
    base_pos   = [float(rng.uniform(0.0, 1.0)) for _ in range(3)]
    base_mip   = [float(rng.uniform(0.65, 1.0)) for _ in range(3)]
    base_phase = [float(rng.uniform(0.0, 1.0)) for _ in range(3)]

    # roots
    roots = [name_c_oct(args.start_oct + i) for i in range(int(args.num_roots))]

    # vel bins
    vel_bins = velocity_bins(int(args.vel_layers))

    out_dir = args.out_dir
    samples_dir = os.path.join(out_dir, "Samples")
    os.makedirs(samples_dir, exist_ok=True)

    print(f"Generando: {len(roots)} roots x {len(vel_bins)} vel = {len(roots)*len(vel_bins)} archivos")
    print(f"SR={sr}  DUR={args.duration}s  BIT={args.bitdepth}")

    for note in roots:
        note_hz = note_name_to_hz(note)

        # subcarpeta por root
        note_dir = os.path.join(samples_dir, safe_name(note))
        os.makedirs(note_dir, exist_ok=True)

        for vi, (vlo, vhi) in enumerate(vel_bins, start=1):
            v_center = 0.5 * (vlo + vhi)
            v_norm = float(np.clip(v_center / 127.0, 0.0, 1.0))

            # RNG local determinístico por nota/vel (si seed != 0)
            # si seed=0, igual queda random real, pero estable dentro del render de ese sample
            local_seed = None if args.seed == 0 else (args.seed * 1000003 + hash((note, vi)) % 2_000_000_000)
            rng_local = np.random.default_rng(local_seed)

            y = synth_one_shot(
                mip_low, mip_mid, mip_high,
                note_hz=note_hz,
                sr=sr,
                n_samples=n_samples,
                v_norm=v_norm,
                base_pos=base_pos,
                base_mip=base_mip,
                base_phase=base_phase,
                rng_local=rng_local,
            )

            fn = f"SYN_{safe_name(note)}_VEL{vi:02d}_({vlo:03d}-{vhi:03d}).wav"
            out_path = os.path.join(note_dir, fn)

            sf.write(out_path, y, sr, subtype=args.bitdepth)
            print(f"Wrote {out_path}")

    print("DONE.")

if __name__ == "__main__":
    main()

