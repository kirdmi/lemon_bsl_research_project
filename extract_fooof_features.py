from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne
from specparam import SpectralModel

# -----------------------------
# CONFIG
# -----------------------------
PREPROC_DIR = Path("derivatives/eeg_preproc")  # где лежат *_clean-raw.fif
PATTERN = "*_clean-raw.fif"

# Частотный диапазон PSD и фитта 1/f
FMIN_PSD, FMAX_PSD = 1.0, 45.0
FMIN_FIT, FMAX_FIT = 2.0, 40.0  # обычно безопаснее для slope
N_FFT = 4096
N_OVERLAP = 2048

# ROI (можешь менять под свою схему каналов)
ROI = {
    "global": None,  # None = все EEG-каналы
    "frontal": ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8"],
    "central": ["Cz", "C3", "C4"],
    "parietal": ["Pz", "P3", "P4"],
    "occipital": ["Oz", "O1", "O2"],
}

# Диапазоны для поиска “основного” пика
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (13, 30),
}

def subject_from_filename(path: Path) -> str:
    # sub-032301_clean-raw.fif -> sub-032301
    m = re.match(r"^(sub-[^_]+)_clean-raw\.fif$", path.name)
    return m.group(1) if m else path.stem.replace("_clean-raw", "")

def compute_mean_psd(raw: mne.io.BaseRaw, picks: list[int], fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    # PSD по каналам -> усредняем по каналам (и только EEG)
    psd = raw.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=N_FFT,
        n_overlap=N_OVERLAP,
        picks=picks,
        verbose="ERROR",
    )
    freqs = psd.freqs
    data = psd.get_data()  # shape (n_ch, n_freq)
    mean_psd = np.mean(data, axis=0)
    return freqs, mean_psd

def fit_aperiodic_and_peaks(freqs: np.ndarray, psd: np.ndarray, fit_range: tuple[float, float]) -> dict:
    sm = SpectralModel(
        peak_width_limits=(1, 12),
        max_n_peaks=6,
        min_peak_height=0.0,
        peak_threshold=2.0,
    )
    sm.fit(freqs, psd, list(fit_range))

    # aperiodic
    offset = float(sm.get_params("aperiodic", "offset"))
    exponent = float(sm.get_params("aperiodic", "exponent"))

    # peaks: rows [CF, PW, BW]
    peaks = None
    for key in ("peak", "periodic"):
        try:
            peaks = sm.get_params(key)
            break
        except Exception:
            continue

    if peaks is None:
        peaks = np.empty((0, 3), dtype=float)

    peaks = np.asarray(peaks, dtype=float)
    if peaks.ndim == 1:
        peaks = peaks.reshape(0, 3) if peaks.size == 0 else peaks.reshape(1, 3)

    # строгая проверка формы
    if peaks.size and (peaks.shape[1] != 3):
        peaks = np.empty((0, 3), dtype=float)

    return {
        "offset": offset,
        "exponent": exponent,
        "peaks": peaks,
    }

def pick_peak_in_band(peaks: np.ndarray, band: tuple[float, float]) -> tuple[float | None, float | None, float | None]:
    """
    Выбираем “главный” пик в диапазоне (по максимальному PW).
    Возвращаем (CF, PW, BW) или (None, None, None)
    """
    if peaks is None or len(peaks) == 0:
        return None, None, None

    lo, hi = band
    in_band = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)]
    if len(in_band) == 0:
        return None, None, None

    # главный пик = максимальная мощность пика над фоном (PW)
    idx = np.argmax(in_band[:, 1])
    cf, pw, bw = in_band[idx]
    return float(cf), float(pw), float(bw)

def resolve_roi_picks(raw: mne.io.BaseRaw, roi_channels: list[str] | None) -> list[int]:
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, exclude="bads")
    if roi_channels is None:
        return list(eeg_picks)

    present = [ch for ch in roi_channels if ch in raw.ch_names]
    if len(present) == 0:
        return []  # ROI отсутствует у данного субъекта
    picks = mne.pick_channels(raw.ch_names, include=present)
    # пересечение с EEG picks
    picks = [p for p in picks if p in set(eeg_picks)]
    return picks

def process_file(path: Path) -> list[dict]:
    subject = subject_from_filename(path)

    raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    # Если reference уже применён в preprocessing — хорошо. Если нет, можно включить:
    # raw.set_eeg_reference("average", projection=True)

    rows: list[dict] = []

    for roi_name, roi_chs in ROI.items():
        picks = resolve_roi_picks(raw, roi_chs)
        if len(picks) == 0:
            continue

        freqs, mean_psd = compute_mean_psd(raw, picks, FMIN_PSD, FMAX_PSD)

        fit = fit_aperiodic_and_peaks(freqs, mean_psd, (FMIN_FIT, FMAX_FIT))
        peaks = fit["peaks"]

        out = {
            "subject": subject,
            "roi": roi_name,
            "aperiodic_offset": fit["offset"],
            "aperiodic_exponent": fit["exponent"],  # это и есть 1/f slope (в терминах SpecParam)
        }

        # Пики по диапазонам
        for band_name, band_rng in BANDS.items():
            cf, pw, bw = pick_peak_in_band(peaks, band_rng)
            out[f"{band_name}_cf_hz"] = cf
            out[f"{band_name}_pw"] = pw
            out[f"{band_name}_bw_hz"] = bw

        rows.append(out)

    return rows

def main():
    files = sorted(PREPROC_DIR.glob(PATTERN))
    if not files:
        raise SystemExit(f"No files found: {PREPROC_DIR}/{PATTERN}")

    all_rows: list[dict] = []
    for f in files:
        print(f"[PROCESS] {f.name}")
        try:
            all_rows.extend(process_file(f))
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

    df = pd.DataFrame(all_rows)

    out_csv = PREPROC_DIR / "spectral_features_specparam.csv"
    df.to_csv(out_csv, index=False)
    print(f"[DONE] Saved: {out_csv} rows={len(df)} cols={len(df.columns)}")

if __name__ == "__main__":
    main()
