from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne
from specparam import SpectralModel


# =============================
# CONFIG
# =============================
PREPROC_DIR = Path("derivatives/eeg_preproc")
PATTERN = "*_clean-raw.fif"

TMIN_SEC = 5.0
TMAX_SEC = 125.0

FMIN_PSD, FMAX_PSD = 1.0, 45.0
FMIN_FIT, FMAX_FIT = 2.0, 40.0

WELCH_WIN_SEC = 2.0
WELCH_OVERLAP_SEC = 1.0

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (13, 30),
}

ROI = {
    "global": None,  # None = all EEG channels

    # Frontal / prefrontal
    "frontal": [
        "Fp1", "Fp2", "AF3", "AF4",
        "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    ],

    # Central / sensorimotor
    "central": [
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    ],

    # Parietal
    "parietal": [
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
        "PO3", "PO4",
    ],

    # Occipital
    "occipital": [
        "PO7", "PO8", "O1", "Oz", "O2",
    ],
}


# =============================
# HELPERS
# =============================
def subject_from_filename(path: Path) -> str:
    m = re.match(r"^(sub-[^_]+)_clean-raw\.fif$", path.name)
    return m.group(1) if m else path.stem.replace("_clean-raw", "")


def _safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def resolve_eeg_picks(raw: mne.io.BaseRaw) -> np.ndarray:
    """EEG channesl, excluding bads and no-EEG."""
    return mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, emg=False, exclude="bads")


def resolve_roi_channel_names(raw: mne.io.BaseRaw, roi_channels: list[str] | None) -> list[str]:
    eeg_picks = resolve_eeg_picks(raw)
    eeg_names = [raw.ch_names[i] for i in eeg_picks]

    if roi_channels is None:
        return eeg_names

    present = [ch for ch in roi_channels if ch in eeg_names]
    return present


def compute_psd_all_eeg(
    raw: mne.io.BaseRaw,
    tmin: float,
    tmax: float,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    PSD Welch all EEG channels, fixed time window.
      freqs: (n_freq,)
      psd:   (n_eeg_ch, n_freq)
      ch_names
    """
    eeg_picks = resolve_eeg_picks(raw)
    eeg_names = [raw.ch_names[i] for i in eeg_picks]

    sfreq = float(raw.info["sfreq"])
    n_per_seg = int(round(WELCH_WIN_SEC * sfreq))
    n_overlap = int(round(WELCH_OVERLAP_SEC * sfreq))

    n_fft = n_per_seg

    if raw.times.size > 0:
        max_time = float(raw.times[-1])
        tmax_eff = min(tmax, max_time)
    else:
        tmax_eff = tmax

    psd_obj = raw.compute_psd(
        method="welch",
        tmin=tmin,
        tmax=tmax_eff,
        fmin=fmin,
        fmax=fmax,
        picks=eeg_picks,
        n_fft=n_fft,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        verbose="ERROR",
    )

    freqs = psd_obj.freqs
    psd = psd_obj.get_data()
    return freqs, psd, eeg_names


def fit_specparam_channel(
    freqs: np.ndarray,
    psd_1ch: np.ndarray,
    fit_range: tuple[float, float],
) -> dict:
    """
    Fit SpecParam for 1 channel
    """
    sm = SpectralModel(
        peak_width_limits=(1, 12),
        max_n_peaks=6,
        min_peak_height=0.0,
        peak_threshold=2.0,
    )

    sm.fit(freqs, psd_1ch, [float(fit_range[0]), float(fit_range[1])])

    offset = _safe_float(sm.get_params("aperiodic", "offset"))
    exponent = _safe_float(sm.get_params("aperiodic", "exponent"))

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
    if peaks.size and (peaks.shape[1] != 3):
        peaks = np.empty((0, 3), dtype=float)

    r2 = None
    err = None
    for attr in ("r_squared_", "r_squared", "r2_", "r2"):
        if hasattr(sm, attr):
            r2 = _safe_float(getattr(sm, attr))
            break
    for attr in ("error_", "error", "fit_error_"):
        if hasattr(sm, attr):
            err = _safe_float(getattr(sm, attr))
            break

    return {
        "aperiodic_offset": offset,
        "aperiodic_exponent": exponent,
        "peaks": peaks,
        "r_squared": r2,
        "fit_error": err,
    }


def pick_peak_in_band(peaks: np.ndarray, band: tuple[float, float]) -> tuple[float | None, float | None, float | None]:
    """Main peak: max PW."""
    if peaks is None or len(peaks) == 0:
        return None, None, None

    lo, hi = band
    in_band = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)]
    if len(in_band) == 0:
        return None, None, None

    idx = int(np.argmax(in_band[:, 1]))
    cf, pw, bw = in_band[idx]
    return _safe_float(cf), _safe_float(pw), _safe_float(bw)


def robust_iqr(x: pd.Series) -> float:
    x = x.dropna().astype(float)
    if len(x) == 0:
        return np.nan
    return float(np.percentile(x, 75) - np.percentile(x, 25))


# =============================
# MAIN PIPELINE
# =============================
def process_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = subject_from_filename(path)

    raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")

    # PSD for all EEG channels
    freqs, psd_all, eeg_names = compute_psd_all_eeg(
        raw,
        tmin=TMIN_SEC,
        tmax=TMAX_SEC,
        fmin=FMIN_PSD,
        fmax=FMAX_PSD,
    )

    # channel-wise features
    ch_rows = []
    for ch_idx, ch_name in enumerate(eeg_names):
        psd_1ch = psd_all[ch_idx, :]

        fit = fit_specparam_channel(freqs, psd_1ch, (FMIN_FIT, FMAX_FIT))
        peaks = fit["peaks"]

        row = {
            "subject": subject,
            "channel": ch_name,
            "aperiodic_offset": fit["aperiodic_offset"],
            "aperiodic_exponent": fit["aperiodic_exponent"],
            "r_squared": fit["r_squared"],
            "fit_error": fit["fit_error"],
        }

        # band peaks
        for band_name, band_rng in BANDS.items():
            cf, pw, bw = pick_peak_in_band(peaks, band_rng)
            row[f"{band_name}_cf_hz"] = cf
            row[f"{band_name}_pw"] = pw
            row[f"{band_name}_bw_hz"] = bw

        ch_rows.append(row)

    df_ch = pd.DataFrame(ch_rows)

    # ROI mapping
    roi_map = {}
    for roi_name, roi_chs in ROI.items():
        if roi_chs is None:
            continue
        for ch in roi_chs:
            roi_map.setdefault(ch, [])
            roi_map[ch].append(roi_name)

    def _assign_roi(ch: str) -> str | None:
        lst = roi_map.get(ch, [])
        return lst[0] if lst else None

    df_ch["roi"] = df_ch["channel"].map(_assign_roi)

    df_roi = (
        df_ch.dropna(subset=["aperiodic_exponent"])
             .assign(roi=df_ch["roi"])
             .dropna(subset=["roi"])
             .groupby(["subject", "roi"], as_index=False)
             .agg(
                 exponent_median=("aperiodic_exponent", "median"),
                 exponent_mean=("aperiodic_exponent", "mean"),
                 exponent_sd=("aperiodic_exponent", "std"),
                 exponent_iqr=("aperiodic_exponent", robust_iqr),
                 offset_median=("aperiodic_offset", "median"),
                 n_channels=("aperiodic_exponent", "count"),
                 r2_median=("r_squared", "median"),
                 fit_error_median=("fit_error", "median"),
             )
    )

    df_global = (
        df_ch.dropna(subset=["aperiodic_exponent"])
             .groupby(["subject"], as_index=False)
             .agg(
                 exponent_median=("aperiodic_exponent", "median"),
                 exponent_mean=("aperiodic_exponent", "mean"),
                 exponent_sd=("aperiodic_exponent", "std"),
                 exponent_iqr=("aperiodic_exponent", robust_iqr),
                 offset_median=("aperiodic_offset", "median"),
                 n_channels=("aperiodic_exponent", "count"),
                 r2_median=("r_squared", "median"),
                 fit_error_median=("fit_error", "median"),
             )
    )
    df_global.insert(1, "roi", "global")

    df_roi = pd.concat([df_global, df_roi], ignore_index=True)

    return df_ch, df_roi


def main():
    files = sorted(PREPROC_DIR.glob(PATTERN))
    if not files:
        raise SystemExit(f"No files found: {PREPROC_DIR}/{PATTERN}")

    all_ch = []
    all_roi = []

    for f in files:
        print(f"[PROCESS] {f.name}")
        try:
            df_ch, df_roi = process_file(f)
            all_ch.append(df_ch)
            all_roi.append(df_roi)
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

    df_ch = pd.concat(all_ch, ignore_index=True) if all_ch else pd.DataFrame()
    df_roi = pd.concat(all_roi, ignore_index=True) if all_roi else pd.DataFrame()

    if not df_roi.empty:
        df_wide = df_roi.pivot(index="subject", columns="roi", values="exponent_median").add_prefix("").copy()
        df_wide.columns = [f"{c}_exponent_median" for c in df_wide.columns]
        df_wide = df_wide.reset_index()
    else:
        df_wide = pd.DataFrame()

    out_ch = PREPROC_DIR / "spectral_features_channels.csv"
    out_roi_long = PREPROC_DIR / "spectral_features_roi_long.csv"
    out_roi_wide = PREPROC_DIR / "spectral_features_roi_wide.csv"

    df_ch.to_csv(out_ch, index=False)
    df_roi.to_csv(out_roi_long, index=False)
    df_wide.to_csv(out_roi_wide, index=False)

    print(f"[DONE] Saved:")
    print(f"  - {out_ch} rows={len(df_ch)} cols={len(df_ch.columns)}")
    print(f"  - {out_roi_long} rows={len(df_roi)} cols={len(df_roi.columns)}")
    print(f"  - {out_roi_wide} rows={len(df_wide)} cols={len(df_wide.columns)}")


if __name__ == "__main__":
    main()
