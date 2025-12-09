# modules/eeg_features_rest.py

import numpy as np
import pandas as pd
import mne

# --- Frequency bands ---
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
}


def compute_bandpower(psd_1d, freqs, fmin, fmax):
    """
    Compute average power within a given frequency band.

    Parameters
    ----------
    psd_1d : 1D np.ndarray
        Power spectral density (n_freqs,) - already averaged over channels.
    freqs : 1D np.ndarray
        Frequency vector (Hz), length == n_freqs.
    fmin, fmax : float
        Band edges in Hz.

    Returns
    -------
    float
        Mean power in the specified band.
    """
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return np.nan
    return psd_1d[idx].mean()


def compute_peak_alpha(psd_1d, freqs, fmin=8.0, fmax=12.0):
    """
    Find peak frequency within the alpha band.

    Returns
    -------
    float
        Frequency (Hz) of the maximum power within [fmin, fmax].
    """
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return np.nan

    band_freqs = freqs[idx]
    band_psd = psd_1d[idx]
    peak_idx = np.argmax(band_psd)
    return float(band_freqs[peak_idx])


def compute_1f_slope(psd_1d, freqs, freq_range=(1.0, 40.0)):
    """
    Estimate 1/f slope by linear regression in log-log space.

    Parameters
    ----------
    psd_1d : 1D np.ndarray
        PSD averaged across channels (n_freqs,).
    freqs : 1D np.ndarray
        Frequencies (Hz).
    freq_range : tuple
        Frequency range for fitting.

    Returns
    -------
    float
        Slope of log10(PSD) ~ slope * log10(freq) + intercept.
    """
    fmin, fmax = freq_range
    idx = (freqs >= fmin) & (freqs <= fmax)

    freqs_fit = freqs[idx]
    psd_fit = psd_1d[idx]

    if len(freqs_fit) < 2 or np.any(psd_fit <= 0):
        return np.nan

    x = np.log10(freqs_fit)
    y = np.log10(psd_fit)

    # degree=1 -> linear fit, p[0] is slope, p[1] is intercept
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope)


def extract_features_from_raw(raw: mne.io.BaseRaw, subject: str) -> pd.DataFrame:
    """
    Extract simple spectral features from a cleaned continuous Raw object.

    The pipeline:
      1) Compute PSD (Welch) from 1 to 45 Hz.
      2) Average PSD across channels -> one spectrum per subject.
      3) Compute band powers, relative alpha, alpha peak, 1/f slope.

    Parameters
    ----------
    raw : mne.io.Raw
        Cleaned, continuous EEG recording (e.g., raw_clean).
    subject : str
        Subject ID.

    Returns
    -------
    df : pd.DataFrame
        Single-row dataframe with spectral features for this subject.
    """
    # Compute PSD using MNE's built-in method
    psd_obj = raw.compute_psd(
        method="welch",
        fmin=1.0,
        fmax=45.0,
        n_fft=2048,
        n_overlap=1024,
        picks="eeg",
    )
    psd = psd_obj.get_data()   # shape: (n_channels, n_freqs)
    freqs = psd_obj.freqs      # shape: (n_freqs,)

    # Average over channels -> one PSD vector per subject
    psd_mean = psd.mean(axis=0)  # (n_freqs,)

    features = {}

    # Absolute bandpowers
    for band, (lo, hi) in FREQ_BANDS.items():
        features[f"{band}_power"] = compute_bandpower(psd_mean, freqs, lo, hi)

    # Total power for relative metrics (1–30 Hz)
    total_power = compute_bandpower(psd_mean, freqs, 1.0, 30.0)
    alpha_power = features["alpha_power"]
    features["alpha_rel"] = alpha_power / total_power if total_power and total_power > 0 else np.nan

    # Alpha peak frequency
    features["alpha_peak_hz"] = compute_peak_alpha(psd_mean, freqs, 8.0, 12.0)

    # 1/f slope (approximate)
    features["aperiodic_slope"] = compute_1f_slope(psd_mean, freqs, (1.0, 40.0))

    df = pd.DataFrame([features])
    df["subject"] = subject
    return df[["subject"] + [c for c in df.columns if c != "subject"]]
