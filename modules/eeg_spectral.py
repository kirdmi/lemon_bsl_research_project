import numpy as np
import pandas as pd
import mne
from specparam import SpectralGroupModel


# --- Frequency bands ---
FREQ_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
}


def compute_bandpower(psd, freqs, fmin, fmax):
    """Compute average power in a given frequency range."""
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return psd[:, idx].mean(axis=1)


def compute_peak_alpha(psd, freqs):
    """Find the peak frequency within the alpha band (8–12 Hz)."""
    idx = np.logical_and(freqs >= 8, freqs <= 12)
    peak_idx = psd[:, idx].argmax(axis=1)
    return freqs[idx][peak_idx]


def compute_1f_slope(psd, freqs):
    """Estimate the 1/f slope using the specparam toolbox."""
    model = SpectralGroupModel(peak_width_limits=[1, 8])
    model.fit(freqs, psd)
    return np.array([m.aperiodic_params_[1] for m in model.get_results()])


def extract_features_from_epochs(epochs: mne.Epochs, condition: str, subject: str):
    """
    Extract spectral features for a single subject and a single condition (EO/EC).

    Returns a dataframe with per-epoch values before averaging.
    """
    psd, freqs = mne.time_frequency.psd_welch(
        epochs,
        fmin=1.0,
        fmax=45.0,
        n_fft=1000,
        average=None,
    )  # shape: (n_epochs, n_channels, n_freqs)

    # Average over channels → (n_epochs, n_freqs)
    psd = psd.mean(axis=1)

    features = {}

    # Band powers
    for band, (lo, hi) in FREQ_BANDS.items():
        features[f"{condition}_{band}_power"] = compute_bandpower(psd, freqs, lo, hi)

    # Relative alpha power
    total_power = compute_bandpower(psd, freqs, 1, 30)
    alpha_power = features[f"{condition}_alpha_power"]

    features[f"{condition}_alpha_rel"] = alpha_power / total_power

    # Peak alpha frequency
    features[f"{condition}_peak_alpha"] = compute_peak_alpha(psd, freqs)

    # 1/f slope
    features[f"{condition}_slope"] = compute_1f_slope(psd, freqs)

    df = pd.DataFrame(features)
    df["subject"] = subject
    df["condition"] = condition

    return df


def extract_subject_features(sub_id: str, epochs_dir: str):
    """
    Load EO and EC epochs for a subject and return a combined dataframe
    with averaged spectral features.
    """
    path_ec = f"{epochs_dir}/{sub_id}_ec-epo.fif"
    path_eo = f"{epochs_dir}/{sub_id}_eo-epo.fif"

    epochs_ec = mne.read_epochs(path_ec, preload=True)
    epochs_eo = mne.read_epochs(path_eo, preload=True)

    df_ec = extract_features_from_epochs(epochs_ec, "ec", sub_id)
    df_eo = extract_features_from_epochs(epochs_eo, "eo", sub_id)

    # Average over epochs → one row per condition
    df_ec_mean = df_ec.mean().to_frame().T
    df_eo_mean = df_eo.mean().to_frame().T

    return pd.concat([df_ec_mean, df_eo_mean], ignore_index=True)
