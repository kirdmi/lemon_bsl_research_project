from pathlib import Path
import mne
from autoreject import AutoReject


def extract_rest_state_events(raw):
    """
    Extract resting-state EO/EC markers from BrainVision annotations.

    LEMON coding:
    - Stimulus/S200 = Eyes Open
    - Stimulus/S210 = Eyes Closed
    """

    events, event_id = mne.events_from_annotations(raw)

    eo_code = event_id.get("Stimulus/S200")
    ec_code = event_id.get("Stimulus/S210")

    if eo_code is None or ec_code is None:
        raise RuntimeError("EO/EC markers not found in annotations.")

    return events, {"EO": eo_code, "EC": ec_code}


def create_epochs_from_events(raw,
                              events,
                              event_id,
                              epoch_length=4.0):
    """
    Cut long EO / EC segments into fixed-length epochs.
    """

    # Build epochs for EO and EC separately
    epochs_eo = mne.Epochs(
        raw,
        events,
        event_id=event_id["EO"],
        tmin=0,
        tmax=epoch_length,
        baseline=None,
        detrend=None,
        preload=True,
        reject_by_annotation=True
    )

    epochs_ec = mne.Epochs(
        raw,
        events,
        event_id=event_id["EC"],
        tmin=0,
        tmax=epoch_length,
        baseline=None,
        detrend=None,
        preload=True,
        reject_by_annotation=True
    )

    # Now chop long intervals into 4-second windows
    epochs_eo = epochs_eo.crop(None, None).as_data_frame  # placeholder? no
    # Actually better:
    epochs_eo = mne.make_fixed_length_epochs(
        raw,
        id=event_id["EO"],
        duration=epoch_length,
        preload=True
    )

    epochs_ec = mne.make_fixed_length_epochs(
        raw,
        id=event_id["EC"],
        duration=epoch_length,
        preload=True
    )

    return epochs_eo, epochs_ec


def apply_autoreject(epochs):
    """
    Apply AutoReject (local + global) to remove bad epochs.
    """

    ar = AutoReject(
        n_interpolate=[1, 2, 4],
        consensus=[0.80, 0.90, 1.0],
        verbose=True
    )

    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

    return epochs_clean, reject_log


def epoch_rest_state(raw,
                     epoch_length=4.0,
                     apply_ar=True):
    """
    Full pipeline:

    1. Extract EO/EC
    2. Make fixed-length epochs
    3. Autoreject (optional)

    Returns:
        epochs_eo_clean, epochs_ec_clean, logs
    """

    # Find EO / EC markers
    events, event_id = extract_rest_state_events(raw)

    # Make fixed-length 4-second epochs
    epochs_eo, epochs_ec = create_epochs_from_events(
        raw,
        events,
        event_id,
        epoch_length=epoch_length
    )

    logs = {"EO": None, "EC": None}

    # Autoreject
    if apply_ar:
        epochs_eo, logs["EO"] = apply_autoreject(epochs_eo)
        epochs_ec, logs["EC"] = apply_autoreject(epochs_ec)

    return epochs_eo, epochs_ec, logs
