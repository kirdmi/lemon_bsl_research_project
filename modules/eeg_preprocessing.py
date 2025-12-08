from pathlib import Path
import os
import mne
from mne.preprocessing import ICA


def load_raw_lemon_subject(base_dir: str | Path, subject_id: str) -> mne.io.Raw:
    """
    Upload raw EEG LEMON for one subject with link fix in .vhdr.

    base_dir   - folder - e.g. EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID
    subject_id - raw - e.g. 'sub-032301'
    """
    base_dir = Path(base_dir)
    vhdr_path = base_dir / subject_id / "RSEEG" / f"{subject_id}.vhdr"
    folder = vhdr_path.parent
    base = vhdr_path.stem  

    # read .vhdr and extract expected filenames
    with vhdr_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    expected_vmrk = None
    expected_eeg = None

    for line in lines:
        if line.startswith("MarkerFile="):
            expected_vmrk = line.strip().split("=")[1]
        elif line.startswith("DataFile="):
            expected_eeg = line.strip().split("=")[1]

    # real filenames
    real_vmrk = base + ".vmrk"
    real_eeg = base + ".eeg"

    # create symlinks
    if expected_vmrk and not (folder / expected_vmrk).exists():
        os.symlink(real_vmrk, folder / expected_vmrk)

    if expected_eeg and not (folder / expected_eeg).exists():
        os.symlink(real_eeg, folder / expected_eeg)

    # read raw data
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)

    return raw



def prepare_channels_and_montage(raw: mne.io.Raw) -> mne.io.Raw:
    """
    channels type tweaking (VEOG как EOG) + standard montage 1020
    """
    # mark VEOG as EOG-канал
    if "VEOG" in raw.ch_names:
        raw.set_channel_types({"VEOG": "eog"})

    # standard montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")

    return raw


def apply_notch(raw: mne.io.Raw, freq: float = 50.0) -> mne.io.Raw:
    """
    notch-filtering - freq (EU default - 50 Hz).
    returns raw copy.
    """
    raw_notch = raw.copy()
    raw_notch.notch_filter(freqs=[freq])
    return raw_notch


def apply_bandpass(raw: mne.io.Raw,
                   l_freq: float = 0.1,
                   h_freq: float = 45.0) -> mne.io.Raw:
    """
    filtering [l_freq, h_freq]
    returns raw copy
    """
    raw_filt = raw.copy()
    raw_filt.filter(l_freq=l_freq, h_freq=h_freq)
    return raw_filt



def fit_ica(raw_for_ica: mne.io.Raw,
            n_components: float | int | None = 0.99,
            random_state: int = 42,
            method: str = "fastica") -> ICA:
    """
    teach ICA with filtered data
    """
    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter="auto",
        fit_params=dict(fun="logcosh"),
    )

    ica.fit(raw_for_ica, reject_by_annotation="omit")

    return ica

def apply_ica_and_get_clean(raw_filt: mne.io.Raw,
                            ica: ICA,
                            exclude: list[int] | None = None) -> mne.io.Raw:
    """
    Implement ICA to filtered signal and return cleaned raw_clean.

    raw_filt - signal after notch + band-pass
    ica      - final ICA
    exclude  - excluded numbers list (if None -> no components excluded)
    """
    if exclude is None:
        exclude = []

    ica.exclude = exclude
    raw_clean = raw_filt.copy()
    ica.apply(raw_clean)
    return raw_clean

def preprocess_subject(base_dir: str | Path,
                       subject_id: str,
                       ica_exclude: list[int] | None = None
                       ) -> tuple[mne.io.Raw, ICA, mne.io.Raw, list[int]]:
    """
    Full preprocessing of LEMON subject.

    Returns tuple:
    (raw_filt, ica, raw_clean, used_exclude)

    raw_filt    - filtered signal (0.1-45) before ICA-clean
    ica         - ICA model
    raw_clean   - signal after deleting excluded components
    used_exclude - final list of excluded components (auto or manual)
    """
    raw = load_raw_lemon_subject(base_dir, subject_id)

    prepare_channels_and_montage(raw)

    raw.set_eeg_reference(ref_channels='average')

    raw_notch = apply_notch(raw, freq=50.0)

    raw_filt = apply_bandpass(raw_notch, l_freq=0.1, h_freq=45.0)

    raw_ica = raw_notch.copy()
    raw_ica.filter(l_freq=1.0, h_freq=None)
    ica = fit_ica(raw_ica, n_components=0.99, random_state=42, method="fastica")

    if ica_exclude is None:
        eog_inds, scores = ica.find_bads_eog(raw_filt, ch_name="VEOG")
        used_exclude = eog_inds
    else:
        used_exclude = ica_exclude

    raw_clean = apply_ica_and_get_clean(raw_filt, ica, exclude=used_exclude)

    return raw_filt, ica, raw_clean, used_exclude
