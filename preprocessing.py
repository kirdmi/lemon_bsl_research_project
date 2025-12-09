from pathlib import Path
from modules.eeg_preprocessing import preprocess_subject

# load subject list from file
with open("subjects.txt", "r") as f:
    subjects = [line.strip() for line in f if line.strip()]

# output dir for preprocessed continuous data
preproc_dir = Path("derivatives/eeg_preproc")
preproc_dir.mkdir(parents=True, exist_ok=True)

base_dir = "EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID"

for sub in subjects:

    clean_path = preproc_dir / f"{sub}_clean-raw.fif"
    ica_path   = preproc_dir / f"{sub}_ica.fif"

    # --- SKIP IF BOTH FILES ALREADY EXIST ---
    if clean_path.exists() and ica_path.exists():
        print(f"[SKIP] {sub}: already processed.")
        continue

    print(f"[PROCESS] {sub}...")

    raw_filt, ica, raw_clean, used_exclude = preprocess_subject(
        base_dir=base_dir,
        subject_id=sub,
        ica_exclude=None,
    )

    print(f"{sub}: excluded ICA components: {used_exclude}")

    # save cleaned continuous data and ICA model
    raw_clean.save(clean_path, overwrite=True)
    ica.save(ica_path)

    print(f"[DONE] {sub} saved.")
