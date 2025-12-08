from modules.preprocessing import preprocess_subject

base_dir = "EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID"
subjects = ["sub-032301", "sub-032302", "sub-032303"]  # и т.д.

for sub in subjects:
    print(f"Processing {sub}...")

    raw_filt, ica, raw_clean, used_exclude = preprocess_subject(
        base_dir=base_dir,
        subject_id=sub,
        ica_exclude=None,   # <-- автоматический поиск по EOG
    )

    print(f"{sub}: excluded components {used_exclude}")

    # при желании – сохранить
    # raw_clean.save(f"derivatives/eeg_preproc/{sub}_clean-raw.fif", overwrite=True)
    # ica.save(f"derivatives/eeg_preproc/{sub}_ica.fif")
