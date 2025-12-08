from modules.eeg_preprocessing import preprocess_subject
from modules.eeg_epoching import epoch_rest_state


base_dir = "EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID"
subjects = ["sub-032301"]

for sub in subjects:
    raw_filt, ica, raw_clean, used_exclude = preprocess_subject(
        base_dir=base_dir,
        subject_id=sub,
        ica_exclude=None,
    )

    print(f"{sub}: excluded ICA components: {used_exclude}")

    raw_clean.save(f"derivatives/eeg_preproc/{sub}_clean-raw.fif", overwrite=True)
    ica.save(f"derivatives/eeg_preproc/{sub}_ica.fif")


    # Epoching
    epochs_eo, epochs_ec, logs = epoch_rest_state(
        raw_clean,
        epoch_length=4.0,
        apply_ar=True
    )

    print(f"{sub}: EO epochs = {len(epochs_eo)}, EC epochs = {len(epochs_ec)}")

    epochs_eo.save(f"derivatives/eeg_epochs/{sub}_eo-epo.fif", overwrite=True)
    epochs_ec.save(f"derivatives/eeg_epochs/{sub}_ec-epo.fif", overwrite=True)
