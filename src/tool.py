import mne

def define_verbose(debug: bool):
    if not debug:
        mne.set_log_level("CRITICAL")