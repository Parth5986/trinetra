from dataclasses import dataclass


@dataclass
class EnvConfig:
    n_bands = 10
    max_steps = 1000
    n_emitters = 3
    step_time_sec = 0.01

    tsrd_path = "../data/tsrd/archive/test/test_0.h5"

    min_frequency = 1000.0
    max_frequency = 12000.0

    #Receiver model

    receiver_bandwith = 1100.0
    sensitive_dbm = -100.0

    detection_threshold = 0.50

    balanced_sampling = True

    history_decay = 0.95

    #reward
    hit = +10.0
    miss = -5.0
    false_alarm = -3.0
    empty_band = -0.5


