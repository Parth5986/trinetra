from dataclasses import dataclass


@dataclass
class EnvConfig:
    n_bands:int = 10
    max_steps:int = 1000
    n_emitters:int = 3
    step_time_sec:int = 0.01

    tsrd_path:str = "../data/tsrd/archive/test/test_0.h5"

    min_frequency:float = 1000.0
    max_frequency:float = 12000.0

    #Receiver model

    receiver_bandwith:float = 1100.0
    sensitive_dbm:float = -100.0

    detection_threshold:float = 0.50

    balanced_sampling:bool = True

    history_decay:float = 0.95

    #reward
    hit:float = +10.0
    miss:float = -5.0
    false_alarm:float = -3.0
    empty_band:float = -0.5


