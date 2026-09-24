from dataclasses import dataclass
from pathlib import Path

# Absolute default: the TSRD archive lives in the repo, so derive the path
# from this file's location. This makes the env work no matter what the
# process cwd is (dataclass __init__ defaults are fixed at class creation,
# so later EnvConfig.tsrd_path = ... assignments do NOT reach new instances).
DEFAULT_TSRD_PATH = str(
    (Path(__file__).resolve().parents[1] / "data" / "tsrd" / "archive" / "test" / "test_0.h5")
)


@dataclass
class EnvConfig:
    n_bands:int = 10
    max_steps:int = 1000
    n_emitters:int = 3
    step_time_sec:int = 0.01

    tsrd_path:str = DEFAULT_TSRD_PATH

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


