import h5py
import ast
from collections import Counter
from pathlib import Path

file_path = str(Path(__file__).resolve().parent / "archive" / "test" / "test_0.h5")

frequency_modes = Counter()
pri_modes = Counter()
scan_types = Counter()

with h5py.File(file_path, "r") as f:

    transmitters = f["metadata/transmitters"]

    for transmitter_bytes in transmitters[:]:

        transmitter = ast.literal_eval(
            transmitter_bytes.decode("utf-8")
        )

        frequency_config = transmitter["frequency_config"]
        pri_config = transmitter["pri_config"]
        scan_config = transmitter["scan_config"]

        frequency_modes[
            frequency_config["freq_mode"]
        ] += 1

        pri_modes[
            pri_config["pri_mode"]
        ] += 1

        scan_types[
            scan_config["scan_type"]
        ] += 1


print("\n===== FREQUENCY MODES =====")

for mode, count in frequency_modes.items():
    print(f"{mode}: {count}")


print("\n===== PRI MODES =====")

for mode, count in pri_modes.items():
    print(f"{mode}: {count}")


print("\n===== SCAN TYPES =====")

for mode, count in scan_types.items():
    print(f"{mode}: {count}")