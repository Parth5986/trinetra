
import h5py
import ast
import pprint
from pathlib import Path

file_path = str(Path(__file__).resolve().parent / "archive" / "test" / "test_0.h5")

with h5py.File(file_path, "r") as f:

    transmitters = f["metadata/transmitters"]

    print(f"\nTotal transmitters: {len(transmitters)}")

    for i, transmitter_bytes in enumerate(transmitters[:]):

        # Convert bytes → string
        transmitter_string = transmitter_bytes.decode("utf-8")

        # Convert string → Python dictionary safely
        transmitter = ast.literal_eval(transmitter_string)

        print("\n" + "=" * 70)
        print(f"TRANSMITTER {i}")
        print("=" * 70)

        pprint.pprint(transmitter, sort_dicts=False)