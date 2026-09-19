#To see the result of this file make the commented code below to code again and run this file.
# you will get this output for now
# -------TSRD TRANSMITTER SUMMARY--------
# FixedMultiSimultaneous: 6
# FixedSingle: 49
# HoppingSawtooth: 11
# RandomRange: 6
# RandomFixed: 7
# HoppingLinear: 9

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import h5py
from config import EnvConfig

class TsrdLoader:
    def __init__(self,h5_path:str):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"TSRD file not found on:\n{self.h5_path.resolve()} ")

        self.transmitters = self.load_transmitters()
        self.frequency_mode = self.group_frequency_mode()

    def load_transmitters(self):
        transmitters = []

        with h5py.File(self.h5_path,"r") as f:
            if "metadata/transmitters" not in f:
                raise KeyError("could not find metadata in tsrd file")

            raw_transmitters = f["metadata/transmitters"][:]
            for index,raw in enumerate(raw_transmitters):
                if isinstance(raw,bytes):   #this convert raw data which is in byte to string
                    raw = raw.decode("utf-8")

                config = ast.literal_eval(raw)

                config["dataset_index"] = index
                transmitters.append(config)

        if not transmitters:
            raise RuntimeError("No transmitter configuration found")

        return transmitters

    def group_frequency_mode(self):

        groups = defaultdict(list)

        for transmitter in self.transmitters:
            mode = transmitter["frequency_config"]["freq_mode"]
            groups[mode].append(transmitter)

        return dict(groups)

    def get_all(self):
        return self.transmitters

    def get_frequency_modes(self):
        return list(self.frequency_mode.keys())

    def summary(self):

        print("\n-------TSRD TRANSMITTER SUMMARY--------")

        for mode, emitters in self.frequency_mode.items():
            print(f"{mode}: {len(emitters)}")



# use this command to test
loader = TsrdLoader(EnvConfig.tsrd_path)
loader.summary()



