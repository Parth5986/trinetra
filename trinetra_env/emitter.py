import numpy as np

class TsrdEmitters:

    def __init__(self,config: Dict,rng: np.random.Generator):
        self.config = config
        self.rng = rng

        self.name = config.get("function","unknown")

        # frequency

        self.frequency_config = config["frequency_config"]

        self.freq_mode = self.frequency_config["freq_mode"]

        self.freqs = np.asarray(self.frequency_config["freq_mhz"],dtype="f")

        self.frequency_group_size = int(self.frequency_config.get("freq_grp_size",len(self.freqs)))

        self.freq_std = float(self.frequency_config.get("freq_fixed_std_mhz",0.0))

        self.frequency_variability = float(self.frequency_config.get("freq_variability_factor",0.0))



