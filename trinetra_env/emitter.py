import math
from typing import Dict, List

import numpy as np
from config import EnvConfig

class TsrdEmitters:

    def __init__(self,config: Dict,rng: np.random.default_rng):
        self.config = config
        self.rng = rng
        self.envconfig = EnvConfig()

        self.name = config.get("function","unknown")

        # frequency

        self.frequency_config = config["frequency_config"]

        self.freq_mode = self.frequency_config["freq_mode"]

        self.freqs = np.asarray(self.frequency_config["freqs_mhz"],dtype="f")

        self.frequency_group_size = int(self.frequency_config.get("freq_grp_size",len(self.freqs)))

        self.freq_std = float(self.frequency_config.get("freq_fixed_std_mhz",0.0))

        self.frequency_variability = float(self.frequency_config.get("freq_variability_factor",0.0))

        # Pulse Width

        self.pw_config = config["pulse_width_config"]

        self.pws_us = np.asarray(self.pw_config["pws_us"],dtype = "f")

        # PRI

        self.pri_config = config["pri_config"]
        self.pri_mode = self.pri_config["pri_mode"]
        self.pri_us = np.asarray(self.pri_config["pris_us"],dtype = "f")
        self.pri_index = 0

        # Scan

        self.scan_config = config["scan_config"]
        self.scan_type = self.scan_config["scan_type"]
        self.scan_rate = float(self.scan_config.get("scan_rate_rpm",0.0))
        self.scan_start_angle = float(self.scan_config.get("scan_start_angle",0.0))
        self.beam_width_deg =  float(self.scan_config.get("beam_width_deg",0.0))

        # Power

        self.power_config = config["power_config"]
        self.power_w = float(self.power_config.get("power_w",0.0))
        self.gain = float(self.power_config.get("gain",0.0))

        # Position

        self.position_config = config["position_config"]
        self.start_position_km = self.position_config.get("start_position_km",(100.0,100.0))
        self.travel_angle_deg = float(self.position_config.get("travel_angle_deg",0.0))
        self.speed_km_s = float(self.position_config.get("speed_km_s",0.0))
        self.position = np.array(self.start_position_km,dtype = "f")

        self.step_count = 0
        self.frequency_index = 0

    def get_active_frequencies(self) -> List[float]:
        self.mode = self.freq_mode

        if self.mode == "FixedSingle":
            frequency = self.freqs[0]+self.rng.uniform(0.0,self.freq_std)
            return [frequency]

        elif self.mode == "FixedMultiSimultaneous":
            freq_fms = []
            for freq in self.freqs:
                freq = freq + self.rng.uniform(0.0,self.freq_std)
                freq_fms.append(float(freq))
            return freq_fms

        elif self.mode == "HoppingSawtooth":
            freq_hs = self.rng.choice(self.freqs) + self.rng.uniform(0.0,self.freq_std)

            return [freq_hs]

        elif self.mode == "HoppingLinear":
            index = self.frequency_index%len(self.freqs)
            freq_hl = self.freqs[index] + self.rng.uniform(0.0,self.freq_std)

            return [freq_hl]

        elif self.mode == "RandomRange":
            freq_rr = self.rng.uniform(self.freqs[0],self.freqs[1]) + self.rng.uniform(0.0,self.freq_std)

            return [float(freq_rr)]

        elif self.mode == "RandomFixed":
            freq_rf = self.rng.choice(self.freqs) + self.rng.uniform(0.0,self.freq_std)

            return [float(freq_rf)]

        else: [float(self.freqs[0])]

    def get_current_pri(self):

        mode = self.pri_mode

        if mode == "Fixed":
            return float(self.pri_us[0])

        elif mode == "Staggered":
            index = self.pri_index % len(self.pri_us) # in this we are making array circular so the values repeat
            return float(self.pri_us[index])

        elif mode == "Jittered":
            base = self.rng.choice(self.pri_us)
            jitter = base + self.rng.uniform(0.0,base*0.05)
            return jitter

        elif mode == "Sliding":
            # in this we are using sinosodal wave for sliding which changes when step_count changes and the peak if pri_us[1]
            sliding = self.pri_us[0] + ((math.sin(self.step_count*0.05)+1.0)/0.02) * (self.pri_us[1] - self.pri_us[0])
            return sliding

        elif mode == "SwitchDwell":
            dwell_steps = 2
            index = (self.step_count//dwell_steps)%len(self.pri_us)
            return float(self.pri_us[index])

        else: return float(self.pri_us[0])

    def get_scan_visibility(self,time_s:float):

        mode = self.scan_type

        if mode == "Omni":
            return 1.0

        if self.scan_rate<=0:
            return 1.0

        angle = (self.scan_start_angle - (self.scan_rate * 360.0 * time_s) / 60.0) % 360.0

        receiver_angle = 0.0

        difference = (angle - receiver_angle + 180.0) % 360.0 - 180.0

        half_beam = self.beam_width_deg / 2.0

        if difference<=half_beam:
            return 1.0

        return 0.05

    def get_signal_power_dbm(self,time_s:float):

        distance_km = max(float(np.linalg.norm(self.start_position_km)),1.0)

        power_dbm = 10.0 * math.log10(max(self.power_w,1e-12) * 1000.0)

        path_loss = 20.0 * math.log10(distance_km) + 80.0 #this is only approximation later make it more physical by applying real path loss formula

        scan_visibility = self.get_scan_visibility(time_s)

        visibility_db = 10.0 * math.log10(max(scan_visibility,1e-12))

        received_power_dbm = power_dbm + self.gain - path_loss + visibility_db

        return float(received_power_dbm)

    def step(self,dt:float):

        self.step_count += 1
        if self.freq_mode in ("HoppingSawtooth","HoppingLinear"):
            self.frequency_index += 1

        self.pri_index += 1

        #update position

        angle_rad = math.radians(self.travel_angle_deg)

        self.position[0] += (math.cos(angle_rad) * self.speed_km_s * dt)

        self.position[1] += (math.sin(angle_rad) * self.speed_km_s * dt)










