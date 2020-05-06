from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DeformationMap:
    def __init__(self, file_path: str):
        self.x_trim = 0
        self.y_trim = 0
        
        data = pd.read_csv(file_path, skiprows=1, usecols=[0, 1, 2, 3]).to_numpy()

        self.x = data[:, 0]
        self.y = data[:, 1]
        self.x_displacement = data[:, 2]
        self.y_displacement = data[:, 3]
        
        binning_x = min(abs(np.diff(self.x)))
        binning_y = max(abs(np.diff(self.y)))
        assert binning_x == binning_y
        assert binning_x % 1 == 0
        self.binning = int(binning_x)
        
        self.x_size = int((self.x.max() - self.x.min()) / self.binning) + 1
        self.y_size = int((self.y.max() - self.y.min()) / self.binning) + 1
        
        self.x_map = self.map_missing(self.x_displacement)
        self.y_map = self.map_missing(self.y_displacement)

        self.f11 = self._grad(self.x_map)[1]
        self.f22 = self._grad(self.y_map)[0]
        self.f12 = self._grad(self.x_map)[0]
        self.f21 = self._grad(self.y_map)[1]
        self.max_shear = np.sqrt((((self.f11-self.f22)/2.)**2) + ((self.f12+self.f21)/2.)**2)
        self.max_shear = self.max_shear[self.y_trim:-self.y_trim, self.x_trim:-self.x_trim]
        self.map_shape = np.shape(self.max_shear)
        
    def map_missing(self, data_col):
        data_map = np.full((self.y_size, self.x_size), np.nan)

        xc = ((self.x - self.x.min()) / self.binning).astype(int)
        yc = ((self.y - self.y.min()) / self.binning).astype(int)

        # Note the reversed x/y coords
        data_map[yc, xc] = data_col

        return data_map

    def _grad(self, data_map):
        data_grad = np.gradient(data_map, self.binning, self.binning)
        return data_grad