from typing import List

import pandas as pd
import numpy as np
import matplotlib

class Deformation_map():
    def __init__(self, path, fname) :
        self.xtrim = 0
        self.ytrim = 0
        self.path = path
        self.fname = fname
        
        self.data = pd.read_csv(self.path + '/' + self.fname, skiprows=1).to_numpy()

        self.xc = self.data[:,0]
        self.yc = self.data[:,1]
        self.xd = self.data[:,2]
        self.yd = self.data[:,3]
        
        binning_x = min(abs(np.diff(self.xc)))
        binning_y = max(abs(np.diff(self.yc)))
        assert binning_x == binning_y
        assert binning_x % 1 == 0
        self.binning = int(binning_x)
        
        self.xdim = int((self.xc.max() - self.xc.min()) / binning_x) + 1
        self.ydim = int((self.yc.max() - self.yc.min()) / binning_y) + 1
        
#         self.x_map = self._map(self.xd)
#         self.y_map = self._map(self.yd)
        self.x_map, _ = self._map_wmissing(self.xd)
        self.y_map, _ = self._map_wmissing(self.yd)
        
        self.f11 = self._grad(self.x_map)[1]
        self.f22 = self._grad(self.y_map)[0]
        self.f12 = self._grad(self.x_map)[0]
        self.f21 = self._grad(self.y_map)[1]
        self.max_shear = np.sqrt((((self.f11-self.f22)/2.)**2) + ((self.f12+self.f21)/2.)**2)
        self.max_shear = self.max_shear[self.ytrim:-self.ytrim,self.xtrim:-self.xtrim]
        self.mapshape = np.shape(self.max_shear)
        
    def _map(self, data_col):
        data_map = np.reshape(np.array(data_col), (self.ydim, self.xdim))
        return data_map

    def _map_wmissing(self, data_col):
        data_map = np.full((self.ydim, self.xdim), np.nan)
        
        xc = self.xc - self.xc.min()
        yc = self.yc - self.yc.min()
        
        locs = []
        
        for val, x, y in zip(data_col, xc, yc):
            loc = tuple(int(d / self.binning) for d in (x, y))
            if loc in locs:
                print("Multiple data values for 1 point.")
            else:
                locs.append(loc)

            data_map[loc[1], loc[0]] = val
        
        return data_map, locs

    def _grad(self,data_map) :
        data_grad = np.gradient(data_map, self.binning, self.binning)
        return data_grad
  
 
def scrubF(measurement: int, fig: matplotlib.pyplot.figure, f_list: List[Deformation_map]):
    fmap1=f_list[measurement - 1]
    fmap2=f_list[measurement]
    fmap3=f_list[measurement + 1]
    average_data=(fmap1.f22 + fmap2.f22 + fmap3.f22) / 3

    fig.set_data(average_data)
    plt.draw()
 
 
def scrubF_single(measurement: int, fig: matplotlib.pyplot.figure, f_list: List[Deformation_map]):
    fmap=f_list[measurement]
    cropped_map=fmap.f22

    fig.set_data(cropped_map)
    plt.draw()