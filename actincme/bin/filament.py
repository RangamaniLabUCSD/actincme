import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Filament(object):

    def __init__(self, path, filename):
        """

        """
        self.path = path
        with open(self.path + filename + '.txt', 'rb') as f:
            coords_df = pd.read_table(f, delim_whitespace=True)
            if len(coords_df.columns)==4:
                coords_df.columns=['fil', 'X', 'Y', 'Z']
                coords_df=coords_df.set_index(['fil'])
            elif len(coords_df.columns)==5:
                coords_df.columns=['object', 'fil', 'X', 'Y', 'Z']
                coords_df=coords_df.set_index(['object', 'fil'])
            else:
                print("unexpected number of columns!")
        self.filament_dataframe = coords_df
        self._filament_orientation_dataframe = []
    
    def calculate_directionality(self):

        filaments = {'filament_ID': [], 
                    'ydir': [], 
                    'zdir': [], 
                    'length': [], 
                    'x_coords': [],
                    'y_coords': [], 
                    'z_coords': []
                    }

        for filament in self.filament_dataframe.index.unique():
            cur_filament = self.filament_dataframe[self.filament_dataframe.index==filament]
            xx = cur_filament.X.values
            yy = cur_filament.Y.values
            zz = cur_filament.Z.values
            filaments['filament_ID'].append(filament)
            filaments['x_coords'].append(xx)
            filaments['y_coords'].append(yy)
            filaments['z_coords'].append(zz)

            deltaxx = sum(np.diff(xx))
            deltayy = sum(np.diff(yy))
            deltazz = sum(np.diff(zz))

            fil_length = np.sqrt(deltaxx*deltaxx+deltayy*deltayy+deltazz+deltazz)

        #     define direction theta such that 1 is up , 0 is parallel and -1 is down.
        # arcsin(z/L)
        #   take inverse so that + faces membrane (which is toward zero I think)
            ydir = np.degrees(np.arcsin(deltayy/fil_length))
            zdir = np.degrees(-(np.arcsin(deltazz/fil_length)))
        #     zdir = -np.arcsin(deltazz/fil_length)
            filaments['length'].append(fil_length)
            filaments['ydir'].append(ydir)
            filaments['zdir'].append(zdir)
        
        self._filament_orientation_dataframe = pd.DataFrame(filaments)

    def plot_filaments(self, ax):

        cmap = plt.get_cmap('hsv')
        new_cmap = self.truncate_colormap(cmap, 0, 0.75)
        colorss = cm = plt.get_cmap('seismic') 
        cNorm  = colors.Normalize(vmin=-90, vmax=90)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colorss)

        for filament in self._filament_orientation_dataframe.filament_ID.unique():
            cur_filament = self._filament_orientation_dataframe[self._filament_orientation_dataframe.filament_ID==filament]
            try:
                if len(cur_filament['x_coords'].values[0]) > 3:
                    colorVal = scalarMap.to_rgba(cur_filament['ydir'].values)
                    ax.plot(xs=cur_filament['x_coords'].values[0], ys=cur_filament['y_coords'].values[0], zs=cur_filament['z_coords'].values[0],  color=colorVal[0], linewidth=3)
            except:
                print("Error in plotting")

    def truncate_colormap(self, cmap, minval=0.0, maxval=1.0, n=100):

        new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
        return new_cmap

