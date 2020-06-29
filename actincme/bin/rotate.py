import matplotlib.pyplot as plt
import logging
import numpy as np
from actincme.bin.symmetricize import Symmetricize
from mpl_toolkits.mplot3d import Axes3D

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

class Rotate:
    """Rotate an axisymmetric curve to 3D and plot it
    """
    def __init__(self, x, y, z, mean_x, mean_y):
        """
        contour is current contour
        slice_range = 1:end
        here x and y are normalized, z is not
        we need mean_x and mean_y information from original data before it was symmetricized
        """
        self.x = x[len(x)//2:]
        self.y = y[len(y)//2:]
        self.z = z
        self.mean_x = mean_x
        self.mean_y = mean_y 
        xs, ys, zs = self.rotate_steps()
        self._x3d_norm = xs + self.mean_x
        self._y3d_norm = zs.T + self.mean_y
        self._z3d_norm = ys + np.mean(self.z)

    def rotate_steps(self):

        revolve_steps = np.linspace(0, np.pi*2, len(self.x)).reshape(1,len(self.x))
        theta = revolve_steps
        #convert rho to a column vector
        rho_column = self.x.reshape(len(self.x),1)
        x = rho_column.dot(np.cos(theta))
        y = rho_column.dot(np.sin(theta))
        # # expand z into a 2d array that matches dimensions of x and y arrays..
        zs, rs = np.meshgrid(self.y, self.x)
        self._zs = zs
        return x, y, zs

    def rotate_single_curve(self, ax, xlims=False, save=False):

        x, y, zs = self.rotate_steps()
        if xlims is True:
            ax.set_zlim([0, 400])
            ax.set_ylim([400, 800])
            ax.set_xlim([900, 1300])
        ax.plot_surface(x + self.mean_x, self._zs.T + self.mean_y, y + np.mean(self.z), shade=True)

        if save is True:
            fname = '_tmp%05d.png' % int(self.z[0])
            plt.savefig(fname)
            plt.clf()
            plt.cla()
            plt.close()

class AverageRotate():
    
    def __init__(self, path, start_list, end_list):
        """
        Average tomogram slices located in path, return a single rotate object
        start_list and end_list are manually determined selections
        """

        self.path = path
        self.start_list = start_list
        self.end_list = end_list

    def rotate_many_curves(self, cutoff_value=1, not_includes = [13, 17, 18, 20, 22, 24, 26]):
        """
        Handles logic for averaging all curves
        cutoff value included in case we want to average curves that have a specific number of elements in them
        default is 1 (so we average all curves)
        """

        for j, i in enumerate(range(len(self.start_list))):
            # manually determined
            if i not in not_includes:
                shape = Symmetricize(self.path, i+1, self.start_list[i], self.end_list[i])
                this_x, this_y, this_z = shape.do_everything_2d("fit", plot=False)
                mean_x, mean_y = shape.get_mean_coords()
                # No longer normalized
                this_x = this_x + mean_x
                this_y = this_y + mean_y
                if j == 0:
                    # chose arbitrary length of 40, must be large enough
                    max_length = 40
                    all_x = np.empty((0,max_length), int)
                    all_y = np.empty((0,max_length), int)
                    all_z = np.empty((0,max_length), int)
                # I chose 28 as a cutoff length because i only want to include files that have atleast 29 points

                if len(this_x) > cutoff_value:
                    this_x = np.pad(this_x, (max_length - len(this_x), 0), 'constant', constant_values=(np.NaN, np.NaN))
                    this_y = np.pad(this_y, (max_length - len(this_y), 0), 'constant',constant_values=(np.NaN, np.NaN))
                    this_z = np.pad(this_z, (max_length - len(this_z), 0), 'constant', constant_values=(np.NaN, np.NaN))
                    # this_x = np.pad(this_x, (max_length - len(this_x), 0), 'constant')
                    # this_y = np.pad(this_y, (max_length - len(this_y), 0), 'constant')
                    # this_z = np.pad(this_z, (max_length - len(this_z), 0), 'constant')
                    all_x = np.append(all_x, [this_x], axis=0)
                    all_y = np.append(all_y, [this_y], axis=0)
                    all_z = np.append(all_z, [this_z], axis=0)
        
        # Average across all shapes (if NaN and element, choose element)
        avg_x = np.nanmean(all_x, axis=0)
        avg_y = np.nanmean(all_y, axis=0)
        avg_z = np.nanmean(all_z, axis=0)

        # Remove NaNs from padding
        avg_x = avg_x[~np.isnan(avg_x)]
        avg_y = avg_y[~np.isnan(avg_y)]
        avg_z = avg_z[~np.isnan(avg_z)]

        # Find average x and y to normalize data
        mean_of_avg_x = np.mean(avg_x)
        mean_of_avg_y = np.mean(avg_y)

        # Create rotate object
        this_rotate = Rotate(avg_x - mean_of_avg_x, avg_y - mean_of_avg_y, avg_z, mean_of_avg_x , mean_of_avg_y)

        return this_rotate

    def plot_averaged_curve(self, ax, cutoff_value=35, not_includes = [13, 17, 18, 20, 22, 24, 26], xlims=False, save=False):

        # Do the averaging, return rotate object
        this_rotate = self.rotate_many_curves(cutoff_value, not_includes)

        # Plot it
        this_rotate.rotate_single_curve(ax, xlims, save)

