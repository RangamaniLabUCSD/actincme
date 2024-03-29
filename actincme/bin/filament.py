import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
import matplotlib.cm as cmx
from numpy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Filament(object):

    def __init__(self, path, filename):
        """
        Path where the BranchedActinCoordinates_Integers txt file is
        filename is BranchedActinCoordinates_Integers
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
    
    def calculate_directionality(self, num_of_neighbours=5, rotated_surface=None):
        """
        Compute directionality of filaments
        Out - filament dataframe containing 
        filament_ID, ydir, zdir, length, x_coords, y_coords, z_coords
        If rotated_surface object is provided, computes orientation relative to that
        outputs extra columns in filament dataframe
        Note - For the unbranched filaments, it is currently arbitrary which coordinate is 1st versus last
        """

        if rotated_surface is not None:
            x_meshgrid = rotated_surface._x3d_norm
            y_meshgrid = rotated_surface._y3d_norm
            z_meshgrid = rotated_surface._z3d_norm
            filaments = {'filament_ID': [], 
                        'ydir': [], 
                        'zdir': [], 
                        'normal_angle': [], 
                        'length': [], 
                        'x_coords': [],
                        'y_coords': [], 
                        'z_coords': []
                        }
        else:
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

            first_point = [xx[-1], yy[-1], zz[-1]]
            closest_distance = np.inf
            closest_points = []
            all_points = []
            
            for i in range(len(x_meshgrid)):
                for j in range(len(x_meshgrid)):
                    this_point = [x_meshgrid[i][j], y_meshgrid[i][j], z_meshgrid[i][j]]
                    all_points.append(this_point)
                    this_distance = np.linalg.norm(np.subtract(first_point, this_point))
                    if this_distance < closest_distance:
                        closest_distance = this_distance
                        closest_points.append(this_point)

            deltaxx = sum(np.diff(xx))
            deltayy = sum(np.diff(yy))
            deltazz = sum(np.diff(zz))

            # Lets take the 5 closest points and compute a plane going through it
            close_points = np.zeros([3, num_of_neighbours])
            for i in range(num_of_neighbours):
                close_points[:,i] = np.array(closest_points[-1-i]).reshape(3,)

            # vector1 = np.subtract(second_closest_point, closest_point)
            # vector2 = np.subtract(third_closest_point, closest_point)

            # # the cross product is a vector normal to the plane
            # cp = np.cross(vector1, vector2)

            # # This evaluates a * x3 + b * y3 + c * z3 which equals d
            # d = np.dot(cp, closest_point)
            # # print('The equation of the plane is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
            center, normal = self.planeFit(close_points)
            filament_vector = np.array([deltaxx, deltayy, deltazz])

            if rotated_surface is not None:
                unit_vector_1 = normal / np.linalg.norm(normal)
                unit_vector_2 = filament_vector / np.linalg.norm(filament_vector)
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                normal_angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
                # surface_normal = cp
                # rel_x = surface_normal[0] - filament_vector[0]
                # rel_y = surface_normal[1] - filament_vector[1]
                # rel_z = surface_normal[2] - filament_vector[2]
                # normalization_length = np.linalg.norm(np.array([rel_x, rel_y, rel_z]))
                # ydir_rel = np.degrees(np.arcsin(rel_y/normalization_length))
                # zdir_rel = np.degrees(-(np.arcsin(rel_z/normalization_length)))
                # degrees = np.degrees(
                #             np.arccos(
                #                 np.clip(
                #                     np.dot(
                #                         filament_vector/np.linalg.norm(filament_vector), 
                #                         surface_normal/np.linalg.norm(surface_normal)
                #                         ),
                #                     -1.0, 1.0)
                #                     )
                #                     )
                # print(degrees)
                filaments['normal_angle'].append(normal_angle)


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

    def planeFit(self,points):
        """
        Based on this -> https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
        p, n = planeFit(points)

        Given an array, points, of shape (d,...)
        representing points in d-dimensional space,
        fit an d-dimensional plane to the points.
        Return a point, p, on the plane (the point-cloud centroid),
        and the normal, n.
        """
        points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
        assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
        ctr = points.mean(axis=1)
        x = points - ctr[:,np.newaxis]
        M = np.dot(x, x.T) # Could also use np.cov(x) here.
        return ctr, svd(M)[0][:,-1]

    def plot_filaments(self, fig, ax, dir='ydir'):
        """
        Plot specified filament orientation on specified axes 
        """

        cmap = plt.get_cmap('hsv')
        new_cmap = self.truncate_colormap(cmap, 0, 0.75)
        colorss = cm = plt.get_cmap('seismic') 
        cNorm  = colors.Normalize(vmin=-90, vmax=90)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colorss)

        for filament in self._filament_orientation_dataframe.filament_ID.unique():
            cur_filament = self._filament_orientation_dataframe[self._filament_orientation_dataframe.filament_ID==filament]
            try:
                if len(cur_filament['x_coords'].values[0]) > 3:
                    if dir == 'ydir':
                        colorVal = scalarMap.to_rgba(cur_filament['ydir'].values)
                    elif dir == 'normal_angle':
                        cNorm  = colors.Normalize(vmin=0, vmax=180)
                        colorVal = scalarMap.to_rgba(cur_filament['normal_angle'].values)
                    elif dir == 'zdir':
                        colorVal = scalarMap.to_rgba(cur_filament['zdir'].values)
                    ax.plot(xs=cur_filament['x_coords'].values[0], ys=cur_filament['y_coords'].values[0], zs=cur_filament['z_coords'].values[0],  color=colorVal[0], linewidth=3)
            except:
                print("Error in plotting")
#         cbar = fig.colorbar(cmx.ScalarMappable(norm=cNorm, cmap=new_cmap), ax=ax)
#         cbar.set_label(dir, rotation=270)

    def truncate_colormap(self, cmap, minval=0.0, maxval=1.0, n=100):
        """
        Colormap truncation
        """

        new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    def plot_filaments_and_shape(self,i, fig, ax, rotated_figure, direction='normal_angle'):
        """
        Plot both filaments and rotated shape
        """

        rotated_figure.rotate_single_curve(ax, xlims=False, save = False)
        self.plot_filaments(fig, ax, direction)
        fname = '_tmp%05d.png' % int(i)
        plt.savefig(fname)


