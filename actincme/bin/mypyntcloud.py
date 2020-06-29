import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
import matplotlib
from actincme.bin.symmetricize import Symmetricize
from mpl_toolkits.mplot3d import Axes3D
import pyntcloud
from pyntcloud import PyntCloud

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

class MyPyntCloud:
    """Based on this 
    https://stackoverflow.com/questions/42476839/is-gaussian-mean-curvatures-applicable-for-rough-surfaces
    """
    def __init__(self, x_meshgrid, y_meshgrid, z_meshgrid):

        self.x = x_meshgrid
        self.y = y_meshgrid
        self.z = z_meshgrid
        self._actual_pyntcloud_object = None

    def make_cloud_object(self):

        all_points = {'x': [], 'y': [], 'z': []}
        for i in range(len(self.x)):
            for j in range(len(self.x)):
                all_points['x'].append(self.x[i][j]) 
                all_points['y'].append(self.y[i][j]) 
                all_points['z'].append(self.z[i][j]) 
        data = pd.DataFrame(all_points)
        this_cloud_object = PyntCloud(data)
        self._actual_pyntcloud_object = this_cloud_object
        return this_cloud_object

    def compute_scalars(self, num_of_neighbours=4):
        this_cloud_object = self.make_cloud_object()
        # Get k-neighbors of each point:
        k_neighbors = this_cloud_object.get_neighbors(k=num_of_neighbours)
        # Compute the eigenvalues for each point using it's k (4 in this case) neighbours:
        ev = this_cloud_object.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
        # Compute the curvature from those eigenvalues:
        this_cloud_object.add_scalar_field("curvature", ev=ev)
        # Compute normals 
        this_cloud_object.add_scalar_field("normals", k_neighbors=k_neighbors)
        # save ply file
        this_cloud_object.to_file("out.ply")
        self._actual_pyntcloud_object = this_cloud_object

        return this_cloud_object
    
    def plot_curve(self, fig, ax, xlims=False, label='curvature(5)', name_of_file='pyntcloud', save=True):
        """
        Plot a specific label as a color on the mesh 
        label examples - curvature, nx, ny, nz - these scalar attributes are computed in the compute scalars method
        """

        x = self._actual_pyntcloud_object.points['x'].values
        y = self._actual_pyntcloud_object.points['y'].values
        z = self._actual_pyntcloud_object.points['z'].values

        if xlims is True:
            ax.set_zlim([0, 400])
            ax.set_ylim([400, 800])
            ax.set_xlim([900, 1300])

        color_dimension = self._actual_pyntcloud_object.points[label].values # change to desired fourth dimension
        minn, maxx = color_dimension.min(), color_dimension.max()
        norm = matplotlib.colors.Normalize(minn, maxx)
        m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)

        fcolors_meshgrid = np.zeros((len(self.x), len(self.x), 4))
        count = 0
        for i in range(len(self.x)):
            for j in range(len(self.x)):
                fcolors_meshgrid[i][j][:] = fcolors[count, :]
                count += 1

        ax.plot_surface(self.x,self.y,self.z, rstride=1, cstride=1, facecolors=fcolors_meshgrid, vmin=minn, vmax=maxx, shade=False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        cbar = fig.colorbar(m, ax=ax)
        cbar.set_label(label)

        if save is True:
            fname = name_of_file + '.png'
            plt.savefig(fname)
            plt.clf()
            plt.cla()
            plt.close()

