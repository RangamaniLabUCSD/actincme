import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd
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

    def make_cloud_object(self):

        all_points = {'x': [], 'y': [], 'z': []}
        for i in range(len(self.x)):
            for j in range(len(self.x)):
                all_points['x'].append(self.x[i][j]) 
                all_points['y'].append(self.y[i][j]) 
                all_points['z'].append(self.z[i][j]) 
        data = pd.DataFrame(all_points)
        this_cloud_object = PyntCloud(data)
        return this_cloud_object

    def compute_curvatures(self):
        this_cloud_object = self.make_cloud_object()
        # Get k-neighbors of each point:
        k_neighbors = this_cloud_object.get_neighbors(k=4)
        # Compute the eigenvalues for each point using it's k (4 in this case) neighbours:
        ev = this_cloud_object.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
        # Compute the curvature from those eigenvalues:
        this_cloud_object.add_scalar_field("curvature", ev=ev)
        # save ply file
        this_cloud_object.to_file("out.ply")

        return this_cloud_object