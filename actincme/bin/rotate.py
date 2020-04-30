import matplotlib.pyplot as plt
import numpy as np
from actincme.bin.symmetricize import Symmetricize
from mpl_toolkits.mplot3d import Axes3D

class Rotate:
    """Rotate an axisymmetric curve to 3D and plot it
    """
    def __init__(self, x, y, z):
        """
        contour is current contour
        slice_range = 1:end
        """
        self.x = x[len(x)//2:]
        self.y = y[len(y)//2:]
        self.z = z

    def rotate_single_curve(self, save=False):

        revolve_steps = np.linspace(0, np.pi*2, len(self.x)).reshape(1,len(self.x))
        theta = revolve_steps
        #convert rho to a column vector
        rho_column = self.x.reshape(len(self.x),1)
        x = rho_column.dot(np.cos(theta))
        y = rho_column.dot(np.sin(theta))
        # # expand z into a 2d array that matches dimensions of x and y arrays..
        # i used np.meshgrid
        zs, rs = np.meshgrid(self.y, self.x)

        #plotting
        fig, ax = plt.subplots(figsize=[16,8], subplot_kw=dict(projection='3d'))
        fig.tight_layout(pad = 0.0)
        #transpose zs or you get a helix not a revolve.
        # you could add rstride = int or cstride = int kwargs to control the mesh density
        ax.plot_surface(x, y, zs.T, shade = True)

        #view orientation
        ax.elev = 30 #30 degrees for a typical isometric view
        ax.azim = 30
        ax.set_zlim([-200, 150])
        ax.set_ylim([-200, 200])
        ax.set_xlim([-250, 250])
        ax.set_aspect('equal')
        #turn off the axes to closely mimic picture in original question
        # ax.set_axis_off()

        if save is True:
            fname = '_tmp%05d.png' % int(self.z[0])
            plt.savefig(fname)
            plt.clf()
            plt.cla()
            plt.close()
        else:
            plt.show()
            
#         matt trying to return XYZ info
        self.x3d = x
        self.y3d = y
        self.z3d = zs.T

        
