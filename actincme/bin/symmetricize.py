import numpy as np
import matplotlib.pyplot as plt
import imageio
import skimage.filters
import scipy.io as sio
import math
import pickle as pkl
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import similaritymeasures

class Symmetricize:
    """Performs necessary calculations to symmetricize irregularly shaped cell buds
    
    Attributes:
    -----------
        path: str
            relative path from where function is called to .mat files
        x_fit: np.ndarray
            x fit data extracted from file at given path
        y_fit: np.ndarray
            y fit data extracted from file at given path
        x_outerfit: np.ndarray
            x outerfit data (protein coat) extracted from file at given path
        y_outerfit: np.ndarray
            y outerfit data (protein coat) extracted from file at given path
        locs : np.ndarray
            Indices necessary for truncate_tails to determine where to truncate          
        x_fit_symmetric : np.ndarray
            x fit data fully symmetricized
        y_fit_symmetric : np.ndarray
            y fit data fully symmetricized
        x_outerfit_symmetric : np.ndarray
            x outerfit data fully symmetricized
        y_outerfit_symmetric : np.ndarray
            y outerfit data fully symmetricized
    """
    def __init__(self, path, contour, slice_start=1, slice_end=-1):
        """
        contour is current contour
        slice_range = 1:end
        """
        self.path = path
        with open(self.path + 'CCP_membrane_coordinates.pkl', 'rb') as f:
            data = pkl.load(f)
        data = data.reset_index()
        cur_contour = data[data['contour']==contour]

        self.x_fit = cur_contour['X'].to_numpy(dtype = 'float64')[slice_start:slice_end]#extracts x fit data
        self.y_fit = cur_contour['Y'].to_numpy(dtype = 'float64')[slice_start:slice_end] #extracts y fit data
        self.z_slice = cur_contour['Z'].to_numpy(dtype = 'float64')[slice_start:slice_end]

        # store original x y z coordinates
        self.x_orig=cur_contour['X'].to_numpy(dtype = 'float64')[slice_start:slice_end]
        self.y_orig=cur_contour['Y'].to_numpy(dtype = 'float64')[slice_start:slice_end]
        self.z_orig=cur_contour['Z'].to_numpy(dtype = 'float64')[slice_start:slice_end]
        
        # self.x_outerfit = mat_contents['outerfit'][:,0]
        # self.y_outerfit = mat_contents['outerfit'][:,1]
        self.locs = []
        
        self.x_fit_symmetric = []
        self.y_fit_symmetric = []
        # self.x_outerfit_symmetric = []
        # self.y_outerfit_symmetric = []
        
    def make_symmetric(self, fit_or_outerfit = "fit", rot_x = [], rot_y = []):
        """Transforms non-symmetrical curve into symmetrical one
        Parameters
        ----------
        fit_or_outerfit : str
            If equals to "fit", will make the fit data symmetrical.
            If equals to "outerfit", will symmetricize outerfit data
            If equals to "rotated", will symmetricize rotated data passed through function
        rot_x : np.ndarray (optional)
            If "rotated" select for fit_or_outerfit, then will symmetricize rotated x fit data
        rot_y : np.ndarray (optional)
            If "rotated" select for fit_or_outerfit, then wil symmetricize rotated y fit data
        
        Returns
        -------
        np.ndarray, np.ndarray
            Arrays containing the symmetricized x and y coordinates
        """
        if (fit_or_outerfit == "fit"):
            x_fit_data = self.x_fit
            y_fit_data = self.y_fit
        elif (fit_or_outerfit == "outerfit"):
            x_fit_data = self.x_outerfit
            y_fit_data = self.y_outerfit
        elif (fit_or_outerfit == "rotated"):
            x_fit_data = rot_x
            y_fit_data = rot_y
        else:
            raise ValueError("Input can only be fit, outerfit, or rotated")
            
        new_x_fit_data = np.absolute(x_fit_data) #that way we can calculate average x coord

        x_fit_data_transformed = np.zeros(len(new_x_fit_data)) #initialize resulting arrays
        y_fit_data_transformed = np.zeros(len(y_fit_data))

        x_start = np.argmin(np.absolute(0 - new_x_fit_data)) #what index has the value closest to 0 (so we know if the shape is skewed left/right)
        dist_from_mid = abs(len(new_x_fit_data) // 2 - x_start) #how far x_start is away from the actual middle of the array (used to shift resulting graph)

        #since x_start will rarely be exactly at middle of array, below expression ensures against arrayindexoutofbounds.
        for i in range(x_start if x_start <= (len(x_fit_data_transformed) // 2) else len(x_fit_data_transformed) - x_start):
            #since we want resulting figure to be perfectly symmetrical, we start from the middle of the new array and every value to left and right are the same
            x_fit_data_transformed[len(x_fit_data_transformed) // 2 - i] = - (new_x_fit_data[x_start - i] + new_x_fit_data[x_start + i]) / 2
            x_fit_data_transformed[len(x_fit_data_transformed) // 2 + i] = (new_x_fit_data[x_start - i] + new_x_fit_data[x_start + i]) / 2

        for i in range(len(y_fit_data_transformed) // 2 + 1):
            y_fit_data_transformed[i] = (y_fit_data[i] + y_fit_data[-1 - i]) / 2
            y_fit_data_transformed[-1 - i] = (y_fit_data[i] + y_fit_data[-1 - i]) / 2

        x, y = self.truncate_tails(x_fit_data_transformed, y_fit_data_transformed)
        # x, y = x_fit_data_transformed, y_fit_data_transformed
        return x, y

    def truncate_tails(self, sym_x, sym_y):
        """ Removes all instances where x coordinates = 0 in sym_x and readjusts sym_y's shape
        to match that of new sym_x
        
        Parameters
        ----------
        sym_x : np.ndarray
            Symmetricized x coordinates containing 0s at the front/end of the array
        sym_y : np.ndarray
            Symmetricized y coordinates
            
        Returns
        -------
        np.ndarray, np.ndarray
            Arrays containing the new x and y coordinates without the tails.
        """
        self.locs = np.where(sym_x == 0)[0]
        y = np.delete(sym_y, self.locs)
        x = sym_x[sym_x != 0]
        return x, y

    def curve_area(self):
        """ Approximates the area of the symmetricized outerfit protein coat (a0)   
        Parameters
        ----------
        None
        
        Returns
        -------
        int
            The approximate area of the protein coat (a0)
        """
        if (getattr(self, "x_fit_symmetric") == [] or getattr(self, "x_outerfit_symmetric") == []):
            self.do_everything("fit") #in case there is no symmetricized fit/outerfit data
            self.do_everything("outerfit")
            
        fit_length = 0
        outerfit_length = 0
        
        for i in range(len(self.x_fit_symmetric) - 1):
            fit_length += math.sqrt((self.x_fit_symmetric[i] - self.x_fit_symmetric[i + 1])**2 + 
                                    (self.y_fit_symmetric[i] - self.y_fit_symmetric[i + 1])**2) #euclidean distance
        for i in range(len(self.x_outerfit_symmetric) - 1):
            outerfit_length += math.sqrt((self.x_outerfit_symmetric[i] - self.x_outerfit_symmetric[i + 1])**2 +
                                         (self.y_outerfit_symmetric[i] - self.y_outerfit_symmetric[i + 1])**2)
        plt.close()
        return math.sqrt((outerfit_length / fit_length) * 200)
    
    def rotate_figure(self, fit_or_outerfit = "fit", plot=True):
        """ Performs PCA to determine major/minor axes and rotates figure upright according to major axis
        Parameters
        ----------
        fit_or_outerfit : str
            If equals to "fit", will rotate the fit data symmetrical. Otherwise, will rotate outerfit data
            
        Returns
        -------
        np.ndarray, np.ndarray
            Arrays containing rotated x and y coordinates
        """
        if (fit_or_outerfit == "fit"):
            x = self.x_fit
            y = self.y_fit
        elif (fit_or_outerfit == "outerfit"):
            x = self.x_outerfit
            y = self.y_outerfit
        else:
            raise ValueError("Input can only be fit or outerfit")

        x -= np.mean(x)
        y -= np.mean(y)
        coords = np.vstack([x, y])

        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)

        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
        x_v2, y_v2 = evecs[:, sort_indices[1]]

        scale = 20
        if plot is True:
            plt.plot([x_v1*-scale*2, x_v1*scale*2], [y_v1*-scale*2, y_v1*scale*2], color='red') #plots the major axis
            plt.plot([x_v2*-scale, x_v2*scale], [y_v2*-scale, y_v2*scale], color='blue') #plots the minor axis
            plt.axis('equal')
        theta = np.tanh((x_v1)/(y_v1))
        rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)], #rotation matrix
                              [np.sin(theta), np.cos(theta)]])
        transformed_mat = rotation_mat * coords

        # plot the transformed blob
        x_transformed, y_transformed = transformed_mat.A
        return x_transformed, y_transformed

    def compute_dfd(self, x1, y1, x2, y2):
        """Computes the discrete Frechet distance between two curves
        
        Parameters
        ----------
        x2 : np.ndarray
            An array containing x coordinates of the other curve
        y2 : np.ndarray
            An array containing y coordinates of the other curve
        fit_or_outerfit : str
            If equals to "fit", will use fit data for DFD. Otherwise, will use outerfit data
            
        Returns
        -------
        float
            The Frechet distance between the two curves
        """
        return similaritymeasures.frechet_dist((x1, y1), (x2, y2))

    def should_rotate(self, fit_or_outerfit = "fit", plot=True):
        """Compares the Frechet distance between 1) untransformed data and (only) symmetricized data
        to 2) rotated data and rotated+symmetricized data. If 2 is smaller, then returns true.
        
        Parameters
        ----------
        fit_or_outerfit : str
            If equals to "fit", will use fit data for DFDs. Otherwise, will use outerfit data
            
        Returns
        -------
        bool
            True if the figure has a smaller Frechet distance when rotated then symmetricized;
            false otherwise.
        """
        if (fit_or_outerfit == "fit"):
            x = self.x_fit
            y = self.y_fit 
        elif (fit_or_outerfit == "outerfit"):
            x = self.x_outerfit
            y = self.y_outerfit
        else:
            raise ValueError("Input can only be fit or outerfit")

        sym_x, sym_y = self.make_symmetric(fit_or_outerfit)
        dfd_1 = self.compute_dfd(np.delete(x, self.locs), np.delete(y, self.locs), sym_x, sym_y)

        rotated_x, rotated_y = self.rotate_figure(fit_or_outerfit, plot=plot)
        rotate_then_sym_x, rotate_then_sym_y = self.make_symmetric(fit_or_outerfit)

        dfd_2 = self.compute_dfd(np.delete(rotated_x, self.locs), np.delete(rotated_y, self.locs), rotate_then_sym_x, rotate_then_sym_y)

        return (dfd_1 > dfd_2)
    
    def curvature(self):
        """Calculates the curvature of the protein coat (c0)
        Parameters
        ----------
        None
        
        Returns
        -------
        int
            The curvature of the symmetricized protein coat (c0)
        """
        if (getattr(self, "x_fit_symmetric") == [] or getattr(self, "x_outerfit_symmetric") == []):
            self.do_everything("fit") #in case there is no symmetricized data to work off of
            self.do_everything("outerfit")      
        mid_x = self.x_fit_symmetric[len(self.x_fit_symmetric) // 2 ]
        mid_y = self.y_fit_symmetric[len(self.y_fit_symmetric) // 2 ]
        
        curv = 0
        for i in range(len(self.x_outerfit_symmetric)):
            #adds up all the 1 / distance from center to point on protein coat
            curv += 1 / math.sqrt((mid_x - self.x_outerfit_symmetric[i])**2 + (mid_y - self.y_outerfit_symmetric[i])**2)
        curv /= len(self.x_outerfit_symmetric)
        return curv
        
    def do_everything(self, ax, fit_or_outerfit = "fit", plot=True):
        """ Handles all the logic of symmetricizing a shape. 
        
        Usually, there are two scenarios for symmetricizing a shape. 1) If the shape is too skew, then usually a 
        rotation needs to be performed before symmetricizing the shape. 2) If the shape isn't, then a symmetricizing is all
        that's needed. This function handles all the logic behind that.
        
        Parameters
        ----------
        fit_or_outerfit : str
            If equals to "fit", will use fit data for calculations. Otherwise, will use outerfit data
            
        Returns
        -------
        np.ndarray, np.ndarray
            The x and y coordinate arrays of the (fully) symmetricized data
        """
        if (fit_or_outerfit not in ["fit", "outerfit"]):
            raise ValueError("Input can only be fit or outerfit")
            
        if(self.should_rotate(fit_or_outerfit, plot=plot)):
            rot_x, rot_y = self.rotate_figure(fit_or_outerfit, plot=plot)       
            rot_then_sym_x, rot_then_sym_y = self.make_symmetric("rotated", rot_x, rot_y)
            ax.plot(xs = getattr(self, "x_" + fit_or_outerfit), 
                    ys = getattr(self, "y_" + fit_or_outerfit), 
                    zs = self.z_slice, linewidth = 3)
            ax.plot(xs = rot_then_sym_x, 
                    ys = rot_then_sym_y, 
                    zs = self.z_slice[:len(rot_then_sym_x)], linewidth = 3)
            setattr(self, "x_" + fit_or_outerfit + "_symmetric", rot_then_sym_x)
            setattr(self, "y_" + fit_or_outerfit + "_symmetric", rot_then_sym_y)
            return rot_then_sym_x, rot_then_sym_y, self.z_slice[:len(rot_then_sym_x)]
        else:
            new_x2, new_y2 = self.make_symmetric(fit_or_outerfit)
            ax.plot(xs = getattr(self, "x_" + fit_or_outerfit), 
                    ys = getattr(self, "y_" + fit_or_outerfit), 
                    zs = self.z_slice, linewidth = 3)
            ax.plot(xs = new_x2, 
                    ys = new_y2, 
                    zs = self.z_slice[:len(new_x2)], linewidth = 3)
            setattr(self, "x_" + fit_or_outerfit + "_symmetric", new_x2)
            setattr(self, "y_" + fit_or_outerfit + "_symmetric", new_y2)
            return new_x2, new_y2, self.z_slice[:len(new_x2)]


    def do_everything_2d(self, fit_or_outerfit = "fit", rotate_function=True, plot=True):
        """ Handles all the logic of symmetricizing a shape. 
        
        Usually, there are two scenarios for symmetricizing a shape. 1) If the shape is too skew, then usually a 
        rotation needs to be performed before symmetricizing the shape. 2) If the shape isn't, then a symmetricizing is all
        that's needed. This function handles all the logic behind that.
        
        Parameters
        ----------
        fit_or_outerfit : str
            If equals to "fit", will use fit data for calculations. Otherwise, will use outerfit data
            
        Returns
        -------
        np.ndarray, np.ndarray
            The x and y coordinate arrays of the (fully) symmetricized data
        """
        if (fit_or_outerfit not in ["fit", "outerfit"]):
            raise ValueError("Input can only be fit or outerfit")
            
        if(self.should_rotate(fit_or_outerfit, plot=plot)) and rotate_function==True:
            rot_x, rot_y = self.rotate_figure(fit_or_outerfit, plot=plot)       
            rot_then_sym_x, rot_then_sym_y = self.make_symmetric("rotated", rot_x, rot_y)
            if plot is True:
                plt.plot(getattr(self, "x_" + fit_or_outerfit), getattr(self, "y_" + fit_or_outerfit))
                plt.plot(rot_then_sym_x, rot_then_sym_y)
            #plt.plot(rot_then_sym_x, rot_then_sym_y)
            setattr(self, "x_" + fit_or_outerfit + "_symmetric", rot_then_sym_x)
            setattr(self, "y_" + fit_or_outerfit + "_symmetric", rot_then_sym_y)
            return rot_then_sym_x, rot_then_sym_y, self.z_slice[:len(rot_then_sym_x)]
        else:
            new_x2, new_y2 = self.make_symmetric(fit_or_outerfit)
            if plot is True:
                plt.plot(getattr(self, "x_" + fit_or_outerfit), getattr(self, "y_" + fit_or_outerfit))
                plt.plot(new_x2, new_y2)
            setattr(self, "x_" + fit_or_outerfit + "_symmetric", new_x2)
            setattr(self, "y_" + fit_or_outerfit + "_symmetric", new_y2)
            return new_x2, new_y2, self.z_slice[:len(new_x2)]
        
    def get_mean_coords(self):
        """ Return mean XYZ values for transforming back to original coordinates
        

        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray
            The x y and z coordinate arrays of the original contour
        """
        mean_x = np.mean(self.x_orig)
        mean_y = np.mean(self.y_orig)
        mean_z = np.mean(self.z_slice)
        
        return mean_x, mean_y, mean_z