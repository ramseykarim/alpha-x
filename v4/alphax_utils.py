import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from scipy.misc import factorial
from numpy.linalg import det

"""
v4 alpha x utilities module
Created (along with v4): Oct 13, 2019
"""
__author__ = "Ramsey Karim"


def cayley_menger_vr(simplex_array):
    # From v3 (copy+paste)
    # Desinged to operate on the array returned from:
    #  points_array[Delaunay(points_array).simplices, :]
    # simplex_array is of shape (nsimplices, ndim + 1, ndim)
    # prepare distance matrix
    nsimplices, ndim_p1 = simplex_array.shape[0:2]
    # This will work for sets of n-x simplices embedded in dimension n with x>0
    ndim = ndim_p1 - 1
    d_matrix = np.empty((nsimplices, ndim_p1, ndim_p1))
    for i in range(nsimplices):
        d_matrix[i, :, :] = squareform(np.square(pdist(simplex_array[i, :, :])))
    cm_matrix = np.pad(d_matrix, ((0, 0), (1, 0), (1, 0)), 'constant', constant_values=1)
    cm_matrix[:, 0, 0] = 0.
    cm_det_root = np.sqrt(np.abs(det(cm_matrix)))
    volume = cm_det_root / ((2. ** (ndim/2.)) * factorial(ndim))
    circumradius = np.sqrt(np.abs(det(d_matrix)) / 2.) / cm_det_root
    return volume, circumradius
