import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from scipy.misc import factorial
from numpy.linalg import det


def cayley_menger_vr(simplex_array):
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


"""
The old ways
"""

def OLD_euclidean_distance_matrix(point_array):
	# this is the old method I wrote; too many loops!!!
	# point array is of shape (ndim + 1, ndim)
    n_points = point_array.shape[0]
    d_matrix = np.zeros((n_points, n_points), dtype=np.float64)
    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                d_matrix[i, j] = 0.
            elif j > i:
                d = np.sum((point_array[i, :] - point_array[j, :]) ** 2.)
                d_matrix[i, j] = d_matrix[j, i] = d
    return d_matrix


def OLD_pad_matrix(m):
    m = np.pad(m, ((1, 0), (1, 0)), 'constant', constant_values=1)
    m[0, 0] = 0.
    return m

def OLD_cm_volume_helper(cm_det_abs_root, n):
    return cm_det_abs_root / ((2. ** (n / 2.)) * factorial(n))


# noinspection SpellCheckingInspection
def OLD_cayley_menger_vr(point_array):
    """
    Return volume and circumradius as given by the Cayley Menger determinant
    :param point_array: np.ndarray of dimension (n+1, n) for dimension n
        These points should define a valid n-dimensional simplex of non-zero volume
    :return: tuple of floats (volume, circumradius)
    """
    n_points = point_array.shape[0] - 1 # I did this because of n-x simplices (nonhomogeneous)
    d_matrix = OLD_euclidean_distance_matrix(point_array)
    cm_det_root = np.sqrt(np.abs(det(OLD_pad_matrix(d_matrix))))
    volume = OLD_cm_volume_helper(cm_det_root, n_points)
    circumradius = np.sqrt(np.abs(det(d_matrix)) / 2.) / cm_det_root
    return volume, circumradius


# noinspection SpellCheckingInspection
def OLD_cayley_menger_volume(point_array):
    """
    Return volume as given by the Cayley Menger determinant.
    Repeats cayley_menger_vr in some aspects, but will save some time when circumradius isn't needed (Edges)
    :param point_array: np.ndarray of dimension (n+1, m) for simplex dimension n embedded in dimension m
        These points should define a valid n-dimensional simplex of non-zero volume
    :return: float volume
    """
    cm_det_root = np.sqrt(np.abs(det(OLD_pad_matrix(OLD_euclidean_distance_matrix(point_array)))))
    return OLD_cm_volume_helper(cm_det_root, point_array.shape[0] - 1)
