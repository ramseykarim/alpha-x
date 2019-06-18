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


def dendrogram(root):
	"""
	Dendrogram routine. Roughly follows v2 routine but optimized for v3.
	"""
	# Artificially set an alpha step for block height
	a_step = 0.95
	# X axis width is proportional to total number of Delaunay simplices
	base_width = len(root)
	# Start just prior to the first branch, if possible
	if root.isleaf():
		# unlikely but possible
		first_child = root
	else:
		# likely that there are multiple clusters
		first_child = max(root.children, key=lambda x: x.max_alpha())
	# set y axis limits; will adjust lower limit as tree is traversed
	lim_alpha_lo = lim_alpha_hi = first_child.max_alpha() / a_step
	# start traversing with root; reference the center of the x axis
	stack = [(root, lim_alpha_hi, base_width/2),]
	# seems like this was for a debug print statement
	count = 0
	# collect patches; order doesn't matter
	patch_stack = []
	while stack:
		count += 1
        # msg = ".. %3d ../r" % count
        # could sys.stdout.write(that)
		a, current_alpha, center = stack.pop()
		color = get_color()
		# if step includes fork alpha, split tree
		fork_alphas = [sc.max_alpha() for sc in a.children]
		# if no change in membership, don't end the patch
		stretching_patch_upward = False
		# current patch limits
		start_alpha, end_alpha = None, None
		still_iterating = True
		while still_iterating:
			end_alpha = current_alpha * a_step
			if not stretching_patch_upward:
				start_alpha = current_alpha
			# size minus # of simplices too large
			width = len(a) - np.searchsorted(a.alphas, current_alpha)
			width -= sum(len(sc) for sc in a.children if sc.max_alpha() > current_alpha)
			# should have break condition if width drops below MINIMUM_MEMBERSHIP or something
			if (a.min_alpha() < end_alpha) and (np.searchsorted(a.alphas, [current_alpha, end_alpha]).ptp() == 0):
				stretching_patch_upward = True
				continue
			true_left_edge = center - width / 2
			patch_stack.append(new_patch((true_left_edge, end_alpha), width, start_alpha - end_alpha, color))
			stretching_patch_upward = False
			# # TODO: need to remember/write all the "current fork" stuff...
			pass

		for i, alpha in enumerate(a.alphas[::-1]):
			end_alpha = alpha * a_step
			if not stretching_patch_upward:
				start_alpha = alpha
			width = len(a)

def get_color():
	# placeholder; need to implement
	return 'blue'

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
