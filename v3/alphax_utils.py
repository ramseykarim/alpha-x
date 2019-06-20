import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from scipy.misc import factorial
from random import seed, randrange
from numpy.linalg import det
import matplotlib.pyplot as plt
from matplotlib import colors as m_colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import LineCollection
from AlphaCluster import MINIMUM_MEMBERSHIP


SEED = 635541
seed(SEED)


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
	lim_alpha_lo = lim_alpha_hi = first_child.max_alpha() / (a_step**2)
	# start traversing with root; reference the center of the x axis & large-alpha bound
	stack = [(root, lim_alpha_hi, base_width/2),]
	# seems like this was for a debug print statement
	count = 0
	# collect patches; order doesn't matter
	patch_stack = []
	colors = get_color()
	while stack:
		count += 1
        # msg = ".. %3d ../r" % count
        # could sys.stdout.write(that)
		a, current_alpha, center = stack.pop()
		current_alpha /= a_step
		color = next(colors)
		# if step includes fork alpha, split tree
		fork_alphas = [sc.max_alpha() for sc in a.children]
		# if no change in membership, don't end the patch
		stretching_patch_upward = False
		# current patch limits
		start_alpha, end_alpha = None, None
		still_iterating = True
		while still_iterating:
			current_alpha *= a_step
			end_alpha = current_alpha*a_step
			if not stretching_patch_upward:
				start_alpha = current_alpha
			# width from function
			width = calc_current_width(a, current_alpha)
			# current_forks is a list of children who should get their own block next
			current_forks = [a.children[j] for j, x in enumerate(fork_alphas) if ((x < current_alpha) and (x >= end_alpha))]
			if width >= MINIMUM_MEMBERSHIP and a.min_alpha() < end_alpha:
				# if membership isn't dropping in this interval, then continue the block upwards
				interval_width = np.searchsorted(a.alphas, [current_alpha, end_alpha]).ptp()
				if (interval_width == 0) and not current_forks:
					stretching_patch_upward = True
					continue
			else:
				# minimum was reached, or smallest alpha in this interval
				still_iterating = False

			true_left_edge = center - width / 2
			if true_left_edge < 1:
				print("HEY")
				print(">>", current_forks)
				print(">>", )
			patch_stack.append(new_patch((true_left_edge, end_alpha), width, start_alpha - end_alpha, color))
			stretching_patch_upward = False
			if current_forks:
				# add up length of fork children in this interval
				total_new_width = float(sum(len(sc) for sc in current_forks))
				# add on the next iteration width of this cluster
				total_new_width += calc_current_width(a, end_alpha)
				# left edge of this block
				left_edge = 0
				for sc in [a]+current_forks:
					size = len(sc) if sc != a else calc_current_width(a, end_alpha)
					fractional_width = size / total_new_width
					new_centroid = (left_edge + fractional_width/2)*width + true_left_edge
					left_edge += fractional_width
					if sc == a:
						center = new_centroid
					else:
						stack.append((sc, end_alpha, new_centroid))
		if end_alpha < lim_alpha_lo:
			lim_alpha_lo = end_alpha
	return patch_stack, base_width, (lim_alpha_lo, lim_alpha_hi)


def calc_current_width(current_cluster, current_alpha):
	# sum up all the exceptions to width
	# first, the simplices unique to this cluster that are too large
	# searchsorted gives index in increasing sorted; if index is len(alphas), we want 0 exceptions
	not_width = len(current_cluster.alphas) - np.searchsorted(current_cluster.alphas, current_alpha)
	# second, sum up exceptions from clusters that have already branched off
	not_width += sum(len(sc) for sc in current_cluster.children if sc.max_alpha() >= current_alpha)
	return len(current_cluster) - not_width

def get_color():
	color_dict = get_colors()
	used_colors = set()
	while True:
		c = rand_color(color_dict, used_colors)
		yield c



def get_colors():
    colors = dict(m_colors.BASE_COLORS, **m_colors.CSS4_COLORS)
    return list(zip(*colors.items()))


def dark_color(rgb):
    # rgb is a tuple (r, g, b) integers
    return sum(rgb) < 600


def rand_color(c_dict, used_colors):
    h = (255, 255, 255)
    n = 0
    colors_keys, colors_values = c_dict
    while (not dark_color(h)) or (colors_keys[n] in used_colors):
        n = randrange(0, len(colors_keys))
        h = colors_values[n]
        h = (int(i * 255) for i in h) if isinstance(h, tuple) else tuple(
            int(h.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    used_colors.add(colors_keys[n])
    return colors_values[n]


def new_patch(bottom_left_corner, width, height, color):
    return plt.Rectangle(bottom_left_corner, width, height,
                         facecolor=color, edgecolor=None, linewidth=0.5
                         )


def printcluster(c, prefix=""):
	print(prefix+"---{ cluster size: ", end="")
	print(len(c))
	for sc in c.children:
		printcluster(sc, prefix=prefix+"|  ")
	print(prefix+" }")


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
