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
from collections import deque

SEED = 635541
seed(SEED)

DEFAULT_TRANSPARENCY = 0.4
LOW_TRANSPARENCY = 0.2


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
    # Start just prior to the first branch, if possible
    if root.isleaf():
        # unlikely but possible
        first_child = root
        # set y axis limits; will adjust lower limit as tree is traversed
        lim_alpha_lo = lim_alpha_hi = root.max_alpha() / (a_step**2)
    else:
        # likely that there are multiple clusters
        first_child = max(root.children, key=root.children.get)
        lim_alpha_lo = lim_alpha_hi = root.children[first_child] / (a_step**2)
    # X axis width is proportional to # simplices at this alpha
    base_width = root.width_less_than(lim_alpha_lo)
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
        # it will get stepped forward immediately, so step it back
        current_alpha /= a_step
        a.set_color(next(colors))
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
            width = a.width_less_than(current_alpha)
            # current_forks is a list of children who should get their own block next
            current_forks = [sc for sc in a.children if end_alpha <= a.children[sc] < current_alpha]
            # current_forks = [a.children[j] for j, x in enumerate(fork_alphas) if ((x < current_alpha) and (x >= end_alpha))]
            if width >= MINIMUM_MEMBERSHIP and a.min_alpha() < end_alpha:
                # if membership isn't dropping in this interval, then continue the block upwards
                interval_width = a.width_less_than(current_alpha) - a.width_less_than(end_alpha)
                if (interval_width == 0) and not current_forks:
                    # I checked that this feature works!
                    stretching_patch_upward = True
                    continue
            else:
                # minimum was reached, or smallest alpha in this interval
                still_iterating = False
            true_left_edge = center - width / 2
            patch_stack.append(new_rect_patch((true_left_edge, end_alpha), width, start_alpha - end_alpha, a.get_color()))
            stretching_patch_upward = False
            if current_forks:
                # add up length of fork children in this interval
                total_new_width = float(sum(len(sc) for sc in current_forks))
                # add on the next iteration width of this cluster
                total_new_width += a.width_less_than(end_alpha)
                # import pdb; pdb.set_trace()
                # left edge of this block
                left_edge = 0
                for sc in [a]+current_forks:
                    size = len(sc) if sc != a else a.width_less_than(end_alpha)
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


def new_rect_patch(bottom_left_corner, width, height, color):
    return plt.Rectangle(bottom_left_corner, width, height,
                         facecolor=color, edgecolor=None, linewidth=0.5,
                         alpha=0.9
                         )


def naive_point_grouping(root, dkey):
    # stack is guaranteed to have parents first
    stack = depth_first_search(root)
    plot_surfaces, plot_points = [], []
    used_points = set()
    for a in stack:
        if a == root:
            if a.isleaf():
                # for a lone root, use geometric mean of alpha range
                alpha = np.sqrt(a.min_alpha() * a.max_alpha())
            else:
                # for the root, use alpha of last branch
                alpha = min(a.children[sc] for sc in a.children)
        else:
            # not root; just some subcluster
            alpha = a.alphas_main[2*(len(a.alphas_main)//4)]
        # points may share vertices with subclusters
        points, boundary = a.cluster_at_alpha(alpha, dkey)
        plot_surfaces.append(generate_boundary_artist(boundary, a.get_color()))
        # keep track of the points that are assigned to smaller clusters
        points -= used_points
        used_points |= points
        plot_points.append((tuple(zip(*points)), a.get_color(), DEFAULT_TRANSPARENCY))
    return plot_surfaces, plot_points


def alpha_surfaces(root, dkey, alpha):
    plot_surfaces, plot_points = [], []
    used_points = set()
    stack = [root]
    while stack:
        a = stack.pop()
        if a.min_alpha() <= alpha < a.max_alpha():
            points, boundary = a.cluster_at_alpha(alpha, dkey)
            plot_surfaces.append(generate_boundary_artist(boundary, a.get_color()))
            used_points |= points
            plot_points.append((tuple(zip(*points)), a.get_color(), DEFAULT_TRANSPARENCY))
        for sc in a.children:
            if a.children[sc] > alpha:
                stack.append(sc)
    leftovers = set(map(tuple, dkey.points)) - used_points
    plot_points.append((tuple(zip(*leftovers)), 'gray', LOW_TRANSPARENCY))
    return plot_surfaces, plot_points


def generate_boundary_artist(vertices, color):
    ndim = len(next(iter(vertices[0])))
    vertices = tuple(map(tuple, vertices))
    if ndim == 2:
        artist = LineCollection(vertices)
        artist.set_color(color)
        artist.set_linewidth(2.5)
    elif ndim == 3:
        artist = Poly3DCollection(vertices, linewidth=1)
        # transparency alpha must be set BEFORE face/edge color
        artist.set_alpha(0.5)
        artist.set_facecolor(color)
        artist.set_edgecolor('k')
    else:
        raise ValueError("{:d} dimensions needs special attention for plotting".format(ndim))
    return artist


def prepare_plots(figure, ndim):
    plt.figure(figure.number)
    if ndim == 2:
        dendrogram_params = ((5, 6), (2, 4))
        main_params = ((5, 6), (0, 0))
        d_c, d_r = 2, 3
        m_c, m_r = 4, 5
        projection = "rectilinear"
    elif ndim == 3:
        dendrogram_params = ((2, 4), (1, 3))
        main_params = ((2, 4), (0, 0))
        d_c, d_r = 1, 1
        m_c, m_r = 3, 2
        projection = "3d"
    else:
        raise ValueError("{:d} dimensions needs special attention for plotting.".format(int(ndim)))
    dendrogram_ax = plt.subplot2grid(*dendrogram_params, colspan=d_c, rowspan=d_r)
    main_ax = plt.subplot2grid(*main_params, colspan=m_c, rowspan=m_r, projection=projection)
    if ndim == 3:
        surface_plotter = main_ax.add_collection3d
    elif ndim == 2:
        surface_plotter = main_ax.add_artist
    else:
        raise ValueError("{:d} dimensions needs special attention for plotting.".format(int(ndim)))
    return dendrogram_ax, main_ax, surface_plotter


def depth_first_search(c, f=None, p=None):
    # Return list of clusters under and including c
    # List order is depth-first; order of sibling branches is arbitrary
    #  Children are guaranteed to show up after their parents. That's all.
    # If function f is present, makes a list of the result of that function
    #  instead of a list of clusters
    # Can call a function p on every cluster as well, but ignores output
    dfs_children = deque()
    for sc in c.children:
        dfs_children.extend(depth_first_search(sc))
    if p is not None:
        p(c)
    if f is not None:
        dfs_children.appendleft(f(c))
    else:
        dfs_children.appendleft(c)
    return dfs_children


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
