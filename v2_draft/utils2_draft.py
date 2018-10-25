import numpy as np
from scipy.spatial import Delaunay
from numpy.linalg import det
from scipy.misc import factorial
import AC2_draft as AlphaC
import DK2_draft as DKey

ALPHA_STEP = 0.5  # Geometric, makes alpha progressively smaller
ORPHAN_TOLERANCE = 2
QUIET = False
SPACE = ""
RECURSION_STACK = []
KEY = None

K_PARENT = "parent"
K_CLUSTER_ELEMENTS = "cluster_elements"
K_ALPHA_LEVEL = "alpha_level"
K_NULL_SIMPLICES = "null_simplices"


def initialize(points):
    global KEY
    KEY = DKey.DelaunayKey(Delaunay(points))


# These things need to go to UTILS
def identify_null_simplex(simplex, cluster_list):
    # Figure out if simplex touches a cluster in the list
    # If it does, return the cluster (set(simplices))
    # If it does not, return False
    # This is mostly KEY manipulation so it should be in UTILS
    return False


def tuple_map(coordinate_set):
    return tuple(map(tuple, coordinate_set))


def get_simplex(edge):
    global KEY
    return KEY.get_simplex(edge)


def get_edge(simplex):
    global KEY
    return KEY.get_edge(simplex)


def recurse():
    # does not actually recurse
    RECURSION_STACK.append({
        K_PARENT: KEY,
        K_CLUSTER_ELEMENTS: KEY.simplices(),
        K_ALPHA_LEVEL: None,
        K_NULL_SIMPLICES: []
    })
    while RECURSION_STACK:
        current_job = RECURSION_STACK.pop()
        new_cluster = AlphaC.AlphaCluster(current_job[K_CLUSTER_ELEMENTS],
                                          alpha_level=current_job[K_ALPHA_LEVEL],
                                          null_simplices=current_job[K_NULL_SIMPLICES])
        current_job[K_PARENT].add_branch(new_cluster)


def traverse(remaining_elements):
    # remaining elements is a set of SimplexNodes
    # this function should return a LIST of FROZENSETS of SimplexNodes
    # update this later to be exactly like the old traverse, finding boundaries
    cluster_list = []
    traversal_stack = set()
    while remaining_elements:
        traversal_stack.add(remaining_elements.pop())
        current_cluster = set()
        while traversal_stack:
            t = traversal_stack.pop()
            current_cluster.add(t)
            remaining_elements.discard(t)
            traversal_stack |= set(sum([get_simplex(e) for e in get_edge(t)], [])) & remaining_elements
        cluster_list.append(frozenset(current_cluster))
    return cluster_list


def pad_matrix(m):
    m = np.pad(m, ((1, 0), (1, 0)), 'constant', constant_values=1)
    m[0, 0] = 0.
    return m


def euclidean_distance_matrix(point_array):
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


def cm_volume_helper(cm_det_abs_root, n):
    return cm_det_abs_root / ((2. ** (n / 2.)) * factorial(n))


# noinspection SpellCheckingInspection
def cayley_menger_vr(point_array):
    """
    Return volume and circumradius as given by the Cayley Menger determinant
    :param point_array: np.ndarray of dimension (n+1, n) for dimension n
        These points should define a valid n-dimensional simplex of non-zero volume
    :return: tuple of floats (volume, circumradius)
    """
    n_points = point_array.shape[0] - 1
    d_matrix = euclidean_distance_matrix(point_array)
    cm_det_root = np.sqrt(np.abs(det(pad_matrix(d_matrix))))
    volume = cm_volume_helper(cm_det_root, n_points)
    circumradius = np.sqrt(np.abs(det(d_matrix)) / 2.) / cm_det_root
    return volume, circumradius


# noinspection SpellCheckingInspection
def cayley_menger_volume(point_array):
    """
    Return volume as given by the Cayley Menger determinant.
    Repeats cayley_menger_vr in some aspects, but will save some time when circumradius isn't needed (Edges)
    :param point_array: np.ndarray of dimension (n+1, m) for simplex dimension n embedded in dimension m
        These points should define a valid n-dimensional simplex of non-zero volume
    :return: float volume
    """
    cm_det_root = np.sqrt(np.abs(det(pad_matrix(euclidean_distance_matrix(point_array)))))
    return cm_volume_helper(cm_det_root, point_array.shape[0] - 1)
