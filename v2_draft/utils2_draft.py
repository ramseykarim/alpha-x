import numpy as np
from scipy.spatial import Delaunay
from numpy.linalg import det
from scipy.misc import factorial
from random import randrange
import AC2_draft as AlphaC
import DK2_draft as DKey
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


def track_get_simplex(edge, boundary_dict):
    boundary_dict[edge] = boundary_dict.get(edge, 0) + 1
    return get_simplex(edge)


def recursion_push(parent, cluster_elements, alpha_level, null_simplices):
    RECURSION_STACK.append({
            utils.K_PARENT: parent,
            utils.K_CLUSTER_ELEMENTS: cluster_elements,
            utils.K_ALPHA_LEVEL: alpha_level,
            utils.K_NULL_SIMPLICES: null_simplices
    })
 

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
    # will also return SETS of SimplexEdges
    cluster_list = []
    traversal_stack = set()
    while remaining_elements:
        traversal_stack.add(remaining_elements.pop())
        current_cluster = set()
        current_bound = {}
        while traversal_stack:
            t = traversal_stack.pop()
            current_cluster.add(t)
            remaining_elements.discard(t)
            traversal_stack |= set(sum([track_get_simplex(e, current_bound)
                                        for e in get_edge(t)], [])
                                   ) & remaining_elements
        current_bound = {k for k, v in current_bound.items() if v == 1}
        cluster_list.append((frozenset(current_cluster), current_bound))
    return cluster_list


def boundary_traverse(cluster_bound_pair, outer_only=False):
    # cluster_bound_pair is a tuple of sets: (set(cluster simplices), set(boundary edges))
    cluster_simplices, remaining_edges = cluster_bound_pair
    boundary_list = [] # will contain all sets of edges, one set for each boundary circuit
    traversal_stack = set()
    # As the while loop starts:
    #  remaining_edges has all edges that we haven't yet crossed and need to
    #  traversal_stack is empty but will be populated for the inner while loop
    while remaining_edges:
        current_bound = set()
        t = remaining_edges.pop()
        traversal_stack.add(t)
        # traversal_stack has a single element
        # current_bound is empty
        while traversal_stack:
            # pop from stack; this element is for sure in current_bound
            t = traversal_stack.pop()
            current_bound.add(t)
            # get all faces of this boundary (faces are like edges of edges, ie n-2 for nD)
            faces = {(t - {c}): set() for c in iter(t)}
            for f in faces:
                for b in remaining_elements:
                    if b>f and b!=t:
                        # b (an edge) shares this face, and is not the edge we are testing
                        faces[f].add(b)
                if len(faces[f])>1: # several connecting faces, need to pick
                    faces[f] = {choose_path(t, faces[f], f, cluster_simplices)}
                # update traversal_stack
                # do not update remaining_edges! need to have a complete traversal
                # (i.e. we'll pick bad paths if we remove options)
                traversal_stack |= faces[f] - current_bound
        remaining_edges -= current_bound
    outer_bound_index = boundary_list.index(max(boundary_list, key=lambda x: len(x)))
    outer_bound = boundary_list.pop(outer_bound_index)
    if outer_only:
        return outer_bound
    boundary_list.insert(0, outer_bound)
    gap_list = [frozenset()]
    if len(boundary_list) > 1:
        traversal_stack = set()
        for current_bound in boundary_list[1:]:
            test_edge = next(iter(current_bound))
            interior_simplex = (set(get_simplex(test_edge)) - cluster_simplices).pop()
            traversal_stack.add(interior_simplex)
            gap_simplices = set()
            while traversal_stack:
                t = traversal_stack.pop()
                gap_simplices.add(t)
                traversal_stack |= set(
                    sum([get_simplex(e) for e in get_edge(t) if e not in current_bound], [])
                    ) - gap_simplices
            gap_list.append(frozenset(gap_simplices))
    return_list = list(zip(boundary_list, gap_list))
    return return_list # returns list of (set, frozenset) tuples


def choose_path(incident_edge, other_edges_set, shared_face, cluster_simplices):
    s = centroid(shared_face)
    n0, x0 = describe_bound(incident_edge, cluster_simplices)
    e1 = x0 - s
    e1 = e1/np.sqrt(e1.dot(e1))
    e2 = n0
    other_edges_list = list(other_edges_set)
    corresponding_angles = []
    for i, oe in enumerate(other_edges_list):
        n, x = describe_bound(oe, cluster_simplices)
        m = x - s
        m1, m2 = m.dot(e1), m.dot(e2)
        angle = np.arctan2(m2, m1)
        corresponding_angles.append(angle if angle>0 else angle + np.pi*2)
    return other_edges_list[corresponding_angles.index(min(corresponding_angles))]


def centroid(coordinates):
    return np.mean(np.array(list(map(np.array, coordinates))), axis=0)


def describe_bound(edge, cluster_simplices):
    bounded_simplex = (set(get_simplex(edge)) & cluster_simplices).pop()
    x = edge.centroid
    p = x - bounded_simplex.centroid
    e = np.array(next(iter(edge))) - x
    pe = (p.dot(e) / e.dot(e)) * e
    n = p - pe
    n = n/np.sqrt(n.dot(n))
    return n, x


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


def get_colors():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    return list(zip(*colors.items()))


def dark_color(rgb):
    # rgb is a tuple (r, g, b) integers
    return sum(rgb) < 400


def rand_color(colors):
    h = (255, 255, 255)
    n = 0
    colors_keys, colors_values = colors
    while not dark_color(h):
        n = randrange(0, len(colors_keys))
        h = colors_values[n]
        h = (int(i*255) for i in h) if isinstance(h, tuple) else tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    return colors_keys[n]


def new_patch(bottom_left_corner, width, height, color):
    return plt.Rectangle(bottom_left_corner, width, height,
                         facecolor=color, edgecolor=None, linewidth=0.5
                         )


def npoints_from_nsimplices(simplex_set):
    # simplex_set just needs to be iterable
    return set.union(*tuple(simplex_set))

