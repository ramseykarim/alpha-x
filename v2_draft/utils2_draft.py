import numpy as np
from scipy.spatial import Delaunay
from numpy.linalg import det
from scipy.misc import factorial
from random import seed, randrange
import AC2_draft as AlphaC
import DK2_draft as DKey
import matplotlib.pyplot as plt
from matplotlib import colors as m_colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import deque

ALPHA_STEP = 0.5  # Geometric, makes alpha progressively smaller
ORPHAN_TOLERANCE = 2
QUIET = False
SPACE = ""
RECURSION_STACK = []
KEY = None
COLORS = (None, None)
DIM = None

K_PARENT = "parent"
K_CLUSTER_ELEMENTS = "cluster_elements"
K_ALPHA_LEVEL = "alpha_level"
K_NULL_SIMPLICES = "null_simplices"

# OK color seeds: 60341 93547 1337 334442 332542
SEED = 635541
seed(SEED)

def initialize(points):
    global KEY
    KEY = DKey.DelaunayKey(Delaunay(points))
    DIM = points.shape[1]
    global COLORS
    COLORS = get_colors()


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
            K_PARENT: parent,
            K_CLUSTER_ELEMENTS: cluster_elements,
            K_ALPHA_LEVEL: alpha_level,
            K_NULL_SIMPLICES: null_simplices
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
        new_cluster = AlphaC.AlphaCluster(current_job[K_CLUSTER_ELEMENTS], current_job[K_PARENT],
                                          alpha_level=current_job[K_ALPHA_LEVEL],
                                          null_simplices=current_job[K_NULL_SIMPLICES])
        current_job[K_PARENT].add_branch(new_cluster)

"""
Traversals
"""
def traverse(remaining_elements):
    # remaining elements is a set of SimplexNodes
    # this function should return a LIST of SETS of SimplexNodes
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
        cluster_list.append((current_cluster, current_bound))
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
                for b in remaining_edges:
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
        boundary_list.append(current_bound)
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
            gap_list.append(gap_simplices)
    return_list = list(zip(boundary_list, gap_list))
    return return_list  # returns list of (set, set) tuples

"""
Traversal helpers, math stuff
"""
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

"""
Simplex volume, circumradius math stuff
"""
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

"""
Generating random colors
"""
def get_colors():
    colors = dict(m_colors.BASE_COLORS, **m_colors.CSS4_COLORS)
    return list(zip(*colors.items()))


def dark_color(rgb):
    # rgb is a tuple (r, g, b) integers
    return sum(rgb) < 400


def rand_color():
    h = (255, 255, 255)
    n = 0
    colors_keys, colors_values = COLORS
    while not dark_color(h) and h not in KEY.treeIndex.colors:
        n = randrange(0, len(colors_keys))
        h = colors_values[n]
        h = (int(i*255) for i in h) if isinstance(h, tuple) else tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    KEY.treeIndex.colors.add(colors_keys[n])
    return colors_keys[n]


"""
Post-processing clusters
"""
def npoints_from_nsimplices(simplex_set):
    # simplex_set just needs to be iterable
    return set.union(*tuple(map(set, simplex_set)))


def clusters_at_alpha(alpha):
    all_alphas = sorted(list(KEY.treeIndex.alpha_range.keys()), reverse=True)
    if alpha > all_alphas[0]:
        # all elements
        alpha = all_alphas[0]
    elif all_alphas[-1]*ALPHA_STEP > alpha:
        # no elements
        return None
    all_alphas = [x for x in all_alphas if x >= alpha]
    # FORCING ALPHA TO BE THE CLOSEST STEP TAKEN
    alpha = min(all_alphas, key=lambda x: x - alpha)
    valid_shapes = {k: k.cluster_at_alpha(alpha) for k in KEY.treeIndex.alpha_range[alpha]}
    return valid_shapes

"""
Dendrogram, plotting
"""
def new_patch(bottom_left_corner, width, height, color):
    return plt.Rectangle(bottom_left_corner, width, height,
                         facecolor=color, edgecolor=None, linewidth=0.5
                         )


def dendrogram():
    """
    Generates the MATPLOTLIB patches for the dendrogram.
    Should we enforce persistence here?
    """
    base_width = len(KEY.alpha_root.cluster_elements)
    first_child = max(KEY.alpha_root.subclusters, key=lambda x: x.alpha_range[0]) if KEY.alpha_root.subclusters else KEY.alpha_root
    lim_alpha_lo = lim_alpha_hi = first_child.alpha_range[0] / ALPHA_STEP
    stack = deque(((KEY.alpha_root, base_width / 2),))
    count = 0
    patch_stack = deque()
    while stack:
        count += 1
        msg = ".. %3d ../r" % count
        # could sys.stdout.write(that)
        a, centroid = stack.pop()
        color = a.color
        # Centroid at which *this* cluster should start
        # Alpha levels at which we must branch
        fork_alphas = [sc.alpha_range[0] for sc in a.subclusters]
        stretching_patch_upward = False
        start_alpha, end_alpha = None, None
        for i, alpha in enumerate(a.alpha_range):
            end_alpha = alpha*ALPHA_STEP
            if not stretching_patch_upward:
                start_alpha = alpha
            width = a.nsimplex_range[i]
            if (i+1 != len(a.alpha_range)) and a.nsimplex_range[i+1] == width:
                stretching_patch_upward = True
                continue
            true_left_edge = centroid - width / 2
            patch_stack.append(new_patch((true_left_edge, end_alpha), width, start_alpha - end_alpha, color))
            stretching_patch_upward = False
            current_fork_i = [j for j, x in enumerate(fork_alphas) if x == end_alpha]
            # Calculate the new width and centroid of the stacks, since we're splitting off a child cluster
            if current_fork_i:
                current_forks = deque((a.subclusters[j], 0) for j in current_fork_i)
                current_forks.appendleft((a, i+1))
                total_new_width = float(sum(j.nsimplex_range[idx] for j, idx in current_forks))
                left_edge = 0
                for j, idx in current_forks:
                    fractional_width = j.nsimplex_range[idx] / total_new_width
                    new_centroid = (left_edge + fractional_width / 2) * width + true_left_edge
                    left_edge += fractional_width
                    if j == a:
                        centroid = new_centroid
                    else:
                        stack.append((j, new_centroid))
        if end_alpha < lim_alpha_lo:
            # using lim_alpha_hi/lo as plot boundaries
            lim_alpha_lo = end_alpha
    return patch_stack, base_width, (lim_alpha_lo, lim_alpha_hi)


def naive_point_grouping():
    print("grouping naively..")
    # Sort all clusters by their *end*points, smallest first
    stack = deque(sorted(set.union(*KEY.treeIndex.alpha_range.values()), key=lambda x: x.nsimplex_range[0]))
    print(stack)
    point_clusters = deque()
    used_points = set()
    while stack:
        a = stack.popleft()
        points = npoints_from_nsimplices(a.cluster_elements)
        points -= used_points
        used_points |= points
        print("new group of %d points" % len(points))
        points = tuple(map(tuple, zip(*points)))
        point_clusters.append((points, a.color))
    return point_clusters


def alpha_surfaces(alpha):
    plot_surfaces = deque()
    plot_points = deque()
    clusters = clusters_at_alpha(alpha)
    for a in clusters:
        boundary, elements = clusters[a][0]
        color = a.color
        for b in boundary:
            print()
            # need to make shapes depending on dimension
        points = tuple(map(tuple, zip(*npoints_from_nsimplices(elements))))
        plot_points.append((points, color))


def generate_boundary_artist(boundary):
    # FIXME
    return None
