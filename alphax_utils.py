import numpy as np
from scipy.spatial import Delaunay
from collections import Iterable
import AlphaCluster as AlphaC
import DelaunayKey as DKey
from matplotlib import colors as mcolors
from numpy.linalg import det
from scipy.misc import factorial
from random import randrange
import sys
import matplotlib.pyplot as plt

DIM = 2
ORPHAN_TOLERANCE = 0
ALPHA_STEP = -1
PERSISTENCE_THRESHOLD = 1
MAIN_CLUSTER_THRESHOLD = -1
QUIET = True
KEY = None
SP = ""

"""
Utility file for alpha-x cluster characterization.

Parameters are defined/redefined here.
"""


def flatten(list_like):
    """
    Flattens an unknown-depth, irregular embedded list
    :param list_like: Iterable whose elements may also be iterables, to unknown depth.
    :return: Flattened list; single iterable whose elements are not iterable.
    """
    for element in list_like:
        if isinstance(element, Iterable) and not isinstance(element, AlphaC.AlphaCluster):
            yield from flatten(element)
        else:
            yield element


def tuple_map(coordinate_set):
    """
    Convenient map; see below
    :param coordinate_set: iterable containing iterable elements
    :return: tuple of tuples
    """
    return tuple(map(tuple, coordinate_set))


def traverse_cluster(remaining_elements):
    """
    Traverse a single cluster through its simplices and their connecting edges.
    The outer while loop runs for each disconnected cluster withing the set, while the inner while loop runs for every
        simplex in each disconnected cluster.
    :param remaining_elements: set of Simplices presumably associated with this cluster.
    :return: list of disconnected (clusters, boundaries) tuples.
        Disconnected clusters are sets of SimplexNodes.
        Boundaries are sets of SimplexEdges.
        The sets are paired together in tuples
        e.g. [(set(c), set(b)), (set(c), set(b)), (set(c), set(b))]
    """
    cluster_list = []  # Holds the resulting tuples, is return value
    traversal_stack = set()  # SimplexNode traversal stack; order doesn't matter, so set is fine
    while remaining_elements:  # While we haven't hit every simplex
        traversal_stack.add(remaining_elements.pop())  # Seed the stack with an element
        current_cluster = set()  # Prepare set for all connected simplices in this cluster
        current_bound = {}  # Counting dictionary for edges (hit 2x, contained. hit 1x, boundary!)
        while traversal_stack:  # While the stack isn't empty
            t = traversal_stack.pop()  # Pop off the stack (order doesn't matter)
            current_cluster.add(t)  # Current element is definitely part of cluster so add it to set
            remaining_elements.discard(t)  # Remove current element from remaining_elements to avoid looping
            # See apy_9 for unintelligible comments
            traversal_stack |= set(sum([track_get_simplex(e, current_bound)
                                        for e in get_edge(t)], [])
                                   ) & remaining_elements  # That's a doozy, but it makes sense
        current_bound = {k for k, v in current_bound.items() if v == 1}  # Isolate boundary edges
        cluster_list.append((current_cluster, current_bound))  # Append current cluster tuple to return value list
    return cluster_list


def track_get_simplex(edge, boundary_dict):
    """
    Wrapper for simplex getter; calls the real getter and handles tracking
    :param edge: Edge instance
    :param boundary_dict: dictionary counting number of times boundary has been visited
    :return: list of Simplex instances
    """
    boundary_dict[edge] = boundary_dict.get(edge, 0) + 1
    return get_simplex(edge)


def get_simplex(edge):
    """
    Getter for simplices joined by this edge, via KEY
    :param edge: Edge instance
    :return: list of Simplex instances
    """
    global KEY
    return KEY.get_simplex(edge)


def get_edge(simplex):
    """
    Getter for edges bounding this simplex, via KEY
    :param simplex: Simplex instance
    :return: list of Edge instances
    """
    global KEY
    return KEY.get_edge(simplex)


def initialize(points):
    """
    Sanitizes parameters and sets up the DelaunayKey given the point cloud
    :param points: 2D numpy array; list of coordinate pairs
    """
    # This is where we should check conditions on edited globals
    global ORPHAN_TOLERANCE
    ORPHAN_TOLERANCE = -ORPHAN_TOLERANCE
    if 1 <= ORPHAN_TOLERANCE < 100:
        ORPHAN_TOLERANCE /= 100.
    elif ORPHAN_TOLERANCE >= 100:
        raise RuntimeError("ORPHAN_TOLERANCE must be: (-inf, 0] (drop one) | (0, 1) (fraction) | [1, 100) (percentage)."
                           " %f is not an acceptable value." % float(ORPHAN_TOLERANCE))
    ORPHAN_TOLERANCE = -ORPHAN_TOLERANCE
    global ALPHA_STEP
    if ALPHA_STEP < 0:  # Need to be careful with this; still in testing phase
        ALPHA_STEP = 10 ** (-np.log10(10) / 10.)
    elif ALPHA_STEP >= 1:
        raise RuntimeError("ALPHA_STEP should be between (0, 1)."
                           " %f is not an acceptable value." % float(ALPHA_STEP))
    global MAIN_CLUSTER_THRESHOLD
    if 1 <= MAIN_CLUSTER_THRESHOLD < 100:
        MAIN_CLUSTER_THRESHOLD /= 100.
    elif MAIN_CLUSTER_THRESHOLD >= 100:
        raise RuntimeError(
            "MAIN_CLUSTER_THRESHOLD must be: (-inf, 0] (drop one) | (0, 1) (fraction) | [1, 100) (percentage)."
            " %f is not an acceptable value." % float(MAIN_CLUSTER_THRESHOLD))
    global KEY
    KEY = DKey.DelaunayKey(Delaunay(points))


def reinitialize(points):
    """
    Unambiguously deletes the last version
    This function is almost certainly unnecessary but gives me peace of mind.
    :param points: 2D numpy array; list of coordinate pairs
    """
    global KEY
    del KEY
    initialize(points)


def recurse():
    """
    Begins tree generation using the initialized key.
    :return: The root of the completed tree9
    """
    if KEY is None:
        raise RuntimeError("DelaunayKey is not set. "
                           "Please initialize it with your data set using the 'INITIALIZE' function.")
    return AlphaC.AlphaCluster(KEY.simplices())


def get_colors():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    return list(zip(*colors.items()))


def dark_color(rgb):
    r, g, b = rgb
    return r + g + b < 400


def rand_color(colors):
    h = (255, 255, 255)
    n = 0
    colors_keys, colors_values = None, None
    while not dark_color(h):
        colors_keys, colors_values = colors
        n = randrange(0, len(colors_keys))
        h = colors_values[n]
        h = (int(i * 255) for i in h) if isinstance(h, tuple) else tuple(
            int(h.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    return colors_keys[n]


def new_patch(bottom_left_corner, width, height):
    """
    Add new patch for dendrogram
    :param bottom_left_corner: tuple of (x, y) for bottom left corner
    :param width: numerical width
    :param height: numerical height
    :return: Patch
    """
    return plt.Rectangle(bottom_left_corner, width, height,
                         facecolor='k', edgecolor=None, linewidth=0.5,
                         )


def dendrogram(alpha_root):
    """
    Generates the MATPLOTLIB patches and coordinates colors for the dendrogram.
    Persistence threshold here or elsewhere?
    :param alpha_root: AlphaCluster instance root of the tree
    :return: dict of colors:points,
        list of colors,
        list of Rectangle lists,
        dendrogram base width,
        tuple of limits (lo_lim, hi_lim)
    """
    # Dendrogram setup
    hash_match_rect = {}
    base_width = alpha_root.member_range[0]
    first_child = max(alpha_root.subclusters, key=lambda x: x.alpha_range[0]) if alpha_root.subclusters else alpha_root
    lim_alpha_lo = lim_alpha_hi = first_child.alpha_range[0] / ALPHA_STEP
    stack = [alpha_root]
    centroid_stack = [base_width / 2.]
    hash_match_color = {}
    # Color setup
    points = {}
    colors = get_colors()
    count = 0
    while stack:
        count += 1
        msg = ".. %3d ..\r" % count
        sys.stdout.write(msg)
        sys.stdout.flush()
        a = stack.pop()
        color = rand_color(colors)
        centroid = centroid_stack.pop()
        persistent = len(a.alpha_range) - 1 >= PERSISTENCE_THRESHOLD
        if persistent:
            point_set = set()
            for t in a.cluster_elements:
                point_set |= t
            for p in point_set:
                points[p] = color
            hash_match_color[hash(a)] = color
            hash_match_rect[hash(a)] = []
        fork_alphas = [sc.alpha_range[0] for sc in a.subclusters]
        continuing = False
        start_alpha, end_alpha = None, None
        for i in range(len(a.alpha_range) - 1):
            end_alpha = a.alpha_range[i + 1]
            if not continuing:
                start_alpha = a.alpha_range[i]
            width = a.member_range[i + 1]
            if (i < len(a.member_range) - 2) and a.member_range[i + 2] == width:
                continuing = True
                continue
            if persistent:
                hash_match_rect[hash(a)].append(new_patch(
                    (centroid - width / 2., end_alpha),
                    width, start_alpha - end_alpha,
                ))
            continuing = False
            current_fork_i = [j for j, x in enumerate(fork_alphas) if x == end_alpha]
            true_left_edge = centroid - width / 2.
            if current_fork_i:
                current_forks = [a.subclusters[j] for j in current_fork_i]
                # noinspection PyUnusedLocal
                current_fork_i = [0 for j in current_forks]
                current_forks = [a] + current_forks
                current_fork_i = [i + 2] + current_fork_i
                total_new_width = float(sum([j.member_range[idx] for j, idx in zip(current_forks, current_fork_i)]))
                left_edge = 0
                for j, idx in zip(current_forks, current_fork_i):
                    fractional_width = j.member_range[idx] / total_new_width
                    new_centroid = (left_edge + fractional_width / 2.) * width + true_left_edge
                    left_edge += fractional_width
                    if j == a:
                        centroid = new_centroid
                    else:
                        stack.append(j)
                        centroid_stack.append(new_centroid)
        if end_alpha < lim_alpha_lo:
            # Using lim_alpha_hi/lo as plot boundaries
            lim_alpha_lo = end_alpha
    colors = {}
    for p, c in points.items():
        if c in colors:
            colors[c].append(p)
        else:
            colors[c] = [p]
    color_list, recs = [], []
    for h in hash_match_color.keys():
        color_list.append(hash_match_color[h])
        recs.append(hash_match_rect[h])
    return colors, color_list, recs, base_width, (lim_alpha_lo, lim_alpha_hi)


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


def old_vr(point_array):
    pa, pb, pc = tuple(point_array)
    a = np.sqrt(np.sum((pa - pb) ** 2.))
    b = np.sqrt(np.sum((pb - pc) ** 2.))
    c = np.sqrt(np.sum((pc - pa) ** 2.))
    s = (a + b + c) / 2.
    volume = np.sqrt(s * (s - a) * (s - b) * (s - c))
    if volume == 0:
        # Check for co-linear; send warning up to constructor call line
        print("Shape appears to contain co-linear corners")
        circumradius = np.inf
    else:
        circumradius = a * b * c / (4. * volume)
    return volume, circumradius


def boundary_traverse(cluster_bound_pair):
    """
    Traverses all the boundaries present in this cluster
    :param cluster_bound_pair: tuple (set(cluster simplices), set(boundary edges))
    :return: list of boundary circuit tuples of set, frozenset [(set(b0), frozenset()), (set(b1), frozenset(g1)), ..]
        The first frozenset is EMPTY, since the outer boundary does not constitute a gap
    """
    # TODO we should standardize all the traversals...
    cluster_simplices, remaining_elements = cluster_bound_pair  # Split apart pair of sets
    boundary_list = []  # This will contain all sets of edges, one for each boundary circuit
    min_max = []
    traversal_stack = set()  # Traversal stack; order doesn't matter
    while remaining_elements:
        current_bound = set()
        t = set(remaining_elements).pop()
        traversal_stack.add(t)
        while traversal_stack:
            t = traversal_stack.pop()
            current_bound.add(t)
            faces = {(t - {c}): set() for c in iter(t)}
            for f in faces:
                for b in remaining_elements:
                    if b > f and b != t:
                        faces[f].add(b)
                if len(faces[f]) > 1:  # There are several connecting faces; need to pick the outermost
                    best_path = choose_path(t, faces[f], f, cluster_simplices)
                    faces[f] = {best_path}  # Assign correct boundary to this face
                elif len(faces[f]) < 1:  # We've already traversed this direction
                    faces[f] = set()
                traversal_stack |= faces[f] - current_bound
        remaining_elements -= current_bound
        minx = min(current_bound, key=lambda x: min(x, key=lambda y: y[0]))
        maxx = max(current_bound, key=lambda x: max(x, key=lambda y: y[0]))
        boundary_list.append(current_bound)
        min_max.append((minx, maxx))
    # outer_bound_index = min_max.index((min(min_max, key=lambda x: x[0])[0], max(min_max, key=lambda x: x[1])[1]))
    outer_bound_index = boundary_list.index(max(boundary_list, key=lambda x: len(x)))
    outer_bound = boundary_list.pop(outer_bound_index)
    boundary_list.insert(0, outer_bound)  # Outer boundary should be FIRST element of list
    # Now to traverse the gaps!
    gap_list = [frozenset()]
    if len(boundary_list) > 1:
        traversal_stack = set()
        for current_bound in boundary_list[1:]:
            test_edge = set(current_bound).pop()
            interior_simplex = set(get_simplex(test_edge)) - cluster_simplices
            assert len(interior_simplex) == 1
            interior_simplex = interior_simplex.pop()
            traversal_stack.add(interior_simplex)
            gap_simplices = set()
            while traversal_stack:
                t = traversal_stack.pop()
                gap_simplices.add(t)
                traversal_stack |= set(
                    sum([get_simplex(e) for e in get_edge(t) if e not in current_bound], [])
                ) - gap_simplices
            # append the frozenSET of simplices within the gap
            gap_list.append(frozenset(gap_simplices))
    return_list = list(zip(boundary_list, gap_list))
    return return_list  # Returns list of (set, frozenset) tuples


def describe_bound(edge, cluster_simplices):
    bounded_simplex = set(get_simplex(edge)) & cluster_simplices
    assert len(bounded_simplex) == 1
    bounded_simplex = bounded_simplex.pop()
    x = edge.centroid
    p = x - bounded_simplex.centroid
    e = np.array(set(edge).pop()) - x
    pe = (p.dot(e) / e.dot(e)) * e
    n = p - pe
    n = n / (np.sqrt(n.dot(n)))
    return n, x


def centroid(coordinates):
    return np.mean(np.array(list(map(np.array, coordinates))), axis=0)


def choose_path(incident_edge, other_edges_set, shared_face, cluster_simplices):
    """
    Chooses correct traversal path in cases of degeneracy
    :param incident_edge: SimplexEdge from which we are traversing
    :param other_edges_set: set of SimplexEdges connected to the incident_edge
    :param shared_face: set of coord tuples: face (n-2) at which incident_edge joins all others
    :param cluster_simplices: set of SimplexNodes in current cluster
    :return: ???
    """
    s = centroid(shared_face)  # Shared face centroid
    n0, x0 = describe_bound(incident_edge, cluster_simplices)  # Get info for incident bound
    e1 = x0 - s  # Prep the e1 axis analog of new basis
    e1 = e1 / (np.sqrt(e1.dot(e1)))  # Normalize
    e2 = n0  # Prep the e2 axis analog of new basis
    other_edges_list = list(other_edges_set)
    corresponding_angles = []
    for i, oe in enumerate(other_edges_list):
        n, x = describe_bound(oe, cluster_simplices)  # Get info for next bound
        m = x - s  # Get m vector for this bound
        m1, m2 = m.dot(e1), m.dot(e2)  # Project; e vectors are normalized so this should be ok
        angle = np.arctan2(m2, m1)  # first argument is rise, second is run
        corresponding_angles.append(angle if angle > 0 else angle + np.pi*2)  # Save angle between 0-2pi
    # Return boundary corresponding to the minimum angle. This works for interior AND exterior boundaries
    return other_edges_list[corresponding_angles.index(min(corresponding_angles))]


def gap_post_process(boundary_range):
    """
    Processes alpha-range list of gaps and filters for coherence
    :param boundary_range: list of lists of (set(), frozenset()) tuples
        The outer list ranges through alpha, inner list ranges through distinct boundaries
        In the tuples, the sets give boundaries and the frozensets give gap simplices
    :return: A list in the same format as the input but only with persistent gaps
    """
    gap_key = {}  # frozenset(first_membership) : [(set(bound), frozenset(current_membership), idx), ..]
    for i, alpha_list in enumerate(boundary_range):
        # alpha_list is an alpha-level list of (set, frozenset) tuples
        if alpha_list is None:
            for k in gap_key:
                last_bound, last_mem, last_idx = gap_key[k][-1]
                if last_idx + 1 == i:
                    gap_key[k].append((last_bound, last_mem, i))
        else:
            for j, gap_pair in enumerate(alpha_list):
                if j == 0:
                    # This is the outer boundary
                    continue
                found = False
                bound_set, gap_set = gap_pair
                for k in gap_key:
                    if found:
                        continue
                    elif gap_set >= k:
                        gap_key[k].append((bound_set, gap_set, i))
                        found = True
                if not found:
                    gap_key[gap_set] = [(bound_set, gap_set, i)]
    return_range = [[] for x in boundary_range]
    for k in gap_key:
        if len(gap_key[k]) >= 25:
            for bound, membership, index in gap_key[k]:
                return_range[index].append((bound, membership))
    return return_range

