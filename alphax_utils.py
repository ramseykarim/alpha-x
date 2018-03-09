import numpy as np
from scipy.spatial import Delaunay
from collections import Iterable
import AlphaCluster
import DelaunayKey as DKey
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
        if isinstance(element, Iterable) and not isinstance(element, AlphaCluster):
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
    :return: list of disconnected clusters. Disconnected clusters are sets of Simplices.
    """
    cluster_list = []
    traversal_stack = set()
    while remaining_elements:
        traversal_stack.add(remaining_elements.pop())
        current_cluster = set()
        while traversal_stack:
            t = traversal_stack.pop()
            current_cluster.add(t)
            remaining_elements.discard(t)
            # See apy_9 for unintelligible comments
            traversal_stack |= set(sum([get_simplex(e) for e in get_edge(t)], []))
        # Should traverse boundary edges here
        cluster_list.append(current_cluster)


def get_simplex(edge):
    global KEY
    return KEY.get_simplex(edge)


def get_edge(simplex):
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
    global CHOP_ALPHA
    if CHOP_ALPHA < 0:  # Need to be careful with this; still in testing phase
        CHOP_ALPHA = 10 ** (-np.log10(10) / 10.)
    # if 1 <= CHOP_FRACTION < 100:
    # 	CHOP_FRACTION /= 100.
    # elif CHOP_FRACTION >= 100:
    # 	raise RuntimeError("CHOP_FRACTION must be: (-inf, 0] (drop one) | (0, 1) (fraction) | [1, 100) (percentage)."
    # 		" %f is not an acceptable value." % float(CHOP_FRACTION))
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
