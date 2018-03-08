from scipy.spatial import Delaunay
from collections import Iterable
import AlphaCluster
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
    :param list_like: Iterator whose elements may also be iterators, to unknown depth.
    :return: Flattened list; single iterable whose elements are not iterable.
    """
    for element in list_like:
        if isinstance(element, Iterable) and not isinstance(element, AlphaCluster):
            yield from flatten(element)
        else:
            yield element


def tuple_map(coordinate_set):
    return tuple(map(tuple, coordinate_set))
