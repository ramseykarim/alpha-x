import numpy as np
from scipy.spatial import Delaunay
from collections import Iterable
import AlphaCluster as AlphaC
import DelaunayKey as DKey
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches
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
            traversal_stack |= set(sum([get_simplex(e) for e in get_edge(t)], [])) & remaining_elements
        # Should traverse boundary edges here
        cluster_list.append(current_cluster)
    return cluster_list


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
    return list(colors.keys())


def rand_color(colors):
    return colors[randrange(0, len(colors))]


def new_patch(bottom_left_corner, width, height):
    """
    Add new patch for dendrogram
    :param bottom_left_corner: tuple of (x, y) for bottom left corner
    :param width: numerical width
    :param height: numerical height
    :return: Patch
    """
    return plt.Rectangle(bottom_left_corner, width, height,
                         facecolor='k', edgecolor='k', linewidth=0.5,
                         )
    # return mpatches.FancyBboxPatch(bottom_left_corner, width, height,
    #                                boxstyle=mpatches.BoxStyle("Round", pad=0.02*height)
    #                                )


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
    lim_alpha_lo = lim_alpha_hi = alpha_root.alpha_range[0]
    stack = [alpha_root]
    centroid_stack = [base_width/2.]
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
                point_set |= t.corners
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
                    (centroid - width/2., end_alpha),
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
                    new_centroid = (left_edge + fractional_width/2.)*width + true_left_edge
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
