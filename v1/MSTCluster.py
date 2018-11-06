import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree as mst
import alphax_utils as utils
import sys


"""
MSTCluster is NOT meant to be a clean code. It was just thrown together to compare to AlphaX.

There are a couple different attempts at MST clustering code in here, but the good stuff is at the bottom.
"""


KEY = {"GRAPH": None, "EDGES": None}
DIM = 2
ORPHAN_TOLERANCE = 0
ALPHA_STEP = -1
PERSISTENCE_THRESHOLD = 1
MAIN_CLUSTER_THRESHOLD = -1
QUIET = True
SP = ""


def prepare_mst(tri):
    csr_matrix = np.zeros((tri.points.shape[0], tri.points.shape[0]), dtype=np.float64)
    n_indices, n_indptr = tri.vertex_neighbor_vertices
    for i in range(tri.points.shape[0]):
        point_i = tri.points[i, :]
        neighbors = n_indptr[n_indices[i]:n_indices[i + 1]]
        for j in neighbors:
            if i > j:
                sep = point_i - tri.points[j, :]
                sep = np.sqrt(sep.dot(sep))
                csr_matrix[i, j] = sep
                csr_matrix[j, i] = sep
    # noinspection PyTypeChecker
    min_sp_tree = mst(csr_matrix, overwrite=True).toarray()
    graph = {}
    edges = {}
    for i in range(tri.points.shape[0]):
        p0 = tuple(tri.points[i])
        for j in range(tri.points.shape[0]):
            if min_sp_tree[i, j] != 0:
                p1 = tuple(tri.points[j])
                if p0 in graph:
                    graph[p0].append(p1)
                else:
                    graph[p0] = [p1]
                if p1 in graph:
                    graph[p1].append(p0)
                else:
                    graph[p1] = [p0]
                fs = frozenset({p0, p1})
                if fs not in edges:
                    edges[fs] = min_sp_tree[i, j]
    return graph, edges


def get_edge_weight(p0, p1):
    p0 = tuple(p0)
    p1 = tuple(p1)
    return KEY["EDGES"][frozenset({p0, p1})]


def get_neighbors(p):
    p = tuple(p)
    return KEY["GRAPH"][p]


def get_sorted_cluster_edges(cluster_graph):
    present_edges = set()
    for k, vs in cluster_graph.items():
        for v in vs:
            present_edges.add(frozenset({k, v}))
    return sorted([x for x in KEY["EDGES"].items() if x[0] in present_edges], key=lambda x: x[1])


def traverse_cluster(graph_remnant):
    # remaining_edges are in the same form as the return value of get_sorted_cluster_edges
    cluster_list = []
    traversal_stack = set()
    while graph_remnant:
        k, v = graph_remnant.popitem()
        # print("new cluster, popped ", k)
        for i in v:
            traversal_stack.add(i)
            # print("  added ", i, " to stack")
        current_cluster = {}
        while traversal_stack:
            t = traversal_stack.pop()
            # print("popped ", t, " from stack")
            if t in graph_remnant:
                v = graph_remnant.pop(t)
                # print("found ^ in graph_remnant, popping it for i")
                for i in v:
                    # print("trying ", i, " ..", end=" ")
                    if i in graph_remnant:
                        if t not in current_cluster:
                            current_cluster[t] = []
                        if i not in current_cluster:
                            current_cluster[i] = []
                        # print("found ", i, " in graph_remnant")
                        traversal_stack.add(i)
                        current_cluster[t].append(i)
                        current_cluster[i].append(t)
        # print("end cluster")
        cluster_list.append(current_cluster)
    # print("end algorithm \ \ \ ")
    # for cluster in cluster_list:
    #     for k, v in cluster.items():
    #         print(k, "  :  ", v)
    #     print()
    # print()
    return cluster_list


class MSTCluster:
    def __init__(self, cluster_graph, alpha_level=None):
        self.cluster_elements = set(cluster_graph.keys())
        self.cluster_graph = cluster_graph
        self.cluster_edges = get_sorted_cluster_edges(cluster_graph)
        if alpha_level is None:
            self.alpha_range = [self.cluster_edges[-1][1] / ALPHA_STEP]
        else:
            self.alpha_range = [alpha_level]
        self.member_range = [len(self.cluster_edges)]
        self.subclusters = []
        global SP
        if not QUIET:
            print(SP + "<branch init_size=%d>" % len(self.cluster_edges))
            SP += "|  "
        self.exhaust_cluster()
        if not QUIET:
            SP = SP[:-3]
            print(SP + "</branch persist=%d>" % len(self.alpha_range))

    def exhaust_cluster(self):
        """
        Begin with all points in this cluster
        Iterate next alpha_step until cluster breaks
        Generate children
        :return: this instance
        """
        remaining_edges = self.cluster_edges.copy()
        totally_finished = False
        next_alpha = None
        while not totally_finished:  # I feel like we could improve this implementation...
            coherent = True
            while remaining_edges and coherent:
                next_alpha = self.alpha_range[-1] * ALPHA_STEP
                drop_score = 0
                self.member_range.append(len(remaining_edges))
                while remaining_edges and remaining_edges[-1][1] > next_alpha:
                    dropped_edge = remaining_edges.pop()
                    p0, p1 = dropped_edge[0]
                    if p0 in self.cluster_graph and p1 in self.cluster_graph[p0]:
                        self.cluster_graph[p0].remove(p1)
                    if p1 in self.cluster_graph and p0 in self.cluster_graph[p1]:
                        self.cluster_graph[p1].remove(p0)
                    drop_score += 1
                self.alpha_range.append(next_alpha)
                if drop_score == 0:
                    continue  # No need to traverse the same graph as last time
                cluster_list = traverse_cluster(self.cluster_graph)
                orphan_tolerance = ORPHAN_TOLERANCE
                cluster_list = [x for x in cluster_list if len(x) > orphan_tolerance]
                # cluster_list either has several elements, 1 element, or 0 elements
                if len(cluster_list) > 1:
                    coherent = False  # Several! Send below to create children
                else:
                    if cluster_list:
                        remaining_edges = get_sorted_cluster_edges(cluster_list[0])
                        self.cluster_graph = cluster_list[0]
                    else:
                        remaining_edges = cluster_list
                        self.cluster_graph = cluster_list
            # Just finished looping through remaining_simplices! Something stopped it.
            # Two options: we lost coherency (split cluster) or we ran out of simplices and leafed.
            if coherent:  # Definitely has 1 cluster with 0 simplices left; never split
                # This is a leaf: no more simplices
                totally_finished = True
            else:  # Definitely has more than 1 cluster!
                if MAIN_CLUSTER_THRESHOLD > 0:  # Possibility to make children and still persist
                    active_simplices = sum([len(sc) for sc in cluster_list])
                    largest_subcluster = max(cluster_list, key=lambda x: len(x))
                    if len(largest_subcluster) > active_simplices * MAIN_CLUSTER_THRESHOLD:
                        remaining_edges = get_sorted_cluster_edges(largest_subcluster)  # Still persists with children!
                        cluster_list.remove(largest_subcluster)  # Children made from remainder
                        self.cluster_graph = largest_subcluster
                    else:
                        totally_finished = True
                else:  # Dude there HAS to be a better way to control this omg
                    totally_finished = True
                # Get previous children
                old_children = self.subclusters
                # Make subclusters
                self.subclusters = [MSTCluster(sc, alpha_level=next_alpha) for sc in cluster_list]
                # # Filter by persistence threshold
                # self.subclusters = [(ac if len(ac.alpha_range) >= utils.PERSISTENCE_THRESHOLD + 1 else ac.subclusters)
                #                     for ac in self.subclusters]
                # Flatten list of subclusters
                self.subclusters = list(utils.flatten([ac for ac in self.subclusters if ac is not None]))
                # Reintroduce previous children
                self.subclusters = old_children + self.subclusters
        self.member_range.append(0)
        return self


def initialize(points):
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
    tri = Delaunay(points)
    graph, edges = prepare_mst(tri)
    KEY["GRAPH"] = graph
    KEY["EDGES"] = edges


def recurse():
    return MSTCluster(KEY["GRAPH"])


def dendrogram(alpha_root):
    """
    Specifically for MSTCluster
    :param alpha_root: MSTCluster instance root of the tree
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
    colors = utils.get_colors()
    count = 0
    while stack:
        count += 1
        # msg = ".. %3d ..\r" % count
        # sys.stdout.write(msg)
        # sys.stdout.flush()
        a = stack.pop()
        color = utils.rand_color(colors)
        centroid = centroid_stack.pop()
        persistent = len(a.alpha_range) - 1 >= PERSISTENCE_THRESHOLD
        if persistent:
            point_set = set()
            point_set |= a.cluster_elements
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
                hash_match_rect[hash(a)].append(utils.new_patch(
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


def prepare_mst_simple(tri):
    csr_matrix = np.zeros((tri.points.shape[0], tri.points.shape[0]), dtype=np.float64)
    n_indices, n_indptr = tri.vertex_neighbor_vertices
    for i in range(tri.points.shape[0]):
        point_i = tri.points[i, :]
        neighbors = n_indptr[n_indices[i]:n_indices[i + 1]]
        for j in neighbors:
            if i > j:
                sep = point_i - tri.points[j, :]
                sep = np.sqrt(sep.dot(sep))
                csr_matrix[i, j] = sep
                csr_matrix[j, i] = sep
    # noinspection PyTypeChecker
    min_sp_tree = mst(csr_matrix, overwrite=True).toarray()
    return min_sp_tree


def get_mst_edges(min_sp_tree):
    edge_lengths = []
    for i in range(min_sp_tree.shape[0]):
        for j in range(min_sp_tree.shape[0]):
            if min_sp_tree[i, j] != 0:
                edge_lengths.append(min_sp_tree[i, j])
    return np.array(edge_lengths)


def gen_cdf(edge_lengths, n=100):
    lo, hi = np.min(edge_lengths), np.max(edge_lengths)
    dx = (hi - lo) / float(n - 1)
    cdf = np.zeros(n)
    for e in edge_lengths:
        i = int(np.round((e - lo)/dx))
        cdf[i] += 1
    last = 0
    for i in range(n):
        cdf[i] = cdf[i] + last
        last = cdf[i]
    return cdf, lo, dx


def get_cdf_cutoff(min_sp_tree):
    n = 100
    cdf, lo, dx = gen_cdf(get_mst_edges(min_sp_tree), n=n)
    n_range = np.arange(n)*dx + lo
    #  #  fit from 0 to 12 and 60 to 100
    fit1 = np.polyfit(n_range[:12], cdf[:12], deg=1)
    fit2 = np.polyfit(n_range[60:], cdf[60:], deg=1)
    common_x = (fit1[1] - fit2[1]) / (fit2[0] - fit1[0])
    return common_x


def filter_mst(min_sp_tree):
    filtered_tree = np.zeros(min_sp_tree.shape)
    cutoff = get_cdf_cutoff(min_sp_tree)
    for i in range(min_sp_tree.shape[0]):
        for j in range(min_sp_tree.shape[0]):
            if min_sp_tree[i, j] != 0 and min_sp_tree[i, j] < cutoff:
                filtered_tree[i, j] = min_sp_tree[i, j]
    return filtered_tree


def reduce_mst_clusters(min_sp_tree):
    filtered_tree = filter_mst(min_sp_tree)
    clusters = []
    for i in range(filtered_tree.shape[0]):
        for j in range(filtered_tree.shape[0]):
            if filtered_tree[i, j] != 0:
                placed = []
                for k, s in enumerate(clusters):
                    if i in s:
                        s.append(j)
                        placed.append(k)
                    elif j in s:
                        s.append(i)
                        placed.append(k)
                if not placed:
                    clusters.append([i, j])
                elif len(placed) == 2:
                    clusters[placed[0]].extend(clusters.pop(placed[1]))
                elif len(placed) > 2:
                    print("FUCK", placed)
    return clusters
