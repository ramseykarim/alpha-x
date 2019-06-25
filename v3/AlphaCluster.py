import numpy as np
from collections import Counter
"""
ALPHACLUSTER
Author: Ramsey Karim

This object represents a single, independent cluster of simplices associated
with discrete points.

This is the V3 iteration of this object; it is fundamentally different in
operation and implementation than the V2 iteration, but serves the same
general purpose and thus retains the name.
"""

MINIMUM_MEMBERSHIP = 1000


class AlphaCluster:
    def __init__(self):
        # init with no arguments; empty cluster

        # members is a set/frozenset of (int) simplex indices
        self.members = set()
        # alpha_map is a dictionary from circumradii to simplex indices
        self.alpha_map = dict()
        # alphas is a sorted (INCREASING) numpy array of keys to alpha_map
        self.alphas = None
        # size_map has same keys as alpha_map but maps to the size when key was largest shape
        self.size_map = dict()
        # alphas_main is a sorted (INCREASING) numpy array of keys to size_map
        self.alphas_main = None
        # check if this is still mutable; should not be if it has a parent
        self.frozen = False
        # the top-level parent node of this cluster
        self.root = self
        # color variable that will be used by DENDROGRAM
        self.color = None
        # children is a short (2-3 item) map of AlphaCluster object keys
        # the values are the circumradii at which the clusters merged
        self.children = dict()
        # size is the number of members contained in+below this object
        self.size = 0

    def add(self, item, circumradius):
        if self.frozen:
            raise RuntimeError("This shape is frozen.")
        # add an item to this cluster
        # need to specify circumradius
        self.members.add(item)
        self.size += 1
        self.alpha_map[circumradius] = item
        self.size_map[circumradius] = self.size

    def add_child(self, cluster, circumradius):
        self.size += cluster.size
        self.children[cluster] = circumradius
        self.size_map[circumradius] = self.size
        cluster.freeze(self)

    def add_all_children(self, clusters, circumradius):
        for c in clusters:
            self.add_child(c, circumradius)

    def engulf(self, cluster):
        self.members |= cluster.members
        self.alpha_map.update(cluster.alpha_map)
        if cluster not in self.children:
            self.size += cluster.size
        else:
            raise RuntimeWarning("ENGULF: why is this happening? (children)")
        self.children.update(cluster.children)
        # should extend this to updating simp_lookup

    def collapse(self):
        raise NotImplementedError("collapse is obsolete!")

    def freeze(self, parent):
        if self.frozen:
            raise RuntimeError("This shape is *already* frozen")
        self.frozen = True
        self.members = frozenset(self.members)
        self.alphas = np.array(sorted(self.alpha_map.keys()))
        self.alphas_main = np.array(sorted(self.size_map.keys()))
        self.set_root(parent)

    def set_root(self, root):
        self.root = root
        for c in self.children:
            c.set_root(root)

    def set_color(self, color):
        assert self.frozen
        if self.color is not None:
            raise RuntimeError("Color of {:s} is already set!".format(str(self)))
        self.color = color

    def get_color(self):
        if self.color is None:
            raise RuntimeError("Color of {:s} is not set!".format(str(self)))
        return self.color

    def __contains__(self, item):
        # check if an item is in or below this cluster
        return (item in self.members) or any(item in c for c in self.children)

    def __len__(self):
        return self.size

    def isleaf(self):
        return (not self.children)

    def min_alpha(self):
        return self.alphas_main[0]

    def max_alpha(self):
        return self.alphas_main[-1]

    def width_less_than(self, alpha):
        if alpha <= self.alphas_main[0]:
            return 0
        else:
            return self.size_map[self.alphas_main[np.searchsorted(self.alphas_main, alpha)-1]]

    def get_smaller_than(self, alpha):
        # returns tuple of member indices
        #  belonging to members with CRs smaller than alpha
        member_indices = [self.alpha_map[cr] for cr in self.alphas[:np.searchsorted(self.alphas, alpha, side='right')]]
        # also get anything under this as long as the cluster is included at this alpha
        member_indices.extend(sum((sc.get_smaller_than(alpha) for sc in self.children if self.children[sc] < alpha), []))
        return member_indices

    def __repr__(self):
        s = "{:d}/{:d}".format(len(self.members), self.size)
        if self.frozen:
            t = "{:.2E}/{:.2E}".format(self.alphas_main[-1], self.alphas_main[0])
        else:
            t = "::/{:.2E}".format(min(self.size_map.keys()))
        if self.color is not None:
            c = str(self.color)
        else:
            c = ""
        return "AlphaCluster({:s}|{:s}|{:s})".format(s, t, c)

    def __str__(self):
        return self.__repr__()

    def cluster_at_alpha(self, alpha, dkey):
        assert self.min_alpha() <= alpha <= self.max_alpha()
        # TODO: need boundary traverse to do it properly
        #   for boundary traverse, make sure to look up vertex_neighbor_vertices, neighbors
        # This is a hacky version; it will also return gap boundaries without distinction
        faces = []
        # number of dimensions plus one; number of vertices to a simplex
        nv = dkey.simplices.shape[1]
        # spawn off tuples of all vertices but one; x should be part of range(nv)
        iterface = lambda x: tuple(j for j in range(nv) if j!=x)
        # from alpha, get slice of CRs <= alpha and, via alpha_map dict, get simplex indices
        members = AlphaCluster.traverse(self.get_smaller_than(alpha), dkey)
        # grab the points associated with simplices
        points = dkey.points[dkey.simplices[members, :], :]
        points = set().union(*(map(tuple, m) for m in points))
        # loop through each vertex and collect faces from all the other vertices
        for i in range(nv):
            faces.extend(dkey.points[dkey.simplices[members, :][:, iterface(i)], :])
        # cast the face arrays to frozensets of tuples
        #  recall that coordinate order is preserved, but vertex order is arbitrary
        #  and therefore must be standardized (here, via hash function)
        # also count them
        face_counter = Counter(frozenset(map(tuple, f)) for f in faces)
        # if only counted once, not twice, then boundary
        boundary_faces = [f for f, count in face_counter.items() if count == 1]
        # Returns:
        # set(tuple(ndim x float coord)), (points)
        # list(frozensets(ndim x tuple(ndim x float coord))) (boundaries)
        return points, boundary_faces

    @staticmethod
    def traverse(simplices, dkey, largest=True):
        # traverse tests connectivity of simplices
        # it's like a boiled down version of the entire code, but no children
        # simplices is a sequence of simplex indices
        clusters = []
        remaining_simplices = set(simplices)
        while remaining_simplices:
            # if something is in traversal_stack, it is NOT in remaining_simplices
            traversal_stack = []
            traversal_stack.append(remaining_simplices.pop())
            cluster = []
            for simp_idx in traversal_stack:
                for n in dkey.neighbors[simp_idx]:
                    if (n in remaining_simplices) and (n > -1):
                        traversal_stack.append(n)
                        remaining_simplices.remove(n)
                cluster.append(simp_idx)
            clusters.append(cluster)
        if largest:
            return max(clusters, key=len)
        else:
            return clusters
