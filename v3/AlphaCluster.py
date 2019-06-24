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

MINIMUM_MEMBERSHIP = 300


class AlphaCluster:
    def __init__(self, joint_simplex, circumradius, *children):
        # init with the (index of the) simplex that brought the children
        # together, as well as that simplex's circumradius

        # members is a set/frozenset of (int) simplex indices
        self.members = set()
        # alpha_map is a dictionary from circumradii to simplex indices
        self.alpha_map = dict()
        # circumradii is a sorted (INCREASING) numpy array of keys to alpha_map
        self.alphas = None
        # check if this is still mutable; should not be if it has a parent
        self.frozen = False
        # the top-level parent node of this cluster
        self.root = self
        # color variable that will be used by DENDROGRAM
        self.color = None
        # children is a short (2-3 item) list of AlphaCluster objects
        # The 0th index of children contains the largest child cluster
        self.children = AlphaCluster.set_children(list(children))
        # size is the number of members contained in+below this object
        self.size = sum(len(x) for x in self.children)
        for c in self.children:
            c.freeze(self)
        self.add(joint_simplex, circumradius)

    def add(self, item, circumradius):
        if self.frozen:
            raise RuntimeError("This shape is frozen.")
        # add an item to this cluster
        # need to specify circumradius
        self.members.add(item)
        self.size += 1
        self.alpha_map[circumradius] = item

    def add_child(self, cluster):
        self.size += cluster.size
        self.children.append(cluster)
        cluster.freeze(self)

    def engulf(self, cluster):
        self.members |= cluster.members
        self.alpha_map.update(cluster.alpha_map)
        if self.frozen and cluster.frozen:
            self.alphas = np.concatenate([self.alphas, cluster.alphas])
            self.alphas.sort(kind='mergesort')
        if cluster not in self.children:
            self.size += cluster.size
        self.children += cluster.children

    def collapse(self):
        if not self.frozen:
            raise RuntimeError("This shape is not frozen!")
        # also passes identity to largest subcluster
        for i, c in enumerate(list(self.children)):
            c.collapse()
            if (len(c.members) < MINIMUM_MEMBERSHIP) or (i == 0):
                self.engulf(c)
                self.children.remove(c)

    def freeze(self, parent):
        if self.frozen:
            raise RuntimeError("This shape is *already* frozen")
        self.frozen = True
        self.members = frozenset(self.members)
        self.alphas = np.array(sorted(self.alpha_map.keys()))
        self.set_root(parent)

    def set_root(self, root):
        self.root = root
        for c in self.children:
            c.set_root(root)

    def set_color(self, color):
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
        return self.alphas[0]

    def max_alpha(self):
        return self.alphas[-1]

    def get_smaller_than(self, alpha, recursive=False):
        # returns tuple of member indices
        #  belonging to members with CRs smaller than alpha
        member_indices = [self.alpha_map[cr] for cr in self.alphas[:np.searchsorted(self.alphas, alpha, side='right')]]
        if recursive:
            # also get anything under this
            member_indices.extend(sum((sc.get_smaller_than(alpha, recursive=recursive) for sc in self.children), []))
        return member_indices

    def __repr__(self):
        s = "{:d}/{:d}".format(len(self.members), self.size)
        if self.frozen:
            t = "{:.2E}/{:.2E}".format(self.alphas[-1], self.alphas[0])
        else:
            t = "::/{:.2E}".format(min(self.alpha_map.keys()))
        if self.color is not None:
            c = str(color)
        else:
            c = ""
        return "AlphaCluster({:s}|{:s}|{:s})".format(s, t, c)

    def __str__(self):
        return self.__repr__()

    def cluster_at_alpha(self, alpha, dkey, subclusters=False):
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
        members = self.get_smaller_than(alpha, recursive=subclusters)
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
    def set_children(children):
        if not children:
            # Leaf node
            return []
        else:
            # sets the largest subcluster as the 0th index child
            lc_i = max(range(len(children)), key=lambda x: len(children[x]))
            largest_child = children.pop(lc_i)
            return [largest_child] + children
