import numpy as np
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
    def __init__(self, joint_simplex, circumradius, *children):
        # init with the (index of the) simplex that brought the children
        # together, as well as that simplex's circumradius

        # members is a set/frozenset of (int) simplex indices
        self.members = set()
        # circumradius_map is a dictionary from circumradii to simplex indices
        self.alpha_map = dict()
        # circumradii is a sorted numpy array of keys to circumradius_map
        self.alphas = None
        # check if this is still mutable; should not be if it has a parent
        self.frozen = False
        # the top-level parent node of this cluster
        self.root = self
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
            raise RuntimeError("This simplex is frozen.")
        # add an item to this cluster
        # need to specify circumradius
        self.members.add(item)
        self.size += 1
        self.alpha_map[circumradius] = item

    def engulf(self, cluster):
        self.members |= cluster.members
        self.alpha_map.update(cluster.circumradius_map)
        self.alphas = np.concatenate([self.alphas, cluster.circumradii])
        self.alphas.sort(kind='mergesort')

    def __contains__(self, item):
        # check if an item is in or below this cluster
        return (item in self.members) or any(item in c for c in self.children)

    def __len__(self):
        return self.size

    def freeze(self, parent):
        if self.frozen:
            raise RuntimeError("This simplex is *already* frozen")
        self.frozen = True
        self.members = frozenset(self.members)
        self.alphas = np.array(sorted(self.alpha_map.keys()))
        self.set_root(parent)

    def isleaf(self):
        return (not self.children)

    def set_root(self, root):
        self.root = root
        for c in self.children:
            c.set_root(root)

    def collapse(self):
        if not self.frozen:
            raise RuntimeError("This simplex is not frozen!")
        # also passes identity to largest subcluster
        for i, c in enumerate(self.children):
            c.collapse()
            if (i == 0) or (len(c) < MINIMUM_MEMBERSHIP):
                self.engulf(c)
                self.children.remove(c)

    def __repr__(self):
        s = "{:d}/{:d}".format(len(self.members), self.size)
        if self.frozen:
            t = "{:.2E}/{:.2E}".format(self.alphas[-1], self.alphas[0])
        else:
            t = "::/{:.2E}".format(min(self.alpha_map.keys()))
        return "AlphaCluster({:s}|{:s})".format(s, t)

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
