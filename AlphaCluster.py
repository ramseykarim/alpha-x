import alphax_utils as utils


class AlphaCluster:
    """
    AlphaCluster functions as a node in an alpha cluster tree.
    AlphaCluster nodes may have parents and/or children.
    Each node represents a physical cluster of points as characterized by the Delaunay triangulation,
    or more specifically, by the alpha-restricted simplicial complex subset of the Delaunay triangulation.
    """

    def __init__(self, cluster_elements, alpha_level=None):
        """
        AlphaCluster instance constructor.
        KEY should be global and its getter functions available.
        :param cluster_elements: iterable. Contains all triangles that this cluster will manage
        :param alpha_level: float. Indicates the alpha at which this node is born. If it it set to None, it indicates
            that this is the parent node and alpha_level should be taken from the largest triangle in cluster_elements.
        """
        self.cluster_elements = sorted(list(cluster_elements))
        if alpha_level is None:
            self.alpha_range = [self.cluster_elements[-1].circumradius/utils.ALPHA_STEP]
        else:
            self.alpha_range = [alpha_level]
        self.volume_range = [sum([x.volume for x in cluster_elements])]
        self.member_range = [len(self.cluster_elements)]
        self.subclusters = []
        if not utils.QUIET:
            print(utils.SP + "<branch init_size=%d>" % len(self.cluster_elements))
            utils.SP += "|  "
        self.exhaust_cluster()
        if not utils.QUIET:
            utils.SP = utils.SP[:-3]
            print(utils.SP + "</branch persist=%d>" % len(self.alpha_range))

    def exhaust_cluster(self):
        """
        Begin with all triangles in this cluster
        Iterate next alpha_step until cluster breaks
        Generate children
        :return: this instance
        """
        remaining_simplices = self.cluster_elements.copy()
        totally_finished = False
        next_alpha = None
        while not totally_finished:  # I feel like we could improve this implementation...
            coherent = True
            while remaining_simplices and coherent:
                next_alpha = self.alpha_range[-1] * utils.ALPHA_STEP
                dropped_simplex_volume, drop_score = 0, 0
                self.member_range.append(len(remaining_simplices))
                while remaining_simplices and remaining_simplices[-1].circumradius > next_alpha:
                    dropped_simplex = remaining_simplices.pop()
                    dropped_simplex_volume += dropped_simplex.volume
                    drop_score += 1
                self.alpha_range.append(next_alpha)
                if drop_score == 0:
                    continue  # No need to traverse the same graph as last time
                cluster_list = utils.traverse_cluster(set(remaining_simplices))
                orphan_tolerance = utils.ORPHAN_TOLERANCE
                cluster_list = [x for x in cluster_list if len(x) > orphan_tolerance]
                # cluster_list either has several elements, 1 element, or 0 elements
                if len(cluster_list) > 1:
                    coherent = False  # Several! Send below to create children
                else:
                    self.volume_range.append(self.volume_range[-1] - dropped_simplex_volume)
                    # There was a very confusing comment in apy_9 about the line below
                    # I have figured out what I meant:
                    # Below, we discard the dropped simplices. Maybe we want information about them!
                    # If we proceed as we do below, we lose that information forever.
                    # I think we should proceed as below and quit tracking orphaned clusters entirely.
                    if cluster_list:
                        remaining_simplices = sorted(list(cluster_list[0]))
                    else:
                        remaining_simplices = cluster_list
            # Just finished looping through remaining_simplices! Something stopped it.
            # Two options: we lost coherency (split cluster) or we ran out of simplices and leafed.
            if coherent:  # Definitely has 1 cluster with 0 simplices left; never split
                # This is a leaf: no more simplices
                totally_finished = True
            else:  # Definitely has more than 1 cluster!
                if utils.MAIN_CLUSTER_THRESHOLD > 0:  # Possibility to make children and still persist
                    active_simplices = sum([len(sc) for sc in cluster_list])
                    largest_subcluster = max(cluster_list, key=lambda x: len(x))
                    if len(largest_subcluster) > active_simplices * utils.MAIN_CLUSTER_THRESHOLD:
                        remaining_simplices = sorted(list(largest_subcluster))  # Still persists with children!
                        cluster_list.remove(largest_subcluster)  # Children made from remainder
                    else:
                        totally_finished = True
                else:  # Dude there HAS to be a better way to control this omg
                    totally_finished = True
                # Get previous children
                old_children = self.subclusters
                # Make subclusters
                self.subclusters = [AlphaCluster(sc, alpha_level=next_alpha) for sc in cluster_list]
                # # Filter by persistence threshold
                # self.subclusters = [(ac if len(ac.alpha_range) >= utils.PERSISTENCE_THRESHOLD + 1 else ac.subclusters)
                #                     for ac in self.subclusters]
                # Flatten list of subclusters
                self.subclusters = list(utils.flatten([ac for ac in self.subclusters if ac is not None]))
                # Reintroduce previous children
                self.subclusters = old_children + self.subclusters
        self.member_range.append(0)
        return self