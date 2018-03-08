import alphax_utils


class AlphaCluster:
    """
    AlphaCluster functions as a node in an alpha cluster tree.
    AlphaCluster nodes may have parents and/or children.
    Each node represents a physical cluster of points as characterized by the Delaunay triangulation,
    or more specifically, by the alpha-restricted simplicial complex subset of the Delaunay triangulation.
    """

    def __init__(self, cluster_triangles, alpha_level):
        """
        AlphaCluster instance constructor.
        KEY should be global and its getter functions available.
        :param cluster_triangles: iterable. Contains all triangles that this cluster will manage
        :param alpha_level: float. Indicates the alpha at which this node is born. If it it set to -1, it indicates
            that this is the parent node and alpha_level should be taken from the largest triangle in cluster_triangles.
        """
        self.cluster_triangles = sorted(list(cluster_triangles))
        self.alpha_range = [alpha_level]
        self.area_range = [sum([x.area for x in cluster_triangles])]
        self.subclusters = None
        if not alphax_utils.QUIET:
            return 0

    # TODO finish this class; need to implement alpha_level capability
