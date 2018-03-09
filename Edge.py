import alphax_utils as utils


class Edge(frozenset):
    """
    Edge represents a connection between two Simplices.
    It should contain DIM points.
    For example, in DIM=2, an edge is a shared line segment between two triangles.
        In DIM=3, it's a shared triangular face between two tetrahedrons.
    """
    def __new__(cls, coord_set):
        """
        Overriding the __new__ method so that we can use numpy arrays and not worry about order.
        This will shove everything into set-ready tuples.
        We could enforce restrictions on the length here.
        :param coord_set: iterable of coordinate pairs
        :return: the appropriate instance
        """
        tuple_of_tuples = utils.tuple_map(coord_set)
        # noinspection PyArgumentList
        return super().__new__(cls, tuple_of_tuples)

    def safe_pop(self):
        # Pop without creating set. frozenset inherits iterable.
        return next(iter(self))

    def get_others(self, one):
        # TODO implement this
        return None
