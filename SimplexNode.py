import alphax_utils as utils
import numpy as np
import warnings


class SimplexNode:
    """
    Simplex functions as a frozen set of coordinates for hashing purposes.
    It contains some extra functionality for convenience, such as circumradius and volume-equivalent.
    A Simplex instance represents a volume element of the Delaunay triangulation.
    In DIM=2, this is a triangle; in DIM=3, this is a tetrahedron.
    Simplex should comprise DIM+1 points.

    As of March 8, 2018:
        Only triangular (DIM=2) functionality is implemented here.
        See calculate
    """

    def __init__(self, coord_set):
        tuple_of_tuples = utils.tuple_map(coord_set)
        self.corners = frozenset(tuple_of_tuples)
        self.volume, self.circumradius = None, None
        self.calculate(coord_set)
        self.sort_value = self.circumradius

    def calculate(self, points):
        """
        :param points: numpy array of (n + 1, m) points correctly defining an n-dimensional simplex in m dimensions
        """
        # raise NotImplementedError("Implement this method please.")
        self.volume, self.circumradius = utils.cayley_menger_vr(np.array(points))

    def __hash__(self):
        return hash(self.corners)

    def __eq__(self, other):
        try:
            return self.sort_value == other.sort_value
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.sort_value != other.sort_value
        except AttributeError:
            return True

    def __ge__(self, other):
        try:
            return self.sort_value >= other.sort_value
        except AttributeError:
            return False

    def __gt__(self, other):
        try:
            return self.sort_value > other.sort_value
        except AttributeError:
            return False

    def __le__(self, other):
        try:
            return self.sort_value <= other.sort_value
        except AttributeError:
            return False

    def __lt__(self, other):
        try:
            return self.sort_value < other.sort_value
        except AttributeError:
            return False

    def __iter__(self):
        return iter(self.corners)
