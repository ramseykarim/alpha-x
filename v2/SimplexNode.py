import alphax_utils as utils
import numpy as np
import SimplexEdge as Edge


class SimplexNode(Edge.SimplexEdge):
    """
    Simplex functions as a frozen set of coordinates for hashing purposes.
    It contains some extra functionality for convenience, such as circumradius and volume-equivalent.
    A Simplex instance represents a volume element of the Delaunay triangulation.
    In DIM=2, this is a triangle; in DIM=3, this is a tetrahedron.
    Simplex should comprise DIM+1 points.

    Good for N dimensions (Cayley-Menger determinant for circumradius and volume)
    """

    def __init__(self, coord_set):
        self.circumradius = None
        super(SimplexNode, self).__init__(coord_set)

    def calculate(self, points):
        """
        :param points: numpy array of (n + 1, m) points correctly defining an n-dimensional simplex in m dimensions
        """
        self.volume, self.circumradius = utils.cayley_menger_vr(np.array(points))
        self.centroid = np.mean(points, axis=0)

    def __gt__(self, other):
        try:
            return self.circumradius > other.circumradius
        except AttributeError:
            return False

    def __lt__(self, other):
        try:
            return self.circumradius < other.circumradius
        except AttributeError:
            return False
