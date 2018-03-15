import alphax_utils as utils
import numpy as np
import warnings


class Simplex:
    """
    Simplex functions as a frozen set of coordinates for hashing purposes.
    It contains some extra functionality for convenience, such as circumradius and volume-equivalent.
    A Simplex instance represents a volume element of the Delaunay triangulation.
    In DIM=2, this is a triangle; in DIM=3, this is a tetrahedron.
    Simplex should comprise DIM+1 points.

    As of March 8, 2018:
        Only triangular (DIM=2) functionality is implemented here.
        See calculate_circumradius
    """

    def __init__(self, coord_set):
        tuple_of_tuples = utils.tuple_map(coord_set)
        self.corners = frozenset(tuple_of_tuples)
        self.volume, self.circumradius = None, None
        self.approx_center = None
        self.calculate_circumradius()

    def calculate_circumradius(self):
        # TODO implement DIM>2; all of this is for 2D
        pa, pb, pc = map(np.array, self.corners)
        a = np.sqrt(np.sum((pa - pb) ** 2.))
        b = np.sqrt(np.sum((pb - pc) ** 2.))
        c = np.sqrt(np.sum((pc - pa) ** 2.))
        s = (a + b + c) / 2.
        self.volume = np.sqrt(s * (s - a) * (s - b) * (s - c))
        if self.volume == 0:
            # Check for co-linear; send warning up to constructor call line
            warnings.warn("Shape appears to contain co-linear corners", category=RuntimeWarning, stacklevel=3)
            self.circumradius = np.inf
        else:
            self.circumradius = a * b * c / (4. * self.volume)
        # Approximate center
        self.approx_center = tuple((pa + pb + pc) / .3)

    def __hash__(self):
        return hash(self.corners)

    def __eq__(self, other):
        try:
            return self.circumradius == other.circumradius
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.circumradius != other.circumradius
        except AttributeError:
            return True

    def __ge__(self, other):
        try:
            return self.circumradius >= other.circumradius
        except AttributeError:
            return False

    def __gt__(self, other):
        try:
            return self.circumradius > other.circumradius
        except AttributeError:
            return False

    def __le__(self, other):
        try:
            return self.circumradius <= other.circumradius
        except AttributeError:
            return False

    def __lt__(self, other):
        try:
            return self.circumradius < other.circumradius
        except AttributeError:
            return False

    def __iter__(self):
        return iter(self.corners)
