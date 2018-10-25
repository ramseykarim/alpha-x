import utils2_draft as utils
import numpy as np


class SimplexEdge(frozenset):
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

    def __init__(self, coord_set):
        super().__init__()
        self.volume = None
        self.centroid = None
        self.calculate(coord_set)

    def calculate(self, points):
        self.volume = utils.cayley_menger_volume(np.array(points))
        self.centroid = np.mean(points, axis=0)

    def safe_pop(self):
        # Pop without creating set. frozenset inherits iterable.
        return next(iter(self))

    def coord_array(self):
        return np.array(list(map(np.array, self)))
