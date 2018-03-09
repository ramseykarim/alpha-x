import Edge as Edg
import Simplex as Sim


class DelaunayKey:
    def __init__(self, delaunay):
        self.delaunay = delaunay
        self._sim_to_edg = {}
        self._edg_to_sim = {}
        self.initialize_keys()

    def get_simplex(self, edge):
        return self._edg_to_sim[edge]

    def get_edge(self, simplex):
        return self._sim_to_edg[simplex]

    def simplices(self):
        return iter(self._sim_to_edg)

    def initialize_keys(self):
        for simplex_indices in self.delaunay.simplices:
            simplex = Sim.Simplex([self.delaunay.points[i] for i in simplex_indices])
            edges = [Edg.Edge([self.delaunay.points[vtx_id]
                               for j, vtx_id in enumerate(simplex_indices) if j != i])
                     for i in range(len(simplex_indices))]
            self._sim_to_edg[simplex] = edges
            for e in edges:
                if e in self._edg_to_sim:
                    self._edg_to_sim[e].append(simplex)
                else:
                    self._edg_to_sim[e] = [simplex]
