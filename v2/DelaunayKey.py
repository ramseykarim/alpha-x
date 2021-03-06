import SimplexEdge as Edg
import SimplexNode as Sim
import sys


class DelaunayKey:
    # The big idea of the key is that it doesn't change
    # only TreeIndex will be updated during operation
    # (this is why the functionality of TreeIndex is quarantined to an attribute of Key)

    def __init__(self, delaunay):
        self.alpha_step = 0.5
        self.orphan_tolerance = 2
        self.delaunay = delaunay
        self._sim_to_edg = {}
        self._edg_to_sim = {}
        self.treeIndex = AlphaTreeIndex()
        self.initialize_keys()
        self.alpha_root = None
        self.true_categories = None
        self.dim = None

    def get_simplex(self, edge):
        return self._edg_to_sim[edge]

    def get_edge(self, simplex):
        return self._sim_to_edg[simplex]

    def simplices(self):
        return iter(self._sim_to_edg)

    def initialize_keys(self):
        total_simplices = len(self.delaunay.simplices)
        last_count = -1
        print("TOTAL # SIMPLICES", total_simplices)
        for count, simplex_indices in enumerate(self.delaunay.simplices):
            now_count = int(count*100/total_simplices)
            if last_count < now_count:
                sys.stdout.write(f"{now_count}%\r")
                sys.stdout.flush()
                last_count = now_count
            simplex = Sim.SimplexNode([self.delaunay.points[i] for i in simplex_indices])
            edges = [Edg.SimplexEdge([self.delaunay.points[vtx_id] for j, vtx_id in enumerate(simplex_indices) if j!= i]) for i in range(len(simplex_indices))]
            self._sim_to_edg[simplex] = edges
            # Good place to start defining null boundary! If num edges is fewer than DIM, this borders the NULL
            for e in edges:
                if e in self._edg_to_sim:
                    self._edg_to_sim[e].append(simplex)
                else:
                    self._edg_to_sim[e] = [simplex]
        print()

    def add_branch(self, root):
        self.alpha_root = root

    def load_true_answers(self, true_answers):
        self.true_categories = {}
        for i, p in enumerate(self.delaunay.points):
            self.true_categories[tuple(p)] = true_answers[i]


class AlphaTreeIndex:

    def __init__(self):
        # this should *belong* to the KEY

        # alpha range uses a float alpha value as a key
        # values are sets of alpha clusters
        self.alpha_range = {}
        self.colors = set()

    def append_cluster(self, alpha, cluster):
        if alpha not in self.alpha_range:
            self.alpha_range[alpha] = set()
        self.alpha_range[alpha].add(cluster)
