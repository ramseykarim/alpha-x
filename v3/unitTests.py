import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
import alphax_utils as utils
import AlphaCluster as alphac
import pickle

def get_test_data():
    srcfile = "../../PyAlpha_drafting/test_data/Ssets/s1.txt"
    # srcfile = "../../PyAlpha_drafting/test_data/filament5500_sampleNH2_betah1.80.dat"
    points = np.genfromtxt(srcfile)
    return points


def get_sco_ob():
    # RAW:
    with open("../../PyAlpha_drafting/test_data/Sco-3D.pkl", 'rb') as handle:
        d = pickle.load(handle)
    return d


def plot_triangles(triangle_list):
    # should be (ntri, ndim+1, ndim)
    fig, ax = plt.subplots()
    coll = []
    plt.plot(p[:, 0], p[:, 1], 'kx')
    for s in triangle_list:
        coll.append(Polygon(s))
    ax.add_collection(PatchCollection(coll, alpha=0.6, edgecolor='green', facecolor='k'))
    plt.show()

def test_basic_structure():
    """
    Simple version that uses nested sets to outline concept
    """
    p = get_test_data()
    print(p.shape)
    tri = Delaunay(p)
    print(tri.simplices.shape)
    ndim = p.shape[1]
    triangle_coords = p[tri.simplices, :]
    cr_array, vol_array = utils.cayley_menger_vr(triangle_coords)
    simp_sorted_idxs = np.argsort(cr_array)
    print("xxxxxxxxx")

    clusters = []
    for simp_idx  in simp_sorted_idxs:
        neighbors = list(tri.neighbors[simp_idx])
        included = [] # list is faster to build than deque
        for cluster in clusters:
            if any(adj_simp in cluster for adj_simp in neighbors):
                included.append(cluster)
        if len(included) == 0:
            # this simplex is isolated right now
            clusters.append({simp_idx}) # make new set for cluster
        elif len(included) == 1:
            # this simplex borders exactly one existing cluster
            included.pop().add(simp_idx)
        else:
            # included in 2 or more clusters; merge!
            largest_cluster = max(included, key=len)
            # for now, add the smaller cluster(s) as elements of the largest
            for cluster in included:
                if cluster is not largest_cluster:
                    clusters.remove(cluster)
                    largest_cluster |= cluster
                    largest_cluster.add(frozenset(cluster))
    # print(clusters)
    print("clusters: ", len(clusters))
    def printcluster(c, prefix=""):
        print(prefix+"---{ cluster size: ", end="")
        print(len([x for x in c if not isinstance(x, frozenset)]))
        subclusters = [x for x in c if isinstance(x, frozenset)]
        for sc in subclusters:
            printcluster(sc, prefix=prefix+"|  ")
        print(prefix+" }")
    # printcluster(clusters[0])

def test_AlphaCluster_structure():
    """
    Testing ground for structure of cluster code
    Uses AlphaCluster objects
    """
    p = get_test_data()
    tri = Delaunay(p)
    ndim = p.shape[1]
    triangle_coords = p[tri.simplices, :]
    cr_array, vol_array = utils.cayley_menger_vr(triangle_coords)
    simp_sorted_idxs = np.argsort(cr_array)
    simp_lookup = [None]*tri.simplices.shape[0]
    for simp_idx in simp_sorted_idxs:
        neighbors = [simp_lookup[n_idx] for n_idx in tri.neighbors[simp_idx]]
        included = set(n.root for n in neighbors if n is not None)
        cr = cr_array[simp_idx]
        if len(included) == 0:
            # simplex is isolated
            assigned_cluster = alphac.AlphaCluster(simp_idx, cr)
        elif len(included) == 1:
            # simplex borders exactly one existing cluster
            assigned_cluster = included.pop()
            assigned_cluster.add(simp_idx, cr)
        else:
            # included in 2 or more clusters
            assigned_cluster = alphac.AlphaCluster(simp_idx, cr, *included)
        simp_lookup[simp_idx] = assigned_cluster
    clusters = set(simp_lookup)
    root = simp_lookup[0].root
    root.freeze(root)
    all_root = {x.root for x in simp_lookup}
    def census(c, s):
        s.add(c)
        for sc in c.children:
            census(sc, s)
    init_clusters = set()
    census(root, init_clusters)
    print()
    assert root is next(iter(all_root))
    print("roots", len(all_root))
    print("clusters: ", len(clusters))
    # printcluster(clusters[0])
    root.collapse()
    remaining_clusters = set()
    census(root, remaining_clusters)
    print()
    print("initial clusters", len(init_clusters))
    print("remaining clusters", len(remaining_clusters))
    # lsi = [len(x.members) for x in init_clusters]
    lsr = [len(x.members) for x in remaining_clusters]
    plt.plot(lsr, '.')
    plt.show()
    return


def test_dendro():
    def counter():
        i = 0
        while True:
            yield i
            i += 1
    gen_i = counter()
    child1 = alphac.AlphaCluster(next(gen_i), 50)
    for x in range(30):
        child1.add(next(gen_i), np.random.uniform(low=30, high=49))
    child2 = alphac.AlphaCluster(next(gen_i), 70)
    for x in range(40):
        child2.add(next(gen_i), np.random.uniform(low=50, high=69))
    root = alphac.AlphaCluster(next(gen_i), 100, child1, child2)
    for x in range(100):
        root.add(next(gen_i), np.random.uniform(low=40, high=99))
    root.freeze(root)
    patches, base_width, lims = utils.dendrogram(root)
    fig, ax = plt.subplots()
    for p in patches:
        print(p)
        ax.add_artist(p)
    plt.xlim((0, 200))
    plt.ylim((10, 100))
    ax.invert_yaxis()
    plt.yscale('log')
    plt.show()

def test_AlphaCluster():
    """
    Testing ground for structure of cluster code
    Uses AlphaCluster objects
    """
    p = get_test_data()
    tri = Delaunay(p)
    ndim = p.shape[1]
    triangle_coords = p[tri.simplices, :]
    cr_array, vol_array = utils.cayley_menger_vr(triangle_coords)
    simp_sorted_idxs = np.argsort(cr_array)
    simp_lookup = [None]*tri.simplices.shape[0]

    for simp_idx in simp_sorted_idxs:
        if -1 in tri.neighbors[simp_idx]:
            # do not involve edge simplices
            continue
        # all neighboring clusters of this simplex
        neighbors = (simp_lookup[n_idx] for n_idx in tri.neighbors[simp_idx])
        # included if already counted by some cluster
        included = set(n.root for n in neighbors if n is not None)
        # circumradius of this simplex
        cr = cr_array[simp_idx]
        if len(included) == 0:
            # simplex is isolated
            assigned_cluster = alphac.AlphaCluster()
        else:
            # included in at least 1 cluster (assigned to largest/only cluster)
            assigned_cluster = max(included, key=len)
            included.remove(assigned_cluster)
            if any(len(x) < alphac.MINIMUM_MEMBERSHIP for x in included):
                # some small clusters need to be absorbed
                too_small_clusters = [x for x in included if len(x) < alphac.MINIMUM_MEMBERSHIP]
                for small_cluster in too_small_clusters:
                    assigned_cluster.engulf(small_cluster)
                    assert small_cluster.isleaf()
                    for m in small_cluster.members:
                        simp_lookup[m] = assigned_cluster
                    included.remove(small_cluster)
            # for clusters that are large enough, add as children
            for other_cluster in included:
                assigned_cluster.add_child(other_cluster)
        # add this simplex to its assigned cluster
        assigned_cluster.add(simp_idx, cr)
        simp_lookup[simp_idx] = assigned_cluster
    # all simplices should have the same root
    clusters = set(x.root for x in simp_lookup if x)
    root = max(clusters, key=len)
    clusters.remove(root)
    print("JOINING {} CLUSTERS".format(len(clusters)))
    root.add_all_children(clusters)
    root.freeze(root)

    fig = plt.figure()
    d_ax, m_ax, surface_plotter = utils.prepare_plots(fig, ndim)
    rectangles, base_width, lims = utils.dendrogram(root)
    for r in rectangles:
        d_ax.add_artist(r)
    d_ax.set_xlim((-0.05*base_width, base_width*1.05))
    d_ax.set_ylim((lims[0]*0.9, lims[1]*1.1))
    d_ax.invert_yaxis()
    d_ax.set_yscale('log')
    d_ax.set_xlabel("Relative cluster size")
    d_ax.set_ylabel("Alpha")
    d_ax.set_xticklabels([])
    surface_list, points_list = utils.naive_point_grouping(root, tri)
    for s in surface_list:
        surface_plotter(s)
    for points, color, opacity in points_list:
        m_ax.plot(*points, marker='.', color=color, alpha=opacity, linestyle='None', markersize=1)
    # for setter, i in zip((m_ax.set_xlim, m_ax.set_ylim, m_ax.set_zlim), range(3)):
    #     setter(np.sort(p[:, i])[(0, -1),])
    plt.show()
    return root


root = test_AlphaCluster()
