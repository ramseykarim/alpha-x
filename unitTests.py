from random import seed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from scipy.spatial import ConvexHull
from itertools import cycle
import sys

import SimplexNode as Simp
import SimplexEdge as Edg
import alphax_utils as apy
import MSTCluster as mstcluster

# OK color seeds: 60341 93547 1337 334442 332542
SEED = 635541
seed(SEED)


def get_carina():
    data = np.genfromtxt("MYStiX_2014_Kuhn_Carina_2790.csv", delimiter=";", skip_header=26, usecols=[0, 1])
    return data


def get_data():
    # data = np.genfromtxt("../PyAlpha_drafting/test_data/combined1300.txt", skip_header=1)
    # data = np.genfromtxt("../PyAlpha_drafting/test_data/uniform1000_gap.txt", skip_header=1)
    data = np.genfromtxt("../PyAlpha_drafting/test_data/uniform1200_gap.txt", skip_header=1)
    # data = np.genfromtxt("../PyAlpha_drafting/test_data/plummer1000.txt", skip_header=1)[:, 5:7]
    # data = np.genfromtxt("../LearningR/s1.txt", skip_header=1)
    return data


def get_data_fancy():
    data = np.genfromtxt("../PyAlpha_drafting/test_data/3MC.arff", skip_header=12, usecols=[0, 1], delimiter=',')
    # data = np.genfromtxt("../PyAlpha_drafting/test_data/cluto-t5-8k.arff", skip_header=12, usecols=[0, 1], delimiter=',')
    return data


def get_data_n_dim(n):
    if n > 5:
        raise ValueError("Can't support this many dimensions yet")
    data = np.genfromtxt("../PyAlpha_drafting/test_data/plummer1000.txt", skip_header=1)[:, 1:1+n]
    return data


def quickrun_get_membership():
    # This is for AlphaCluster and should be cleaner
    # Should easily support the MAIN_CLUSTER_THRESHOLD option
    data = get_carina()
    apy.QUIET = False
    apy.ORPHAN_TOLERANCE = 150
    apy.ALPHA_STEP = 0.97
    apy.PERSISTENCE_THRESHOLD = 3
    apy.MAIN_CLUSTER_THRESHOLD = 51
    apy.initialize(data)
    a_x = apy.recurse()
    colors, color_list, recs, base_width, lim = apy.dendrogram(a_x)
    lim_alpha_lo, lim_alpha_hi = lim
    plt.figure()
    ax = plt.subplot(122)
    for i, r_list in enumerate(recs):
        for r in r_list:
            r.set_facecolor(color_list[i])
            ax.add_artist(r)
    ax.set_xlim([-0.05 * base_width, 1.05 * base_width])
    ax.set_ylim([lim_alpha_lo * .9, lim_alpha_hi * 1.1])
    ax.set_yscale("log")
    ax.set_xlabel("# triangles")
    ax.set_ylabel("$\\alpha$")
    ax.invert_yaxis()
    ax = plt.subplot(121)
    for c, ps in colors.items():
        x, y = zip(*ps)
        plt.scatter(x, y, color=c, alpha=0.8, s=1)
    ax.invert_xaxis()
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    plt.show()


# noinspection SpellCheckingInspection
def test_cayley_menger():
    a = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    v, r = apy.cayley_menger_vr(a)
    print("Volume", v, "Circumradius", r)
    data = get_carina()[:100, :]
    apy.QUIET = False
    apy.ORPHAN_TOLERANCE = 150
    apy.ALPHA_STEP = 0.97
    apy.PERSISTENCE_THRESHOLD = 3
    apy.MAIN_CLUSTER_THRESHOLD = 51
    apy.initialize(data)
    v, r = zip(*[(s.volume, s.circumradius) for s in apy.KEY.simplices()])
    plt.plot(r, v, '.')
    sr = np.array(sorted(list(r)))
    coeff = (1 + np.sin(np.pi/6))*np.cos(np.pi/6)
    plt.plot(sr, (sr**2)*coeff, label="Equilateral triangle volume: upper bound on $v(r)$")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Circumradius")
    plt.ylabel("Volume")
    plt.legend()
    plt.show()


# noinspection SpellCheckingInspection
def test_cayley_menger_high_d():
    n = 4
    data = get_data_n_dim(n)[:30, :]
    apy.QUIET = False
    apy.ORPHAN_TOLERANCE = 150
    apy.ALPHA_STEP = 0.97
    apy.PERSISTENCE_THRESHOLD = 3
    apy.MAIN_CLUSTER_THRESHOLD = 51
    apy.initialize(data)
    v, r = zip(*[(s.volume, s.circumradius) for s in apy.KEY.simplices()])
    sr = np.array(sorted(list(r)))
    plt.plot(r, v, '.')
    plt.plot(sr, sr**n)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Circumradius")
    plt.ylabel("Volume")
    plt.legend()
    plt.show()


def test_cm_vs_old_vr():
    a = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    v, r = apy.cayley_menger_vr(a)
    v2, r2 = apy.old_vr(a)
    print("Volume", v, "Circumradius", r)
    print("Volume2", v2, "Circumradius2", r2)


def quickrun_get_membership_high_d():
    # This is for AlphaCluster and should be cleaner
    # Should easily support the MAIN_CLUSTER_THRESHOLD option
    data = get_data_n_dim(3)
    apy.QUIET = False
    apy.ORPHAN_TOLERANCE = 100
    apy.ALPHA_STEP = 0.97
    apy.PERSISTENCE_THRESHOLD = 3
    apy.MAIN_CLUSTER_THRESHOLD = 51
    apy.initialize(data)
    a_x = apy.recurse()
    colors, color_list, recs, base_width, lim = apy.dendrogram(a_x)
    lim_alpha_lo, lim_alpha_hi = lim
    plt.figure()
    ax = plt.subplot(122)
    for i, r_list in enumerate(recs):
        for r in r_list:
            r.set_color(color_list[i])
            ax.add_artist(r)
    ax.set_xlim([-0.05 * base_width, 1.05 * base_width])
    ax.set_ylim([lim_alpha_lo * .9, lim_alpha_hi * 1.1])
    ax.set_yscale("log")
    ax.set_xlabel("# triangles")
    ax.set_ylabel("$\\alpha$")
    ax.invert_yaxis()
    ax = plt.subplot(121)
    for c, ps in colors.items():
        x, y, z, w = zip(*ps)
        plt.scatter(x, y, color=c, alpha=0.8, s=1)
    ax.invert_xaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


def test_colors():
    colors = apy.get_colors()
    print(apy.rand_color(colors))
    print(apy.rand_color(colors))
    print(apy.rand_color(colors))
    print(apy.rand_color(colors))
    print(apy.rand_color(colors))
    print(apy.rand_color(colors))


def test_edge():
    data = get_carina()[:3, :]
    t = Simp.SimplexNode(data)
    e = Edg.SimplexEdge(data[1:3, :])
    p = np.array([data[2, :], data[1, :]])
    e2 = Edg.SimplexEdge(p)
    print(e)
    print(e2)
    print(t.volume)
    print(e.volume)
    print(e2.volume)
    print(hash(t))
    print(hash(e))
    print(hash(e2))
    s = set()
    s.add(e2)
    s.add(e)
    print(len(s))
    d = {e: "ok"}
    if e2 in d:
        print("Good!")


def profiler():
    data = get_carina()[:350, :]
    apy.QUIET = False
    apy.ORPHAN_TOLERANCE = 150
    apy.ALPHA_STEP = 0.9
    apy.PERSISTENCE_THRESHOLD = 3
    apy.MAIN_CLUSTER_THRESHOLD = 51
    apy.initialize(data)
    import cProfile
    import re
    cProfile.run("apy.recurse()")


def test_simplex_node_hash():
    a = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    s = Simp.SimplexNode(a)
    print(s)
    s1 = Simp.SimplexNode(np.array([[0, 0], [1, 0]], dtype=np.float64))
    print(s1)
    print(s - s1)
    e = Edg.SimplexEdge(np.array([[0, 0], [1, 0]], dtype=np.float64))
    print(e)
    print(hash(e))
    print(hash(s))
    print(hash(s1))
    print(e == s1)


def test_simplex_node_sort():
    data = get_carina()[:100, :]
    apy.QUIET = False
    apy.ORPHAN_TOLERANCE = 150
    apy.ALPHA_STEP = 0.97
    apy.PERSISTENCE_THRESHOLD = 3
    apy.MAIN_CLUSTER_THRESHOLD = 51
    apy.initialize(data)
    ts = apy.KEY.simplices()
    for t in sorted(ts):
        print(t.circumradius)


def prototype_circumradius(points):
    # This ONLY "works" for an n simplex embedded in n dimensions
    # Points MUST be of shape (n + 1, n)
    # This does NOT work.
    dim = points.shape[0] - 1
    l_ij = np.zeros(dim, dtype=np.float64)
    d = []
    j = points[0, :]
    p = points - j
    j = p[0, :]
    j2 = j.dot(j)
    for idx in range(dim):
        i = p[idx + 1, :]
        i2 = i.dot(i)
        d.append(j - i)
        l_ij[idx] = j2 - i2
    d = np.array(d).transpose()
    l_ij /= 2.
    c = np.linalg.solve(d, l_ij)
    cj = c - j
    return np.sqrt(cj.dot(cj))


def test_proto_circumradius():
    a = np.array([[0, 0], [1.5, 1], [0, 55]], dtype=np.float64)
    print(prototype_circumradius(a))
    print(apy.old_vr(a)[1])
    print(apy.cayley_menger_vr(a)[1])
    # data = get_carina()[:100, :]
    # tri = Delaunay(data)
    # simps, points = tri.simplices, tri.points
    # rs = np.zeros(simps.shape[0])
    # for i, s in enumerate(simps):
    #     ps = points[s]
    #     # v, r = apy.cayley_menger_vr(ps)
    #     v, r = apy.old_vr(ps)
    #     # r = prototype_circumradius(ps)
    #     rs[i] = r
    # print(rs)


def test_mst():
    data = get_carina()[:100, :]
    tri = Delaunay(data)
    csr_matrix = np.zeros((tri.points.shape[0], tri.points.shape[0]), dtype=np.float64)
    n_indices, n_indptr = tri.vertex_neighbor_vertices
    for i in range(tri.points.shape[0]):
        point_i = tri.points[i, :]
        neighbors = n_indptr[n_indices[i]:n_indices[i+1]]
        for j in neighbors:
            if i > j:
                sep = point_i - tri.points[j, :]
                sep = np.sqrt(sep.dot(sep))
                csr_matrix[i, j] = sep
                csr_matrix[j, i] = sep
    # noinspection PyTypeChecker
    min_sp_tree = mst(csr_matrix, overwrite=True).toarray()
    plot_list = []
    for i in range(tri.points.shape[0]):
        x1, y1 = tri.points[i]
        neighbors = []
        for j in range(tri.points.shape[0]):
            if min_sp_tree[i, j] != 0:
                neighbors.append(tri.points[j])
        for n in neighbors:
            plot_list.append([[x1, n[0]], [y1, n[1]]])
    plt.scatter(tri.points[:, 0], tri.points[:, 1], c='r')
    for p in plot_list:
        x, y = p
        plt.plot(x, y, '--', color='k')
    plt.show()


def test_mstcluster():
    data = get_carina()
    mstcluster.QUIET = True
    mstcluster.ORPHAN_TOLERANCE = 100
    mstcluster.ALPHA_STEP = 0.97
    mstcluster.PERSISTENCE_THRESHOLD = 3
    mstcluster.MAIN_CLUSTER_THRESHOLD = 51
    mstcluster.initialize(data)
    m_cluster = mstcluster.recurse()
    colors, color_list, recs, base_width, lim = mstcluster.dendrogram(m_cluster)
    print()
    lim_alpha_lo, lim_alpha_hi = lim
    plt.figure()
    ax = plt.subplot(122)
    for i, r_list in enumerate(recs):
        for r in r_list:
            r.set_color(color_list[i])
            ax.add_artist(r)
    ax.set_xlim([-0.05 * base_width, 1.05 * base_width])
    ax.set_ylim([lim_alpha_lo * .9, lim_alpha_hi * 1.1])
    ax.set_yscale("log")
    ax.set_xlabel("# points")
    ax.set_ylabel("$\\alpha")
    ax.invert_yaxis()
    ax = plt.subplot(121)
    for c, ps in colors.items():
        x, y = zip(*ps)
        plt.scatter(x, y, color=c, alpha=0.8, s=1)
    ax.invert_xaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

    seed(SEED)
    # This is for AlphaCluster and should be cleaner
    # Should easily support the MAIN_CLUSTER_THRESHOLD option
    data = get_carina()
    apy.QUIET = True
    apy.ORPHAN_TOLERANCE = 100
    apy.ALPHA_STEP = 0.97
    apy.PERSISTENCE_THRESHOLD = 3
    apy.MAIN_CLUSTER_THRESHOLD = 51
    apy.initialize(data)
    a_x = apy.recurse()
    colors, color_list, recs, base_width, lim = apy.dendrogram(a_x)
    lim_alpha_lo, lim_alpha_hi = lim
    plt.figure()
    ax = plt.subplot(122)
    for i, r_list in enumerate(recs):
        for r in r_list:
            r.set_facecolor(color_list[i])
            ax.add_artist(r)
    ax.set_xlim([-0.05 * base_width, 1.05 * base_width])
    ax.set_ylim([lim_alpha_lo * .9, lim_alpha_hi * 1.1])
    ax.set_yscale("log")
    ax.set_xlabel("# triangles")
    ax.set_ylabel("$\\alpha$")
    ax.invert_yaxis()
    ax = plt.subplot(121)
    for c, ps in colors.items():
        x, y = zip(*ps)
        plt.scatter(x, y, color=c, alpha=0.8, s=1)
    ax.invert_xaxis()
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    plt.show()


def test_classic_mstcluster_cdf():
    n = 100
    data = get_carina()
    tri = Delaunay(data)
    cdf = mstcluster.gen_cdf(mstcluster.get_mst_edges(tri), n=n)
    nrange = np.arange(n)
    plt.plot(nrange, cdf)
    #  #  fit from 0 to 12 and 60 to 100
    fit1 = np.polyfit(nrange[:12], cdf[:12], deg=1)
    fit2 = np.polyfit(nrange[60:], cdf[60:], deg=1)
    commonx = (fit1[1] - fit2[1]) / (fit2[0] - fit1[0])
    commony = fit1[0]*commonx + fit1[1]
    apply_poly = lambda x, f: f[0]*x + f[1]
    plt.plot(nrange[:25], apply_poly(nrange[:25], fit1))
    plt.plot(nrange[5:], apply_poly(nrange[5:], fit2))
    plt.plot([commonx], [commony], 'x')
    print(commonx, commony)
    plt.show()


def test_classic_mstcluster_mst():
    data = get_carina()
    tri = Delaunay(data)
    min_sp_tree = mstcluster.prepare_mst_simple(tri)
    filtered_tree = mstcluster.filter_mst(min_sp_tree)
    plot_list = []
    color_list = []
    for i in range(tri.points.shape[0]):
        x1, y1 = tri.points[i]
        neighbors = []
        colors = []
        for j in range(tri.points.shape[0]):
            if filtered_tree[i, j] != 0:
                neighbors.append(tri.points[j])
                colors.append('r')
            elif min_sp_tree[i, j] != 0:
                neighbors.append(tri.points[j])
                colors.append('b')
        for n, c in zip(neighbors, colors):
            plot_list.append([[x1, n[0]], [y1, n[1]]])
            color_list.append(c)
    # plt.scatter(tri.points[:, 0], tri.points[:, 1], c='r')
    for p, c in zip(plot_list, color_list):
        x, y = p
        plt.plot(x, y, '--', color=c)
    plt.show()


def test_classic_mstcluster_cluster():
    data = get_carina()
    tri = Delaunay(data)
    min_sp_tree = mstcluster.prepare_mst_simple(tri)
    clusters = mstcluster.reduce_mst_clusters(min_sp_tree)
    min_points = 30
    plt.plot(tri.points[:, 0], tri.points[:, 1], 'k,')
    for c in clusters:
        if len(c) >= min_points:
            coords = np.array([tri.points[i, :] for i in c])
            hull = ConvexHull(coords)
            # noinspection PyUnresolvedReferences
            for simplex in hull.simplices:
                plt.plot(coords[simplex, 0], coords[simplex, 1], '-', color='r', lw=1)
    ax = plt.gca()
    ax.invert_xaxis()
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    plt.show()


def test_compare_mst_alpha():
    data = get_carina()
    apy.QUIET = False
    apy.ORPHAN_TOLERANCE = 100
    apy.ALPHA_STEP = 0.97
    apy.PERSISTENCE_THRESHOLD = 3
    apy.MAIN_CLUSTER_THRESHOLD = 51
    apy.initialize(data)
    a_x = apy.recurse()
    colors, color_list, recs, base_width, lim = apy.dendrogram(a_x)
    lim_alpha_lo, lim_alpha_hi = lim
    plt.figure()
    ax_dend = plt.subplot(122)
    for i, r_list in enumerate(recs):
        for r in r_list:
            r.set_facecolor(color_list[i])
            ax_dend.add_artist(r)
    ax_dend.set_xlim([-0.05 * base_width, 1.05 * base_width])
    ax_dend.set_ylim([lim_alpha_lo * .9, lim_alpha_hi * 1.1])
    ax_dend.set_yscale("log")
    ax_dend.set_xlabel("# triangles")
    ax_dend.set_ylabel("$\\alpha$")
    ax_dend.invert_yaxis()
    ax_points = plt.subplot(121)
    for c, ps in colors.items():
        x, y = zip(*ps)
        plt.scatter(x, y, color=c, alpha=0.8, s=1)
    ax_points.invert_xaxis()
    ax_points.set_xlabel("RA")
    ax_points.set_ylabel("Dec")

    print("Halfway..")

    tri = apy.KEY.delaunay
    min_sp_tree = mstcluster.prepare_mst_simple(tri)
    cutoff_alpha = mstcluster.get_cdf_cutoff(min_sp_tree)
    clusters = mstcluster.reduce_mst_clusters(min_sp_tree)
    for c in clusters:
        if len(c) >= apy.ORPHAN_TOLERANCE/2.:
            coords = np.array([tri.points[i, :] for i in c])
            hull = ConvexHull(coords)
            # noinspection PyUnresolvedReferences
            for simplex in hull.simplices:
                plt.plot(coords[simplex, 0], coords[simplex, 1], '-', color='r', lw=2)
    ax_dend.plot([0, base_width], [cutoff_alpha, cutoff_alpha], '--', 'r')
    plt.show()


def quickrun_mean_vps():
    data = get_carina()
    apy.QUIET = False
    apy.ORPHAN_TOLERANCE = 100
    apy.ALPHA_STEP = 0.97
    apy.PERSISTENCE_THRESHOLD = 1
    apy.MAIN_CLUSTER_THRESHOLD = 51
    apy.initialize(data)
    a_x = apy.recurse()
    stack = [a_x]
    mean_vpss = []
    while stack:
        a = stack.pop()
        mean_vpss.append((a.alpha_range, a.mean_vps))
        stack += a.subclusters
    colors, color_list, recs, base_width, lim = apy.dendrogram(a_x)
    lim_alpha_lo, lim_alpha_hi = lim
    plt.figure()
    ax = plt.subplot(223)
    for i, r_list in enumerate(recs):
        for r in r_list:
            r.set_facecolor(color_list[i])
            ax.add_artist(r)
    ax.set_xlim([-0.05 * base_width, 1.05 * base_width])
    ax.set_ylim([lim_alpha_lo * .9, lim_alpha_hi * 1.1])
    ax.set_yscale("log")
    ax.set_xlabel("# triangles")
    ax.set_ylabel("$\\alpha$")
    ax.invert_yaxis()
    ax = plt.subplot(221)
    for c, ps in colors.items():
        x, y = zip(*ps)
        plt.scatter(x, y, color=c, alpha=0.8, s=1)
    # ax.invert_xaxis()
    ax.invert_yaxis()
    # ax.set_xlabel("RA")
    # ax.set_ylabel("Dec")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax = plt.subplot(122)
    print()
    coeff = (1 + np.sin(np.pi/6))*np.cos(np.pi/6)
    for m, c in zip(mean_vpss, color_list):
        a_r, m_vps = m
        normed_vps = [abs(x) for x, a in zip(m_vps, a_r[:-1])]
        plt.plot(a_r[:-1], normed_vps, '-', color=c)
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("Mean Volume per Simplex, normed to equilateral")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlim([lim_alpha_lo * .9, lim_alpha_hi * 1.1])
    # ax.set_ylim([.01, 100])
    ax.invert_xaxis()
    plt.show()


def test_boundary():
    # This is for AlphaCluster and should be cleaner
    # Should easily support the MAIN_CLUSTER_THRESHOLD option
    data = get_data()
    apy.QUIET = False
    apy.ORPHAN_TOLERANCE = 50
    apy.ALPHA_STEP = .97 #0.97
    apy.PERSISTENCE_THRESHOLD = 3
    apy.MAIN_CLUSTER_THRESHOLD = 51
    apy.initialize(data)
    a_x = apy.recurse()
    colors, color_list, recs, base_width, lim = apy.dendrogram(a_x)
    lim_alpha_lo, lim_alpha_hi = lim
    plt.figure()
    ax = plt.subplot(211)
    for i, r_list in enumerate(recs):
        for r in r_list:
            r.set_facecolor(color_list[i])
            ax.add_artist(r)
    ax.set_xlim([-0.05 * base_width, 1.05 * base_width])
    ax.set_ylim([lim_alpha_lo * .9, lim_alpha_hi * 1.1])
    ax.set_yscale("log")
    ax.set_xlabel("# triangles")
    ax.set_ylabel("$\\alpha$")
    ax.invert_yaxis()
    ax = plt.subplot(224)

    print()
    stack = [a_x]
    while stack:
        a = stack.pop()
        for b_list in a.boundary_range:
            if b_list and len(b_list) > 1:
                # cool_colors = cycle(['k', 'b', 'g', 'y', 'orange', 'navy', 'cyan'])
                for b in b_list:
                    # clr = next(cool_colors)
                    for s in b[1]:
                        ax.add_artist(Polygon(s.coord_array(), alpha=0.1, facecolor='k', edgecolor=None))
                        # x, y = [], []
                        # for p in e:
                        #     x.append(p[0]), y.append(p[1])
                        # if clr == 'r':
                        #     plt.plot(x, y, '--', color=clr)
                        # else:
                        #     plt.plot(x, y, '-', color=clr)
    for c, ps in colors.items():
        x, y = zip(*ps)
        plt.scatter(x, y, color=c, alpha=0.8, s=1)
    ax.invert_xaxis()
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax = plt.subplot(223)
    plt.scatter(apy.KEY.delaunay.points[:, 0], apy.KEY.delaunay.points[:, 1], color='k', alpha=0.6, s=1)
    ax.invert_xaxis()
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    plt.show()


test_boundary()
