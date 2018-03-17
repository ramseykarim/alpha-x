from random import seed
import matplotlib.pyplot as plt
import numpy as np

import SimplexNode as Simp
import SimplexEdge as Edg
import alphax_utils as apy

# OK color seeds: 60341 93547 1337 334442
seed(60341)


def get_carina():
    data = np.genfromtxt("MYStiX_2014_Kuhn_Carina_2790.csv", delimiter=";", skip_header=26, usecols=[0, 1])
    return data


def get_data():
    # data = np.genfromtxt("../PyAlpha_drafting/test_data/combined1300.txt", skip_header=1)
    # data = np.genfromtxt("../PyAlpha_drafting/test_data/plummer1000.txt", skip_header=1)[:, 5:7]
    data = np.genfromtxt("../LearningR/s4.txt", skip_header=1)
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
        x, y, z = zip(*ps)
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


quickrun_get_membership()
