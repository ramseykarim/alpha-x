from random import seed
import matplotlib.pyplot as plt
import numpy as np

import alphax_utils as apy

# OK color seeds: 60341 93547 1337 334442
seed(60341)


def get_carina():
    data = np.genfromtxt("MYStiX_2014_Kuhn_Carina_2790.csv", delimiter=";", skip_header=26, usecols=[0, 1])
    return data


def get_data():
    # data = np.genfromtxt("../PyAlpha_drafting/test_data/combined1300.txt", skip_header=1)
    data = np.genfromtxt("../PyAlpha_drafting/test_data/plummer1000.txt", skip_header=1)[:, 5:7]
    # data = np.genfromtxt("../LearningR/s4.txt", skip_header=1)
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
        plt.scatter(x, y, facecolor=c, edgecolor='k', alpha=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    plt.show()

plt.rcdefaults()
quickrun_get_membership()
