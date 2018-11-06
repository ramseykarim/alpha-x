import numpy as np
import alphax_utils as apy
import matplotlib.pyplot as plt
import sys
import pickle


def get_data():
    return np.genfromtxt("../../PyAlpha_drafting/test_data/uniform600_gap_hiSN.txt", skip_header=1)  # GT = 15


def get_sco_ob():
    with open("../../PyAlpha_drafting/test_data/Sco-3D.pkl", 'rb') as handle:
        d = pickle.load(handle)
    return d[:100, :]


def get_carina():
    data = np.genfromtxt("../MYStiX_2014_Kuhn_Carina_2790.csv", delimiter=";", skip_header=26, usecols=[0, 1])
    return data


def get_benchmark():
    path = "../../PyAlpha_drafting/test_data/Ssets/"
    fn = "s1.txt"
    return np.genfromtxt(path + fn, usecols=[0, 1])


def get_benchmark_and_answers():
    path = "../../PyAlpha_drafting/test_data/Ssets/"
    fn = "s1.txt"
    fn_answers = "s1-label.pa"
    d = np.genfromtxt(path + fn, usecols=[0, 1])
    a = np.genfromtxt(path + fn_answers, skip_header=5)
    return d, a


def get_another_benchmark():
    path = "../../PyAlpha_drafting/test_data/"
    fn = "birch3.txt"
    return np.genfromtxt(path + fn, usecols=[0, 1])


data = get_sco_ob()

apy.initialize(data)
apy.KEY.alpha_step = 0.95
apy.KEY.orphan_tolerance = 100
apy.recurse()

rectangles, base_width, lim = apy.dendrogram()
lim_alpha_lo, lim_alpha_hi = lim
fig = plt.figure()
d_ax, m_ax = apy.prepare_plots(fig)
for r in rectangles:
    d_ax.add_artist(r)
d_ax.set_xlim([-0.05 * base_width, 1.05 * base_width])
d_ax.set_ylim([lim_alpha_lo * 0.9, lim_alpha_hi * 1.1])
d_ax.set_yscale('log')
d_ax.set_xlabel("Relative cluster size")
d_ax.set_ylabel("Alpha")
d_ax.invert_yaxis()

surfaces_list, points_list = apy.naive_point_grouping()
# surfaces_list, points_list = apy.alpha_surfaces(11286)
for s in surfaces_list:
    m_ax.add_artist(s)
for points, color, transparency in points_list:
    m_ax.plot(*points, marker='.', color=color, alpha=transparency, linestyle='None', markersize=3)

m_ax.set_xlabel("RA")
m_ax.invert_xaxis()
m_ax.set_ylabel("Dec")
plt.show()

# categories, leftovers = apy.find_membership(1000)

# NEED ANSWERS FOR THESE TO WORK
# apy.KEY.load_true_answers(answers)

# apy.check_answers_membership(categories, leftovers)
