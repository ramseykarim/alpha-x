import numpy as np
import alphax_utils as apy
import matplotlib.pyplot as plt
import sys


def get_data():
    return np.genfromtxt("../../PyAlpha_drafting/test_data/uniform600_gap_hiSN.txt", skip_header=1)  # GT = 15


def get_benchmark():
    path = "../../PyAlpha_drafting/test_data/Ssets/"
    fn = "s1.txt"
    return np.genfromtxt(path + fn, usecols=[0, 1])[:80, :]


def get_benchmark_and_answers():
    path = "../../PyAlpha_drafting/test_data/Ssets/"
    fn = "s4.txt"
    fn_answers = "s4-label.pa"
    d = np.genfromtxt(path + fn, usecols=[0, 1])[:80, :]
    a = np.genfromtxt(path + fn_answers, skip_header=5)[:80]
    return d, a


def get_another_benchmark():
    path = "../../PyAlpha_drafting/test_data/"
    fn = "birch3.txt"
    return np.genfromtxt(path + fn, usecols=[0, 1])

data = get_another_benchmark()

# apy.initialize(data)
# apy.KEY.alpha_step = 0.95
# apy.KEY.orphan_tolerance = 100
# apy.recurse()

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

"""
points_list = apy.naive_point_grouping()
"""
surfaces_list, points_list = apy.alpha_surfaces(1000)
for s in surfaces_list:
    m_ax.add_artist(s)


for points, color in points_list:
    m_ax.plot(*points, marker='.', color=color, alpha=0.8, linestyle='None', markersize=3)

m_ax.set_xlabel("X")
m_ax.set_ylabel("Y")
plt.show()

categories, leftovers = apy.find_membership(1000)

# NEED ANSWERS FOR THESE TO WORK
# apy.KEY.load_true_answers(answers)

# apy.check_answers_membership(categories, leftovers)
