import numpy as np
import utils2_draft as apy
import matplotlib.pyplot as plt
import sys


def get_data():
    return np.genfromtxt("../../PyAlpha_drafting/test_data/uniform600_gap_hiSN.txt", skip_header=1)  # GT = 15


def get_benchmark():
    path = "../../PyAlpha_drafting/test_data/Ssets/"
    fn = "s1.txt"
    return np.genfromtxt(path + fn, usecols=[0, 1])


"""
data = get_benchmark()

apy.ALPHA_STEP = 0.95
apy.ORPHAN_TOLERANCE = 100
apy.initialize(data)
apy.recurse()
"""

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
surfaces_list, points_list = apy.alpha_surfaces(12175)
for s in surfaces_list:
    m_ax.add_artist(s)


for points, color in points_list:
    m_ax.plot(*points, marker='.', color=color, alpha=0.8, linestyle='None', markersize=3)

m_ax.set_xlabel("X")
m_ax.set_ylabel("Y")
plt.show()
