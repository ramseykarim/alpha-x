import numpy as np
import alphax_utils as apy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import pickle


sky = False


def get_data():
    return np.genfromtxt("../../PyAlpha_drafting/test_data/uniform600_gap_hiSN.txt", skip_header=1)  # GT = 15


def get_sco_ob():
    # RAW:
    with open("../../PyAlpha_drafting/test_data/Sco-3D.pkl", 'rb') as handle:
        d = pickle.load(handle)
    # KEY:
    # with open("../../PyAlpha_drafting/test_data/Sco-AX.pkl", 'rb') as handle:
    #     d = pickle.load(handle)
    global sky
    sky = True
    return d


def get_carina():
    d = np.genfromtxt("../MYStiX_2014_Kuhn_Carina_2790.csv", delimiter=";", skip_header=26, usecols=[0, 1])
    global sky
    sky = True
    return d


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


data, answers = get_benchmark_and_answers()

apy.initialize(data)
apy.KEY.alpha_step = 0.95
apy.KEY.orphan_tolerance = 80
apy.recurse()

# apy.KEY = get_sco_ob()
# apy.DIM = 3

rectangles, base_width, lim = apy.dendrogram()
lim_alpha_lo, lim_alpha_hi = lim
fig = plt.figure()
d_ax, m_ax, surface_plotter = apy.prepare_plots(fig)
for r in rectangles:
    d_ax.add_artist(r)
d_ax.set_xlim([-0.05 * base_width, 1.05 * base_width])
d_ax.set_ylim([lim_alpha_lo * 0.9, lim_alpha_hi * 1.1])
if apy.DIM == 3:
    d_ax.set_xlim([175e2, 24e3])
    d_ax.set_ylim([0.227, 0.545])
d_ax.set_yscale('log')
d_ax.set_xlabel("Relative cluster size")
d_ax.set_ylabel("Alpha")
d_ax.set_xticklabels([])
d_ax.invert_yaxis()

# surfaces_list, points_list = apy.naive_point_grouping()
surfaces_list, points_list = apy.alpha_surfaces(11200)
for s in surfaces_list:
    surface_plotter(s)
for points, color, transparency in points_list:
    m_ax.plot(*points, marker='.', color=color, alpha=transparency, linestyle='None', markersize=1)

if sky:
    m_ax.set_xlabel("RA")
    m_ax.invert_xaxis()
    m_ax.set_ylabel("Dec")
else:
    m_ax.set_xlabel("X")
    m_ax.set_ylabel("Y")
if apy.DIM == 3:
    m_ax.set_zlabel("equivalent radial \"angle\" (deg)")
    # m_ax.set_xlim([232, 254])
    # m_ax.set_ylim([-35, -13])
    # m_ax.set_zlim([-20, 25])
    m_ax.set_xlim([237.5, 248])
    m_ax.set_ylim([-26.5, -17.5])
    m_ax.set_zlim([-10, 5])
plt.show()

categories, leftovers = apy.find_membership(11200)

# NEED ANSWERS FOR THESE TO WORK
apy.KEY.load_true_answers(answers)

apy.check_answers_membership(categories, leftovers)
