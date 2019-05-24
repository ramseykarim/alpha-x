import numpy as np
import alphax_utils as apy
import matplotlib.pyplot as plt
import sys

sky = False
data_dir = "../../PyAlpha_drafting/test_data/"
# dat_file = data_dir + "filament5500.dat"
dat_file = data_dir + "filament1937460_sampleNH2_betah1.80.dat"
data = np.genfromtxt(dat_file)

apy.initialize(data)
apy.KEY.alpha_step = 0.6
apy.KEY.orphan_tolerance = 1000
apy.recurse()

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

surfaces_list, points_list = apy.naive_point_grouping()
# surfaces_list, points_list = apy.alpha_surfaces(6.24)
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
    m_ax.set_aspect('equal')
if apy.DIM == 3:
    m_ax.set_zlabel("equivalent radial \"angle\" (deg)")
    # m_ax.set_xlim([232, 254])
    # m_ax.set_ylim([-35, -13])
    # m_ax.set_zlim([-20, 25])
    m_ax.set_xlim([237.5, 248])
    m_ax.set_ylim([-26.5, -17.5])
    m_ax.set_zlim([-10, 5])
plt.show()
