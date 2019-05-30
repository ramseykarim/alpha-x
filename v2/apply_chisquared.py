import numpy as np
import alphax_utils as apy
import matplotlib.pyplot as plt
import pickle

# Feed in discrete points from chi squared grid
#  and alpha shape them
sky = False
data_dir = "/n/sgraraid/filaments/data/TEST4/helpss_scratch_work/MantiPython/"
data_file = "points_file_smallest.pkl"
with open(data_dir + data_file, 'rb') as pfl:
    data = pickle.load(pfl)[:1000, :]
data[:, 0] *= 4./134.
data[:, 0] += 10
data[:, 1] *= 3./100.
data[:, 1] += 20
data[:, 2] *= 2./67.
data[:, 2] += 20
print("DATA SHAPE:", data.shape)
apy.initialize(data)
apy.KEY.alpha_step = 0.9
apy.KEY.orphan_tolerance = 100
apy.recurse()

tree_file = "atree_file_smallest.pkl"
with open(data_dir + tree_file, 'wb') as pfl:
    pickle.dump(apy.KEY, pfl)
sys.exit()

rectangles, base_width, lim = apy.dendrogram()
lim_alpha_lo, lim_alpha_hi = lim
fig = plt.figure()
d_ax, m_ax, surface_plotter = apy.prepare_plots(fig)
for r in rectangles:
    d_ax.add_artist(r)
d_ax.set_xlim([-0.05 * base_width, 1.05 * base_width])
d_ax.set_ylim([lim_alpha_lo * 0.9, lim_alpha_hi * 1.1])
if apy.DIM == 3:
    pass
    # d_ax.set_xlim([175e2, 24e3])
    # d_ax.set_ylim([0.227, 0.545])
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
    m_ax.set_xlim([10, 14])
    m_ax.set_ylim([20, 22])
    m_ax.set_aspect('equal')
if apy.DIM == 3:
    m_ax.set_zlabel("Z")
    m_ax.set_zlim([20, 23])
    # m_ax.set_zlabel("equivalent radial \"angle\" (deg)")
plt.show()
