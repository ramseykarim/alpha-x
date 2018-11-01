import numpy as np
import utils2_draft as apy
import matplotlib.pyplot as plt
import sys

def get_data():
    data = np.genfromtxt("../../PyAlpha_drafting/test_data/uniform600_gap_hiSN.txt", skip_header=1)  # GT = 15
    return data


def get_benchmark():
	path = "../../PyAlpha_drafting/test_data/Ssets/"
	fn = "s1.txt"
	return np.genfromtxt(path+fn)

data = get_benchmark()[:40, :]
apy.initialize(data)
apy.recurse()
rectangles, base_width, lim = apy.dendrogram()
lim_alpha_lo, lim_alpha_hi = lim
plt.figure()
ax = plt.subplot2grid((2, 4), (1, 3))
for r in rectangles:
	ax.add_artist(r)
ax.set_xlim([-0.05*base_width, 1.05*base_width])
ax.set_ylim([lim_alpha_lo*0.9, lim_alpha_hi*1.1])
ax.set_yscale('log')
ax.set_xlabel("Relative cluster size")
ax.set_ylabel("Alpha")
ax.invert_yaxis()
ax = plt.subplot2grid((2, 4), (0, 0), colspan=3, rowspan=2)
print()
for points, color in apy.naive_point_grouping():
	x, y = points
	ax.plot(x, y, marker='.', color=color, alpha=0.8, linestyle='None', markersize=3)
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show()
