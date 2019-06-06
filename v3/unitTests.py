import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import alphax_utils as utils
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform

def test_something():
	srcfile = "/home/rkarim/Research/AlphaX/PyAlpha_drafting/test_data/Ssets/s1.txt"
	points = np.genfromtxt(srcfile)
	return points

def plot_triangles(triangle_list):
	# should be (ntri, ndim+1, ndim)
	fig, ax = plt.subplots()
	coll = []
	plt.plot(p[:, 0], p[:, 1], 'kx')
	for s in triangle_list:
		coll.append(Polygon(s))
	ax.add_collection(PatchCollection(coll, alpha=0.6, edgecolor='green', facecolor='k'))
	plt.show()


p = test_something()
print(p.shape)
tri = Delaunay(p)
print(tri.simplices.shape)
ndim = p.shape[1]
triangle_coords = p[tri.simplices, :]
cr_array, vol_array = utils.cayley_menger_vr(triangle_coords)
simp_sorted_idxs = np.argsort(cr_array)
print("xxxxxxxxx")

clusters = []
for simp_idx  in simp_sorted_idxs:
	neighbors = list(tri.neighbors[simp_idx])
	included = [] # list is faster to build than deque
	for cluster in clusters:
		if any(adj_simp in cluster for adj_simp in neighbors):
			included.append(cluster)
	if len(included) == 0:
		# this simplex is isolated right now
		clusters.append({simp_idx}) # make new set for cluster
	elif len(included) == 1:
		# this simplex borders exactly one existing cluster
		included.pop().add(simp_idx)
	else:
		# included in 2 or more clusters; merge!
		largest_cluster = max(included, key=len)
		# for now, add the smaller cluster(s) as elements of the largest
		for cluster in included:
			if cluster is not largest_cluster:
				clusters.remove(cluster)
				largest_cluster |= cluster
				largest_cluster.add(frozenset(cluster))
# print(clusters)
print("clusters: ", len(clusters))
def printcluster(c, prefix=""):
	print(prefix+"---{ cluster size: ", end="")
	print(len([x for x in c if not isinstance(x, frozenset)]))
	subclusters = [x for x in c if isinstance(x, frozenset)]
	for sc in subclusters:
		printcluster(sc, prefix=prefix+"|  ")
	print(prefix+" }")
# printcluster(clusters[0])

