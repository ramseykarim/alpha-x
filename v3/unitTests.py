import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import alphax_utils as utils
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform

def get_test_data():
	srcfile = "../../PyAlpha_drafting/test_data/Ssets/s1.txt"
	# srcfile = "../../PyAlpha_drafting/test_data/filament5500_sampleNH2_betah1.80.dat"
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

def test_basic_structure():
	"""
	Simple version that uses nested sets to outline concept
	"""
	p = get_test_data()
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

def test_AlphaCluster():
	"""
	Testing ground for structure of cluster code
	Uses AlphaCluster objects
	"""
	import AlphaCluster as alphac
	p = get_test_data()
	tri = Delaunay(p)
	ndim = p.shape[1]
	triangle_coords = p[tri.simplices, :]
	cr_array, vol_array = utils.cayley_menger_vr(triangle_coords)
	simp_sorted_idxs = np.argsort(cr_array)
	simp_lookup = [None]*tri.simplices.shape[0]
	for simp_idx in simp_sorted_idxs:
		neighbors = [simp_lookup[n_idx] for n_idx in tri.neighbors[simp_idx]]
		included = set(n.root for n in neighbors if n is not None)
		cr = cr_array[simp_idx]
		if len(included) == 0:
			# simplex is isolated
			assigned_cluster = alphac.AlphaCluster(simp_idx, cr)
		elif len(included) == 1:
			# simplex borders exactly one existing cluster
			assigned_cluster = included.pop()
			assigned_cluster.add(simp_idx, cr)
		else:
			# included in 2 or more clusters
			assigned_cluster = alphac.AlphaCluster(simp_idx, cr, *included)
		simp_lookup[simp_idx] = assigned_cluster

	clusters = set(simp_lookup)
	root = simp_lookup[0].root
	root.freeze(root)
	all_root = {x.root for x in simp_lookup}
	utils.dendrogram(root)
	return

	def census(c, s):
		s.add(c)
		for sc in c.children:
			census(sc, s)

	init_clusters = set()
	census(root, init_clusters)
	print()

	assert root is next(iter(all_root))
	print("roots", len(all_root))
	print("clusters: ", len(clusters))
	def printcluster(c, prefix=""):
		print(prefix+"---{ cluster size: ", end="")
		print(len(c))
		for sc in c.children:
			printcluster(sc, prefix=prefix+"|  ")
		print(prefix+" }")
	# printcluster(clusters[0])
	root.collapse()

	remaining_clusters = set()
	census(root, remaining_clusters)
	print()
	print("initial clusters", len(init_clusters))
	print("remaining clusters", len(remaining_clusters))
	# lsi = [len(x.members) for x in init_clusters]
	lsr = [len(x.members) for x in remaining_clusters]
	plt.plot(lsr, '.')
	plt.show()
	return

test_AlphaCluster()
