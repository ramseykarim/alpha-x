import numpy as np
import alphax_utils as utils
from AlphaCluster import AlphaCluster

"""
v4 alpha x unit testing module
Created (along with v4): Oct 13, 2019
"""
__author__ = "Ramsey Karim"

def get_test_data():
    # Some really basic test data
    srcfile = "../../PyAlpha_drafting/test_data/Ssets/s1.txt"
    return np.genfromtxt(srcfile)[::500]

# Get test data set
p = get_test_data()
# Print shape of test data set
print(p.shape)

# Boilerplate Delaunay setup
tri = utils.Delaunay(p)
ndim = p.shape[1]
triangle_coords = p[tri.simplices, :]
cr_array, vol_array = utils.cayley_menger_vr(triangle_coords)
simp_sorted_idxs = np.argsort(cr_array)
# Lookup table for every single simplex
# Will contain some sort of quick mapping to clusters
simp_lookup = [None]*tri.simplices.shape[0]
# See v4idea.readme for the algorithm description
# Staging area may have multiple clusters, will grow and shrink.
staging_area = []
# Cluster area will have multiple clusters while growing, but should end with
# just one, the root.
cluster_area = []
# Loop through simplices in sorted order
for simp_idx in simp_sorted_idxs:
    print(simp_idx)
    # Get all ndim+1 neighbors of this simplex
    included_neighbors = [n for n in tri.neighbors[simp_idx] if simp_lookup[n] is not None]
    cr = cr_array[simp_idx]
    if not included_neighbors:
        # Isolated simplex
        proposed_cluster = set()
        proposed_cluster.add(simp_idx)
        simp_lookup[simp_idx] = proposed_cluster
        staging_area.append(proposed_cluster)
    elif len(included_neighbors) == 1:
        # Exactly one border simplex
        # This is the neighbor index
        neighbor_idx = included_neighbors.pop()
        # Look it up in the lookup list
        cluster_reference = simp_lookup[neighbor_idx]
        # Now we must dereference the lookup result; it may be a proposed
        # cluster, a real cluster, or another simplex (which points to one
        # of those other two)
        # Should wrap this up into a function resolve_(lookup_)reference
        if isinstance(cluster_reference, int):
            # Points to another simplex, which should have cluster reference
            simp_lookup[cluster_reference].add(simp_idx)
            # Add reference index to lookup table for this simplex
            simp_lookup[simp_idx] = cluster_reference
        else:
            # Should be either an AlphaCluster or proposed cluster set
            cluster_reference.add(simp_idx)
            # Since neighbor was the cluster's reference index, use that for
            # lookup list
            simp_lookup[simp_idx] = neighbor_idx
    else:
        # Included in 2 or more clusters
        # Dereference all the neighboring indices (get cluster ref indices)
        neighboring_cluster_idxs = []
        for neighbor_idx in included_neighbors:
            if isinstance(simp_lookup[neighbor_idx], int):
                # Chain ref to another simplex
                neighboring_cluster_idxs.append(simp_lookup[neighbor_idx])
                print('isint: ', simp_lookup[neighbor_idx])
            else:
                neighboring_cluster_idxs.append(neighbor_idx)
                print('iscluster: ', neighbor_idx)
        # Dereference to the actual clusters (retain order)
        neighboring_clusters = [simp_lookup[n] for n in neighboring_cluster_idxs]
        print("nc", neighboring_clusters)
        print('---------------')
        largest_cluster_i = max(range(len(neighboring_clusters)), key=lambda i: len(neighboring_clusters[i]))
        largest_cluster_idx = neighboring_cluster_idxs.pop(largest_cluster_i)
        largest_cluster = neighboring_clusters.pop(largest_cluster_i)
        for c in neighboring_clusters:
            largest_cluster |= c
            for s in c:
                simp_lookup[s] = largest_cluster_idx
            staging_area.remove(c)
        simp_lookup[simp_idx] = largest_cluster_idx
"""
NEED TO DEBUG
probably some issue about simplices bordering the edge
neighboring_clusters contains some ints, should only contain sets
(see print statement, result of running this) (Oct 13, 2019)
"""
