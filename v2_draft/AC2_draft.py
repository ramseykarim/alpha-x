# Draft file for AlphaCluster
import utils2_draft as utils


class AlphaCluster:
    """
    Each cluster is represented by an AlphaCluster object.

    The alpha at each level designates the larger end of the step.
    It indicates, "in this step, all triangles are less than OR EQUAL TO this alpha until we check again"
    The smaller end is implicit.
    This means that alpha shapes start at an alpha level explicitly stated at the beginning of their
    alpha range, but end implicitly at the alpha level one step past their last alpha range element.
    """

    def __init__(self, cluster_elements, alpha_level=None, null_simplices=[]):
        # Cluster_elements is a set
        # Alpha level should be the alpha level at which the last cluster broke (not included in the last cluster)

        # A full list is an awful idea; we should just query the elements for a specific level if we need it
        # This used to be self.cluster_elements = [cluster_elements] but that's dumb, so we're changing it
        # to only store the initial list. We know the initial list is coherent at the first step
        self.cluster_elements = frozenset(cluster_elements) # never needs to change!

        # This is the smallest simplex that is still part of the cluster when it ends as a leaf
        # Usage of this simplex should eliminate ambiguity in tracing this cluster in post-process
        self.tracer_simplex = None

        # Sure, I guess. When would we ever use this? Why not just query it?
        # Consider deleting. Can be done in post-process
        self.volume_range = [sum([x.volume for x in self.cluster_elements])]

        # Again, when do we need this? Post-process.
        self.member_range = [len(self.cluster_elements)]

        # This is necessary; building blocks for tracing out the gap hierarchy
        self.null_simplices = [null_simplices]

        # This is also necessary; it's the child list! All the recursion passes through here.
        self.subclusters = []

        # What do we need to do next? Alpha level!
        # We will start the alpha level AT THE LARGEST CR since we are doing LE(<=) instead of LT(<)
        # That means we don't have to do a backtrack step at the very beginning, we can just start with max(CR)
        if alpha_level is None:
            self.alpha_range = [max(self.cluster_elements).circumradius]
        else:
            self.alpha_range = [alpha_level]

        if not utils.QUIET:
            print(utils.SPACE+"<branch init size(%d)>" % len(self.cluster_elements))
            utils.SPACE += "|  "

        # Private convenience variables for later
        self._dropped_simplices = None
        self._next_alpha = None
        self._coherent = True

        # Go through and do the entire cluster traversal.
        self.exhaust_cluster()
        # Everything should be clustered by now. Then gaps!
        # TODO
        if not utils.QUIET:
            utils.SPACE = utils.SPACE[:-3]
            print(utils.SPACE + "</branch persist=%d>" % len(self.alpha_range))

    def exhaust_cluster(self):
        # Here's where we begin:
        # -- We have self.cluster_elements, which is an unordered set of all initial elements
        # -- We know self.cluster_elements is cohesive in the first step, since we either
        # -- -- a) shattered the old cluster last time and ended up with this thing or
        # -- -- b) are starting at the largest CR
        # -- We have the first element of self.alpha_range, from which we take our first step
        # This means: we do not need need to traverse the first step, just take down some notes

        self._next_alpha = self.alpha_range[-1]

        # Do nothing with the null_simplices list; it already has a list for the first element
        # If that list is empty (by default), then
        # -- the implicit statement here is: the entire cluster is present, the null space has no members yet
        # If the list is not empty, this cluster may have inherited interior gaps from a parent

        # Skip anything with member_range or volume_range; we don't know if we even need to manage these

        # Start creating variables
        remaining_simplices = set(self.cluster_elements)

        # We begin the loop and take the SECOND STEP
        # FIXME this probably is not the optimal criterion
        while self._coherent:
            # We begin a NEW STEP
            # First, adjust the alpha level
            self._next_alpha *= utils.ALPHA_STEP

            # Create a set for triangles dropped this round
            self._dropped_simplices = set()

            # Drop everything GREATER THAN the current alpha
            # This guarantees everything left is LE(<=) the alpha level
            largest_simplex = max(remaining_simplices) # , lambda x: x.circumradius ?
            while len(remaining_simplices) >= utils.ORPHAN_TOLERANCE and largest_simplex.circumradius > self._next_alpha:
                remaining_simplices.remove(largest_simplex)
                self._dropped_simplices.add(largest_simplex)
                largest_simplex = max(remaining_simplices) # , lambda x: x.circumradius ?

            # Here, remaining_simplices should accurately reflect the survivors of this step
            # self._Dropped_simplices should reflect everything that belongs to a null space of some kind
            # We will return to self._dropped_simplices at the end, since we don't know if this cluster is coherent

            # We need to find out if this cluster is coherent
            # What was dropped?
            # -o We did not drop anything in that while loop
            # -- -> We skip the traversal                                               CONTINUE CASE 1
            # -o We dropped something in that while loop
            # -- -> Are there enough remaining elements?
            # -- -- -o No
            # -- -- -- -> We end the cluster; it is a leaf                              END CASE 1
            # -- -- -o Yes
            # -- -- -- -> We traverse; what happens?
            # -- -- -- -- -o The cluster breaks
            # -- -- -- -- -- -> Filter subclusters for size; how many?
            # -- -- -- -- -- -- -o There are several significant subclusters
            # -- -- -- -- -- -- -- -> Create children!                                  spawn children
            # -- -- -- -- -- -- -- -> Also continue with largest subcluster as main     CONTINUE CASE 1
            # -- -- -- -- -- -- -o There is one significant subcluster
            # -- -- -- -- -- -- -- -> Continue from that subcluster as main             CONTINUE CASE 1
            # -- -- -- -- -- -- -o There are no significant children
            # -- -- -- -- -- -- -- -> We end the cluster; it is a leaf                  END CASE 1
            # -- -- -- -- -o The cluster remains cohesive
            # -- -- -- -- -- -> Continue from the cluster as main                       CONTINUE CASE 1
            # ####

            # We now follow the decision tree
            # Did we drop anything?
            if not self._dropped_simplices:
                # We did not drop anything, just continue
                self.continue_case()
            else:
                # We dropped something! What's left?
                if len(remaining_simplices) < utils.ORPHAN_TOLERANCE:
                    # Not enough left for traversal! End it.
                    # We will do this more efficiently by putting cluster list into its own list
                    # It will fail a length check, we already know that, and it can piggyback onto the logic for "no significant children"
                    cluster_list = [remaining_simplices]
                else:
                    # There is enough left for traversal
                    # Traverse! Assume traverse(remaining_simplices.copy()) just returns a list of FROZENsets of simplices
                    cluster_list = utils.traverse(remaining_simplices.copy())  # This means traverse() can do whatever it wants with its argument
                # Filter cluster_list by ORPHAN_TOLERANCE
                negligible_clusters = [cluster for cluster in cluster_list if len(cluster) < utils.ORPHAN_TOLERANCE]
                for cluster in negligible_clusters:
                    cluster_list.remove(cluster)
                    # These are definitely within the null space. There's literally no way these are in gaps.
                    # But we add them to self._dropped_simplices anyway because the infrastructure to handle them is already there
                    remaining_simplices -= cluster
                    self._dropped_simplices |= cluster
                # Don't need to keep all these sets around
                del negligible_clusters
                # What's left in the cluster_list?
                if len(cluster_list) >= 1:
                    # The main cluster continues, with or without children
                    if len(cluster_list) > 1:
                        # There is at LEAST one child
                        remaining_simplices = self.spawn_subclusters(cluster_list)
                    self.continue_case()
                else:
                    self.end_case()
            # At this point, unless we're about to end, we're ready to continue

    """
    There are TWO cases following traversal: continue and end
    Continue:
    -- We have a subcluster set of simplices that are guaranteed cohesive at the current alpha level
    -- We need to record the alpha level for that step, since it was successful
    -- We need to record the dropped simplices: we have already divided up dropped simplices among children (TODO)
    End:
    -- We should not append to alpha_range or null_simplices
    -- We should set the private variables back to NULL
    """

    def spawn_subclusters(self, subcluster_list):
        # This initiates the subcluster initialization (or at least pushes to stack)
        # Also decides which cluster is the main cluster and returns it
        # Remember we need to distribute the dropped_simplices
        # For distribution, send all non-cluster-related null simplices back to the main cluster's list (self._dropped_simplices)

        # NOTE: main cluster is just the largest cluster by NUMBER OF SIMPLICES
        # could edit to be volume, number of points, etc
        main_cluster_set = max(subcluster_list, key=lambda x: len(x))
        subcluster_list.remove(main_cluster_set)
        subcluster_dropped_simplices = {c: [] for c in subcluster_list}
        # Loop through a copy so we can modify the original
        for simplex in self._dropped_simplices.copy():
            destination = utils.identify_null_simplex(simplex, subcluster_list)
            if destination:
                self._dropped_simplices.remove(simplex)
                subcluster_dropped_simplices[destination].append(simplex)
        for cluster_set in subcluster_list:
            # RECURSION_STACK should only contain dictionaries with these same keywords,
            # which are just the argument names of AlphaCluster.__init__()
            # Push to the stack, first-in-last-out
            utils.RECURSION_STACK.append({
                    utils.K_PARENT: self,
                    utils.K_CLUSTER_ELEMENTS: cluster_set,
                    utils.K_ALPHA_LEVEL: self._next_alpha,
                    utils.K_NULL_SIMPLICES: subcluster_dropped_simplices[cluster_set]
                })
            # We must not modify the elements of subcluster_list after this.
        # Return the main cluster! self._dropped_simplices is ready to go
        return set(main_cluster_set)

    def add_branch(self, child):
        self.subclusters.append(child)

    def end_case(self):
        # This doesn't do much but it signifies the end of this cluster
        # The smallest simplex in self._dropped_simplices is the tracer simplex now;
        # It was part of the cluster at last alpha, so it's safe to use
        self.tracer_simplex = min(self._dropped_simplices)
        self._next_alpha = None
        self._dropped_simplices = None
        self._coherent = False

    def continue_case(self):
        # This should prepare the exhaust_cluster loop to go on to the next step
        # The self._next_alpha is safe to append, as is self._dropped_simplices
        self.alpha_range.append(self._next_alpha)
        self.null_simplices.append(self._dropped_simplices)
        utils.KEY.treeIndex.append_cluster(self._next_alpha, self)


