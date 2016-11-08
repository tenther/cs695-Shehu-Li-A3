#!/usr/bin/python3
from __future__ import division
import igraph as ig
import glob
import os
import pdb
import re
import functools
from collections import defaultdict
import random
import time

comments_re = re.compile(r'#.*')

class Timer(object):
    def __init__(self):
        self.last_time = 0.0
        self.total_time = 0.0

    def start(self):
        self.last_time = time.time()

    def stop(self):
        self.total_time += time.time() - self.last_time

    def timeit(self, func):
        self.start()
        res = func()
        self.stop()
        return res

    def total(self):
        return self.total_time

class ModularityMaintainer(object):
    def __init__(self, graph, membership):

        # Code based on function igraph_modularity() in https://github.com/igraph/igraph/blob/master/src/community.c
        no_of_comms = max(membership) + 1
        a           = [0.0 for _ in range(no_of_comms)]
        e           = [0.0 for _ in range(no_of_comms)]
        m           = [0.0 for _ in range(no_of_comms)]
        modularity  = 0.0
        edges       = [(edge.source, edge.target) for edge in graph.es]

        for v1, v2 in edges:
            c1 = membership[v1]
            c2 = membership[v2]
            if (c1==c2):
                e[c1] += 2.0
            a[c1] += 1.0
            a[c2] += 1.0

        no_of_edges = len(graph.es)
        if no_of_edges > 0:
            for i in range(no_of_comms):
                tmp = a[i]/2.0/no_of_edges
                m[i] = e[i]/2.0/no_of_edges - tmp*tmp

        modularity = sum(m)
        # make an adjacency list
        adj = defaultdict(set)
        for v1, v2 in edges:
            adj[v1].add(v2)
            adj[v2].add(v1)

        self.graph       = graph
        self.membership  = membership
        self.edges       = edges
        self.no_of_edges = no_of_edges
        self.no_of_comms = no_of_comms
        self.modularity  = modularity
        self.e           = e
        self.a           = a
        self.m           = m
        self.adj         = adj
        self.previous    = None

    def move_community(self, v, new_community):
        a = self.a
        e = self.e
        m = self.m
        no_of_edges = self.no_of_edges
        no_of_comms = self.no_of_comms
        membership  = self.membership
        adj         = self.adj
        modularity  = self.modularity

        if no_of_edges == 0:
            return

        # The e array tracks when both nodes of an edge are
        # in the same community. Remove the effect of those.
        # The a array tracks just the presence of a node
        # in a community. Remove the effect for the node
        # being changed.
        #
        # Likewise, add these values in for the new community.

        c1 = membership[v]

        # Store previous from and to communities, and the prior a,e,m values.
        self.previous = (v, c1, a[c1], e[c1], m[c1], new_community, a[new_community], e[new_community], m[new_community])

        if c1 == new_community:
            raise Exception("Moving node to it's current community ({0} -> {1}) will likely break move_community".format(c1, new_community))

        for v2 in adj[v]:
            c2 = membership[v2]

            if c1 == c2:
                e[c1] -= 2.0
            if c2 == new_community:
                e[new_community] += 2.0
            a[c1] -= 1.0
            a[new_community] += 1.0

        # m array is used to track modularity component
        # of each community. Recalculate these for the
        # affected commununities.
        for i in [c1, new_community]:
            tmp = a[i]/2.0/no_of_edges
            m[i] = e[i]/2.0/no_of_edges - tmp*tmp

        membership[v] = new_community

        # Recalculate modularity
        self.modularity = sum(m)

    def revert(self):
        if not self.previous:
            raise Exception("No history to be reverted")
        v, prev_c1, prev_a1, prev_e1, prev_m1, prev_c2, prev_a2, prev_e2, prev_m2 = self.previous
        self.membership[v] = prev_c1
        self.a[prev_c1] = prev_a1
        self.e[prev_c1] = prev_e1
        self.m[prev_c1] = prev_m1
        self.a[prev_c2] = prev_a2
        self.e[prev_c2] = prev_e2
        self.m[prev_c2] = prev_m2
        self.modularity = sum(self.m)
        self.previous = None

# Load one set of files 
# def load_facebook(data_dir, file_names=None):
#     data_file_path = functools.partial(os.path.join, data_dir)
#     if not file_names:
#         if not data_dir:
#             raise Exception("Must provide data_dir with file_names")
#         file_names = glob.glob(data_file_path("*.circles"))

#     # It seems to be faster to loaded vertices and edges into sets and
#     # create the graph with them all at once, instead of adding them
#     # to the graph as we go. Didn't rigorously check this though.
#     V = set()
#     E = set()

#     for circles_file_name in file_names:
#         # Add each id from a file name to the vertices
#         ego_id = int(circles_file_name.split('/')[-1].split('.')[0])
#         V.add(ego_id)

#         # Add vertices from feat file
#         feat_file_name = os.path.join(data_dir, "{0}.feat".format(ego_id))
#         with open(feat_file_name) as feat_file:
#             for line in feat_file:
#                 fields = line.strip().split()
#                 id = int(fields[0])
#                 V.add(id)

#         # Add edges
#         edge_file_name = os.path.join(data_dir, "{0}.edges".format(ego_id))
#         with open(edge_file_name) as edge_file:
#             for line in edge_file:
#                 v1, v2 = [int(v) for v in line.strip().split()]
#                 V.add(v1)
#                 V.add(v2)
#                 E.add((v1, v2))

#                 # make sure links are symmetrical
#                 E.add((v2, v1))

#                 # Links from primary vertex to others is implicit. E is a set, so we won't have dups.
#                 E.add((ego_id, v1))
#                 E.add((ego_id, v2))

#                 # make sure links are symmetrical
#                 E.add((v1, ego_id))
#                 E.add((v2, ego_id))


#     # It seems igraph will make all vertices a contiguous range, even
#     # if there are gaps (which would cause the vertices and edges to
#     # get out of sync.) So add them in here and warn.
#     for i in range(max(V) + 1):
#         if i not in V:
#             print("Adding missing vertex {0} to V".format(i))
#             V.add(i)
#     g = ig.Graph(n=len(V), edges=list(E))

#     return g

def load_tsv_edges(data_file_name, directed=None):

    V = set()
    E = set()

    with open(data_file_name) as data_file:
        for line in data_file:
            line = comments_re.sub('', line.strip()).strip()
            if line:
                source, target = line.split()
                V.add(source)
                V.add(target)
                E.add((source, target))

    g = ig.Graph(directed=directed)
    g.add_vertices(list(V))
    g.add_edges(list(E))
    return g

def do_greedy_clustering(graph, tries=100, max_iterations=5000, min_delta=0.0, verbose=False):
    best_vc = None
    for _ in range(tries):
        vc = greedy_clustering(graph, max_iterations, min_delta, verbose)
        if not best_vc or vc.modularity > best_vc.modularity:
            best_vc = vc
    return vc

# def greedy_clustering2(graph, max_iterations=5000, min_delta=0.0, verbose=False):
#     VC = ig.VertexClustering
#     # start with each vertex in its own commuanity
#     vc = VC(graph, [i for i, _ in enumerate(graph.vs)])

#     mm = ModularityMaintainer(graph, vc.membership)

#     vc_timer = Timer()
#     mm_timer = Timer()

#     partition_vertexes = defaultdict(set)
#     for i, p in enumerate(vc.membership):
#         partition_vertexes[p].add(i)

#     partition_counts = dict()
#     for i, s in partition_vertexes.items():
#         partition_counts[i] = len(s)

#     for iteration in range(max_iterations):
#         # Copy membership, just to avoid odd errors. May not be necessary.
#         membership         = list(vc.membership)
#         selected_vertex    = random.randint(0, len(membership) - 1)
#         selected_community = membership[selected_vertex]

#         new_communities    = [c for c in partition_counts.keys() if c != selected_community]
#         random.shuffle(new_communities)
#         found_one          = False
#         found_community    = 0
#         for new_community in new_communities:
#             if new_community == selected_community:
#                 continue
#             mm_timer.timeit(lambda: mm.move_community(selected_vertex, new_community))
#             membership[selected_vertex] = new_community
#             new_vc = vc_timer.timeit(lambda:  VC(graph,membership))
#             delta = new_vc.modularity - vc.modularity
#             if delta > min_delta:
#                 found_community = new_community
#                 found_one = True
#                 break
#             else:
#                 mm.revert()
#         if found_one:
#             partition_counts[vc.membership[selected_vertex]] -= 1
#             if not partition_counts[vc.membership[selected_vertex]]:
#                 del(partition_counts[vc.membership[selected_vertex]])
#             partition_counts[found_community] += 1
#             vc = new_vc
#             if verbose:
#                 print("Greedy clustering. iteration={0} modularity:={1} delta={2}. vc time={3}. mm modularity={4}. mm time={5}".format(iteration, 
#                                                                                                                                        vc.modularity, 
#                                                                                                                                        delta, 
#                                                                                                                                        vc_timer.total(),
#                                                                                                                                        mm.modularity, 
#                                                                                                                                        mm_timer.total()))
#     return vc

# Make communities indexed from 0
def normalize_membership(membership):
    communities = set(membership)
    old_to_new = dict(zip(sorted(list(communities)),
                          range(len(communities))))
    return [old_to_new[x] for x in membership]
    

def greedy_clustering(graph, max_iterations=5000, min_delta=0.0, verbose=False):
    VC = ig.VertexClustering
    # start with each vertex in its own commuanity
    mm = ModularityMaintainer(graph, [i for i, _ in enumerate(graph.vs)])

    partition_vertexes = defaultdict(set)
    for i, p in enumerate(mm.membership):
        partition_vertexes[p].add(i)

    partition_counts = dict()
    for i, s in partition_vertexes.items():
        partition_counts[i] = len(s)

    previous_modularity = mm.modularity
    for iteration in range(max_iterations):
        selected_vertex    = random.randint(0, len(graph.vs) - 1)
        selected_community = mm.membership[selected_vertex]
        new_communities    = [c for c in partition_counts.keys() if c != selected_community]
        random.shuffle(new_communities)
        found_one          = False
        found_community    = 0
        best_modularity    = float("-inf")
        best_community     = 0
        for new_community in new_communities:
            if new_community == selected_community:
                continue
            mm.move_community(selected_vertex, new_community)
            delta = mm.modularity - previous_modularity
            if delta > min_delta:
                found_community = new_community
                found_one = True
                if mm.modularity > best_modularity:
                    best_community = new_community
                    best_modularity = mm.modularity
            mm.revert()
        if found_one:
            partition_counts[selected_community] -= 1
            if not partition_counts[selected_community]:
                del(partition_counts[selected_community])
            partition_counts[best_community] += 1
            mm.move_community(selected_vertex, best_community)
            if verbose:
                print("Greedy clustering. iteration={0} modularity:={1} delta={2}.".format(iteration, mm.modularity, mm.modularity - previous_modularity))
            previous_modularity = mm.modularity
    
    print("Finished greedy_clustering. Clustered {0} communities into {1}.".format(len(mm.membership), len(set(mm.membership))))
    return VC(graph, list(normalize_membership(mm.membership)))

def greedy_clustering2(graph, max_iterations=5000, min_delta=0.0, verbose=False):
    VC = ig.VertexClustering
    # start with each vertex in its own commuanity
    mm = ModularityMaintainer(graph, [i for i, _ in enumerate(graph.vs)])

    partition_vertexes = defaultdict(set)
    for i, p in enumerate(mm.membership):
        partition_vertexes[p].add(i)

    partition_counts = dict()
    for i, s in partition_vertexes.items():
        partition_counts[i] = len(s)

    previous_modularity = mm.modularity
    for iteration in range(max_iterations):
        selected_vertex    = random.randint(0, len(graph.vs) - 1)
        selected_community = mm.membership[selected_vertex]
        new_communities    = [c for c in partition_counts.keys() if c != selected_community]
        random.shuffle(new_communities)
        new_community = new_communities[0]
        if new_community != selected_community:
            mm.move_community(selected_vertex, new_community)
            delta = mm.modularity - previous_modularity
            if delta > min_delta:
                found_one = True
                partition_counts[selected_community] -= 1
                if not partition_counts[selected_community]:
                    del(partition_counts[selected_community])
                partition_counts[new_community] += 1
                if verbose:
                    print("Greedy clustering 2. iteration={0} modularity:={1} delta={2}.".format(iteration, mm.modularity, mm.modularity - previous_modularity))
                previous_modularity = mm.modularity
            else:
                mm.revert()
    
    print("Finished greedy_clustering2. Clustered {0} communities into {1}.".format(len(mm.membership), len(set(mm.membership))))
    return VC(graph, list(normalize_membership(mm.membership)))

def main():
    dataset_file_name = {
        'facebook': os.path.join(*"data/egonets-Facebook/facebook_combined.txt".split("/")),
        'wikivote': os.path.join(*"data/wiki-Vote/wiki-Vote.txt".split("/")),
        'collab':   os.path.join(*"data/ca-GrQc/ca-GrQc.txt".split("/")),
    }

    dataset_is_directed = {
        'facebook':  False,
        'wikivote':  True,
        'collab':    True,
    }

    algorithm_func = {
        'eigenvector': lambda g, kw: g.community_leading_eigenvector(),
        'walktrap':    lambda g, kw: g.community_walktrap().as_clustering(),
        'greedy-1':    lambda g, kw: greedy_clustering(g, **kw),
        'greedy-2':    lambda g, kw: greedy_clustering2(g, **kw),
    }

    dataset_algorithm_params = defaultdict(lambda: defaultdict(dict))
    dataset_algorithm_params['facebook']['greedy-1'] = dict(verbose=True, max_iterations=1000)
    dataset_algorithm_params['facebook']['greedy-2'] = dict(verbose=True, max_iterations=100000)
    dataset_algorithm_params['wikivote']['greedy-1'] = dict(verbose=True, max_iterations=1000)
    dataset_algorithm_params['wikivote']['greedy-2'] = dict(verbose=True, max_iterations=100000)
    dataset_algorithm_params['collab']['greedy-1']   = dict(verbose=True, max_iterations=1000)
    dataset_algorithm_params['collab']['greedy-2']   = dict(verbose=True, max_iterations=100000)

    datasets_to_skip = ['wikivote', 'collab']

    graphs = {}
    clusters = defaultdict(dict)
    dataset_algorithm_time = defaultdict(dict)

#    pdb.set_trace()

    for dataset in dataset_file_name.keys():
        if dataset in datasets_to_skip:
            continue
        graphs[dataset] = load_tsv_edges(dataset_file_name[dataset], directed=dataset_is_directed[dataset])
        for algorithm, func in algorithm_func.items():
            print("Doing {0} {1}".format(dataset, algorithm))
            kw = dataset_algorithm_params[dataset][algorithm]
            t0 = time.time()
            clusters[dataset][algorithm] = func(graphs[dataset], kw)
            dataset_algorithm_time[dataset][algorithm] = time.time() - t0

    for dataset, graph in graphs.items():
        print("Graph summary for dataset {0}: {1}".format(dataset, graph.summary()))
        for algorithm, cluster in clusters[dataset].items():
            print("Clusters summary for dataset {0}.{1}: {2}".format(dataset, algorithm, cluster.summary()))
            print("    modularity: {0}".format(cluster.modularity))
            print("    time: {0}".format(dataset_algorithm_time[dataset][algorithm]))
        print("")

    return

if __name__ == '__main__':
    main()



