#!/usr/bin/python3 -u
from __future__ import division
import argparse
from collections import defaultdict
import functools
import glob
import igraph as ig
import os
import pdb
import random
import re
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

def main(dataset=None, algorithm=None, verbose=False, max_iters1=30000, max_iters2=10000000, write_clusters=False):
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
    dataset_algorithm_params['facebook']['greedy-1'] = dict(verbose=verbose, max_iterations=max_iters1)
    dataset_algorithm_params['facebook']['greedy-2'] = dict(verbose=verbose, max_iterations=max_iters2)
    dataset_algorithm_params['wikivote']['greedy-1'] = dict(verbose=verbose, max_iterations=max_iters1)
    dataset_algorithm_params['wikivote']['greedy-2'] = dict(verbose=verbose, max_iterations=max_iters2)
    dataset_algorithm_params['collab']['greedy-1']   = dict(verbose=verbose, max_iterations=max_iters1)
    dataset_algorithm_params['collab']['greedy-2']   = dict(verbose=verbose, max_iterations=max_iters2)

    graphs = {}
    clusters = defaultdict(dict)
    dataset_algorithm_time = defaultdict(dict)

#    pdb.set_trace()

    for data in dataset_file_name.keys():
        if dataset is not None and data != dataset:
            continue
        graphs[data] = load_tsv_edges(dataset_file_name[data], directed=dataset_is_directed[data])
        for alg, func in algorithm_func.items():
            if algorithm is not None and alg != algorithm:
                continue
            print("Doing {0} {1}".format(data, alg))
            kw = dataset_algorithm_params[data][alg]
            t0 = time.time()
            clusters[data][alg] = func(graphs[data], kw)
            dataset_algorithm_time[data][alg] = time.time() - t0

    for data, graph in graphs.items():
        print("Graph summary for dataset {0}: {1}".format(data, graph.summary()))
        for alg, cluster in clusters[data].items():
            print("Clusters summary for dataset {0}.{1}: {2}".format(data, alg, cluster.summary()))
            print("    modularity: {0}".format(cluster.modularity))
            print("    time: {0}".format(dataset_algorithm_time[data][alg]))
        print("")

    if write_clusters:
        for data in clusters.keys():
            for alg, cluster in clusters[data].items():
                file_name = time.strftime("data/{0}_{1}_%y-%-m-%d_%H-%-M-%S.txt".format(data, alg))
                print("Writing {0}".format(file_name))
                with open(file_name, 'w') as f:
                    for i, c in enumerate(cluster.membership):
                        f.write("{0}\t{1}\n".format(i,c))
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        choices=['facebook','wikivote','collab'],
                        help='Dataset to process',
                        default=None)
    parser.add_argument('-a',
                        choices=['eigenvector', 'walktrap', 'greedy-1', 'greedy-2'],
                        help='Algorithm to run on dataset',
                            default=None)
    parser.add_argument('-v',
                        action='store_true',
                        help='Verbose mode',
                        default=False)
    parser.add_argument('-x1',
                        type=int,
                        help='max iterations to run on first greedy algorithm',
                            default=30000)
    parser.add_argument('-x2',
                        type=int,
                        help='max iterations to run on second greedy algorithm',
                            default=10000000)
    parser.add_argument('-w',
                        action='store_true',
                        help='write created clusters to disk',
                        default=False)
    args = parser.parse_args()

    main(dataset=args.d, algorithm=args.a, verbose=args.v, max_iters1=args.x1,max_iters2=args.x2,write_clusters=args.w)

