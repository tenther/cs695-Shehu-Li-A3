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
import math

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
                source = 'v' + source
                target = 'v' + target
                V.add(source)
                V.add(target)
                E.add((source, target))

    g = ig.Graph(directed=directed)
    g.add_vertices(sorted(list(V)))
    g.add_edges(list(E))
#    g.vs["name"] = [str(v.index) for v in g.vs]
    return g

def export_gephi_csv(graph, membership, nodes_filename="nodes", edges_filename="edges"):
    g = graph
    atts = list(g.vs.attribute_names())
    atts.remove("name")
    with open(nodes_filename + ".csv", 'w') as f:
        line = 'Id,Label,Community' + ','.join(map(str, atts))
        f.write(line + "\n")
        for v in g.vs:
            line = "{0},{1},{2}".format(str(v.index),v["name"],membership[v.index])
            #We really only care about community, so I guess we don't necessarily need this
            if len(atts) > 0:
                temp = [str(v[att]) for att in atts]
                line += "," + ','.join(map(str,temp))
            f.write(line + "\n")
            
    with open(edges_filename + ".csv", 'w') as f:
        f.write("source,target,type\n")
        temp = [e.tuple for e in g.es]
        for s,t in temp:
            line = "{0},{1},{2}".format(str(s),str(t),"undirected")
            f.write(line + "\n")

def do_greedy_clustering(graph, func, tries=100, max_iterations=5000, min_delta=0.0, max_no_progress=500, verbose=False):
    best_vc = None
    for _ in range(tries):
        vc = func(graph, max_iterations, min_delta, verbose, max_no_progress)
        if not best_vc or vc.modularity > best_vc.modularity:
            best_vc = vc
    return best_vc

# Make communities indexed from 0
def normalize_membership(membership):
    communities = set(membership)
    old_to_new = dict(zip(sorted(list(communities)),
                          range(len(communities))))
    return [old_to_new[x] for x in membership]

def greedy_clustering(graph, max_iterations=5000, min_delta=0.0, verbose=False, max_no_progress=0):
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
            # commented this out to see if leaving the total number of partitions the same as the original makes a difference
            # if not partition_counts[selected_community]:
            #     del(partition_counts[selected_community])
            partition_counts[best_community] += 1
            mm.move_community(selected_vertex, best_community)
            if verbose:
                print("Greedy clustering. iteration={0} modularity:={1} delta={2}.".format(iteration, mm.modularity, mm.modularity - previous_modularity))
            previous_modularity = mm.modularity
    
    if verbose:
        print("Finished greedy_clustering. Clustered {0} communities into {1}.".format(len(mm.membership), len(set(mm.membership))))
    return VC(graph, normalize_membership(mm.membership))

def greedy_clustering2(graph, max_iterations=5000, min_delta=0.0, verbose=False, max_no_progress=500):
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
    no_progress_counter = 0
    for iteration in range(max_iterations):
        if no_progress_counter == max_no_progress:
            break
        selected_vertex    = random.randint(0, len(graph.vs) - 1)
        selected_community = mm.membership[selected_vertex]
        new_communities    = [c for c in partition_counts.keys() if c != selected_community]
        random.shuffle(new_communities)
        new_community = new_communities[0]
        mm.move_community(selected_vertex, new_community)
        delta = mm.modularity - previous_modularity
        if delta > min_delta:
            no_progress_counter = 0
            partition_counts[selected_community] -= 1
            if not partition_counts[selected_community]:
                del(partition_counts[selected_community])
            partition_counts[new_community] += 1
            if verbose:
                print("Greedy clustering 2. iteration={0} modularity:={1} delta={2}.".format(iteration, mm.modularity, mm.modularity - previous_modularity))
            previous_modularity = mm.modularity
        else:
            no_progress_counter += 1
            mm.revert()
    
    if verbose:
        print("Finished greedy_clustering2. Clustered {0} communities into {1}.".format(len(mm.membership), len(set(mm.membership))))
    return VC(graph, normalize_membership(mm.membership))

def mc_clustering(graph, alpha, max_iterations=5000, min_delta=0.0, verbose=False, max_no_progress=500):
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
    no_progress_counter = 0
    accept_change = False
    for iteration in range(max_iterations):
        if no_progress_counter == max_no_progress:
            break
        selected_vertex    = random.randint(0, len(graph.vs) - 1)
        selected_community = mm.membership[selected_vertex]
        new_communities    = [c for c in partition_counts.keys() if c != selected_community]
        random.shuffle(new_communities)
        new_community = new_communities[0]
        mm.move_community(selected_vertex, new_community)
        delta = mm.modularity - previous_modularity
        
        if delta > min_delta:
            accept_change = True
        else:
            p = min(math.exp(delta / alpha),1) #Have to think of how e^x works to continue this
            r = random.uniform(0,1)
            if p <= r:
                accept_change = True
            else:
                accept_change = False
        
        if accept_change:
            no_progress_counter = 0
            partition_counts[selected_community] -= 1
            if not partition_counts[selected_community]:
                del(partition_counts[selected_community])
            partition_counts[new_community] += 1
            if verbose:
                print("Greedy clustering 2. iteration={0} modularity:={1} delta={2}.".format(iteration, mm.modularity, mm.modularity - previous_modularity))
            previous_modularity = mm.modularity
        else:
            no_progress_counter += 1
            mm.revert()
    
    if verbose:
        print("Finished mc_clustering. Clustered {0} communities into {1}.".format(len(mm.membership), len(set(mm.membership))))
    return VC(graph, normalize_membership(mm.membership))

valid_datasets = ['facebook','wikivote','collab', 'test', 'karate',]

def main(dataset=None, algorithm=None, verbose=False, max_iters1=30000, max_iters2=10000000, write_clusters=False, tries=1, export=False):
    dataset_file_name = {
        'facebook': os.path.join(*"data/egonets-Facebook/facebook_combined.txt".split("/")),
        'wikivote': os.path.join(*"data/wiki-Vote/wiki-Vote.txt".split("/")),
        'collab':   os.path.join(*"data/ca-GrQc/ca-GrQc.txt".split("/")),
        'karate':   os.path.join(*"data/karate/karate.txt".split("/")),
        'test':     os.path.join(*"data/test/old_test.txt".split("/")),
    }

    dataset_is_directed = {
        'facebook':  False,
        'wikivote':  False,
        'collab':    False,
        'karate':    False,
        'test':      False,
    }

    algorithm_func = {
        'betweenness': lambda g, kw: g.community_edge_betweenness().as_clustering(),
        'eigenvector': lambda g, kw: g.community_leading_eigenvector(),
        'walktrap':    lambda g, kw: g.community_walktrap().as_clustering(),
        'greedy-1':    lambda g, kw: do_greedy_clustering(g, greedy_clustering, **kw),
        'greedy-2':    lambda g, kw: do_greedy_clustering(g, greedy_clustering2, **kw),
#        'greedy-1':    lambda g, kw: greedy_clustering(g, **kw),
#        'greedy-2':    lambda g, kw: greedy_clustering2(g, **kw),
    }
    
    dataset_algorithm_params = defaultdict(lambda: defaultdict(dict))
    for data in valid_datasets:
        dataset_algorithm_params[dataset]['greedy-1'] = dict(verbose=verbose, max_iterations=max_iters1,tries=tries)
        dataset_algorithm_params[dataset]['greedy-2'] = dict(verbose=verbose, max_iterations=max_iters2,tries=tries)
        
    graphs = {}
    clusters = defaultdict(dict)
    dataset_algorithm_time = defaultdict(dict)

    for data in valid_datasets:
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
#            pdb.set_trace()

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
                params = ""
                if alg == 'greedy-1':
                    params = '{0}_'.format(max_iters1)
                if alg == 'greedy-2':
                    params = '{0}_'.format(max_iters2)
                file_name = time.strftime("data/community_{0}_{1}_{2}%y-%m-%d_%H-%M-%S.txt".format(data, alg, params))
                print("Writing {0}".format(file_name))
                with open(file_name, 'w') as f:
                    for i, c in enumerate(cluster.membership):
                        f.write("{0}\t{1}\n".format(i,c))
    
    if export:
        node_filename = "nodes_{0}_{1}".format(dataset, algorithm)
        edge_filename = "edges_{0}_{1}".format(dataset, algorithm)
        export_gephi_csv(clusters[data][alg].graph,clusters[data][alg].membership,node_filename,edge_filename)
        print("Exporting Gephi spreadsheet csv files: {0}.csv, {1}.csv".format(node_filename,edge_filename))
    
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        choices=valid_datasets,
                        help='Dataset to process',
                        default=None)
    parser.add_argument('-a',
                        choices=['eigenvector', 'walktrap', 'greedy-1', 'greedy-2', 'betweenness',],
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
    parser.add_argument('-t',
                        type=int,
                        help='number of tries to run with greedy algorithm',
                            default=1)
    parser.add_argument('-x2',
                        type=int,
                        help='max iterations to run on second greedy algorithm',
                            default=10000000)
    parser.add_argument('-w',
                        action='store_true',
                        help='write created clusters to disk',
                        default=False)
    parser.add_argument('-e',
                        action='store_true',
                        help='export Gephi spreadsheet csv file')
    args = parser.parse_args()

    main(dataset=args.d, algorithm=args.a, verbose=args.v, max_iters1=args.x1,max_iters2=args.x2,write_clusters=args.w,tries=args.t,export=args.e)

