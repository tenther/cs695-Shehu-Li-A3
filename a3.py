#!/usr/bin/python3 -u
from __future__ import division
import argparse
from collections import defaultdict, namedtuple
import functools
import glob
import igraph as ig
import os
import pdb
import random
import re
import time
import math
import cProfile
import copy

VC = ig.VertexClustering

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

    def copy(self):
        other             = copy.copy(self)
        other.membership  = self.membership.copy()
        other.e           = self.e.copy()
        other.a           = self.a.copy()
        other.m           = self.m.copy()
        other.previous    = copy.deepcopy(self.previous)
        return other

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
    return g

def export_gephi_csv(graph, membership, nodes_filename="nodes", edges_filename="edges"):
    g = graph
    atts = list(g.vs.attribute_names())
    atts.remove("name")
    with open(nodes_filename + ".csv", 'w') as f:
        line = 'Id,Label,Community' + (atts and ',' or '') + ','.join(map(str, atts))
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

def do_greedy_clustering(graph, 
                         func, 
                         tries=100, 
                         max_iterations=5000, 
                         min_delta=0.0, 
                         max_no_progress=500, 
                         verbose=False,
                         alpha=None):

    best_vc = None
    for _ in range(tries):
        vc = func(graph, max_iterations, min_delta, verbose, max_no_progress, alpha)
        if not best_vc or vc.modularity > best_vc.modularity:
            best_vc = vc
    return best_vc

# Make communities indexed from 0
def normalize_membership(membership):
    communities = set(membership)
    old_to_new = dict(zip(sorted(list(communities)),
                          range(len(communities))))
    return [old_to_new[x] for x in membership]

def generate_random_membership(n, max_communities=float('inf')):
    membership = []
    if max_communities > n:
        max_communities = n
    for i in range(n):
        membership.append(int(random.random() * max_communities))
    return normalize_membership(membership)

def greedy_clustering(graph, max_iterations=5000, min_delta=0.0, verbose=False, max_no_progress=0, alpha=0.0):
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

def greedy_clustering2(graph, max_iterations=5000, min_delta=0.0, verbose=False, max_no_progress=500, alpha=0.0):
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
    num_vertices = len(graph.vs)
    empty_partitions = []
    
    for iteration in range(max_iterations):
        if no_progress_counter == max_no_progress:
            break
        selected_vertex    = int(random.random() * num_vertices)
        selected_community = mm.membership[selected_vertex]
        new_communities    = [c for c in partition_counts.keys() if c != selected_community]
        # Pick random index 0 to num_vertices, or 0 to num_vertices - 1 if empty_partitions is empty
        add_one            = len(empty_partitions) != 0
        community_index    = int(random.random() * (len(new_communities)+int(add_one)))
        
        if community_index == len(new_communities):
            new_community = empty_partitions.pop()
            partition_counts[new_community] = 0
        else:
            new_community = new_communities[community_index]
            
        mm.move_community(selected_vertex, new_community)
        delta = mm.modularity - previous_modularity
        if delta > min_delta:
            no_progress_counter = 0
            partition_counts[selected_community] -= 1
            if not partition_counts[selected_community]:
                del(partition_counts[selected_community])
                empty_partitions.append(selected_community)
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

def mc_clustering(graph, max_iterations=5000, min_delta=0.0, verbose=False, max_no_progress=500, alpha=1000.0):
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
    num_vertices = len(graph.vs)
    best_ever_modularity = float('-inf')
    best_ever_membership = []
    empty_partitions = []
    
    for iteration in range(max_iterations):
        if no_progress_counter == max_no_progress:
            break
        selected_vertex    = int(random.random() * num_vertices)
        selected_community = mm.membership[selected_vertex]
        new_communities    = [c for c in partition_counts.keys() if c != selected_community]
        # Pick random index 0 to num_vertices, or 0 to num_vertices - 1 if empty_partitions is empty
        add_one            = len(empty_partitions) != 0
        community_index    = int(random.random() * (len(new_communities)+int(add_one)))
        
        if community_index == len(new_communities):
            new_community = empty_partitions.pop()
            partition_counts[new_community] = 0
        else:
            new_community = new_communities[community_index]
            
        mm.move_community(selected_vertex, new_community)
        delta = mm.modularity - previous_modularity
        p = r = 0.0  # Declaring p and r outside if statement for reporting
        if delta > min_delta:
            accept_change = True
        else:
#            p = min(math.exp(delta / alpha),1) #Have to think of how e^x works to continue this
            p = 1/(float(iteration)/1000.0 + 1.0)
            r = random.uniform(0.0, 1.0)
            if p > r:
                accept_change = True
            else:
                accept_change = False
        
        if accept_change:
            if mm.modularity > best_ever_modularity:
                if verbose:
                    print("mc_clustering: best modularity {0} -> {1}. Best #clusters {2} -> {3}".format(best_ever_modularity, mm.modularity,
                                                                                                        len(set([x for x in best_ever_membership])),
                                                                                                        len(set([x for x in mm.membership]))))
                best_ever_modularity = mm.modularity
                best_ever_membership = mm.membership.copy()
            no_progress_counter = 0
            partition_counts[selected_community] -= 1
            if not partition_counts[selected_community]:
                del(partition_counts[selected_community])
                empty_partitions.append(selected_community)
            partition_counts[new_community] += 1
            if verbose:
                print("mc_clustering. iteration={0} modularity={1} delta={2} p={3} r={4} went_back={5}".format(iteration, mm.modularity, delta, p, r, delta < 0.0))
            previous_modularity = mm.modularity
        else:
            no_progress_counter += 1
            mm.revert()
    
    if verbose:
        print("Finished mc_clustering. Clustered {0} communities into {1}.".format(len(best_ever_membership), len(set(best_ever_membership))))
    return VC(graph, normalize_membership(best_ever_membership))
    #     print("Finished mc_clustering. Clustered {0} communities into {1}.".format(len(mm.membership), len(set(mm.membership))))
    # return VC(graph, normalize_membership(mm.membership))

class EA_Mutator():
    def __init__(self, graph, membership):
        self.graph = graph
        self.num_vertices = len(graph.vs)                                       
        self.mm = ModularityMaintainer(graph, membership)

        self.partition_vertexes = defaultdict(set)
        for i, p in enumerate(self.mm.membership):
            self.partition_vertexes[p].add(i)

        self.partition_counts = dict()
        for i, s in self.partition_vertexes.items():
            self.partition_counts[i] = len(s)

        self.empty_partitions = []

    def copy(self):
        other = copy.copy(self)

        # explicitly copy the things that would have
        # only be reference copies
        other.mm                 = self.mm.copy()
        other.partition_vertexes = copy.deepcopy(self.partition_vertexes)
        other.partition_counts   = self.partition_counts.copy()
        other.empty_partitions   = self.empty_partitions.copy()
        return other

    def mutate(self):
        selected_vertex    = int(random.random() * self.num_vertices)
        selected_community = self.mm.membership[selected_vertex]
        new_communities    = [c for c in self.partition_counts.keys() if c != selected_community]

        # Pick random index 0 to num_vertices, or 0 to num_vertices - 1 if empty_partitions is empty
        add_one            = len(self.empty_partitions) != 0
        community_index    = int(random.random() * (len(new_communities)+int(add_one)))
        
        if community_index == len(new_communities):
            new_community = self.empty_partitions.pop()
            self.partition_counts[new_community] = 0
        else:
            new_community = new_communities[community_index]
            
        self.mm.move_community(selected_vertex, new_community)

        self.partition_counts[selected_community] -= 1
        if not self.partition_counts[selected_community]:
            del(self.partition_counts[selected_community])
            self.empty_partitions.append(selected_community)
        self.partition_counts[new_community] += 1

        # Return self to let chaining work below. Big win ;)
        return self

def ea_clustering(graph, n=10, max_iterations=5000, verbose=False):
#    pdb.set_trace()
    # setup initial population
    population = []
    num_vertices = len(graph.vs)

    modularity_key = lambda m: m.mm.modularity

    for i in range(n):
        print(i)
        population.append(EA_Mutator(graph, generate_random_membership(num_vertices)))

    for i in range(max_iterations):
        population = sorted(population, reverse=True, key=modularity_key)[:int(n/2)]
        if i%100 == 0:
            print("ea_clustering: iteration {0}/{1}. Best modularity {2}".format(i,max_iterations, population[0].mm.modularity))
        new_population = [m.copy().mutate() for m in population]
        population = population + new_population

    population.sort(reverse=True, key=modularity_key)
    best = population[0].mm.membership.copy()
    if verbose:
        print("Finished ea_clustering. Clustered into {0} communities.".format(len(set(best))))
    return VC(graph, normalize_membership(best))
    
valid_datasets = ['facebook','wikivote','collab', 'test', 'karate',]

def main(dataset=None, 
         algorithm=None, 
         verbose=False, 
         max_iters1=30000, 
         max_iters2=10000000, 
         write_clusters=False, 
         tries=1,
         max_no_progress=500,
         alpha=1000,
         export=False,
         ):
    dataset_file_name = {
        'facebook': os.path.join(*"data/egonets-Facebook/facebook_combined.txt".split("/")),
        'wikivote': os.path.join(*"data/wiki-Vote/wiki-Vote.txt".split("/")),
        'collab':   os.path.join(*"data/ca-GrQc/ca-GrQc.txt".split("/")),
        'karate':   os.path.join(*"data/karate/karate.txt".split("/")),
        'test':     os.path.join(*"data/test/test.txt".split("/")),
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
        'mc-cluster':  lambda g, kw: do_greedy_clustering(g, mc_clustering, **kw),
        'ea-cluster':  lambda g, kw: ea_clustering(g, **kw),
    }
    
    dataset_algorithm_params = defaultdict(lambda: defaultdict(dict))
    for data in valid_datasets:
        dataset_algorithm_params[dataset]['greedy-1']   = dict(verbose=verbose, max_iterations=max_iters1, tries=tries, max_no_progress=max_no_progress)
        dataset_algorithm_params[dataset]['greedy-2']   = dict(verbose=verbose, max_iterations=max_iters2, tries=tries, max_no_progress=max_no_progress)
        dataset_algorithm_params[dataset]['mc-cluster'] = dict(verbose=verbose, max_iterations=max_iters2, tries=tries, max_no_progress=max_no_progress,alpha=alpha)
        dataset_algorithm_params[dataset]['ea-cluster'] = dict(verbose=verbose, max_iterations=max_iters2)
        
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
                if alg in ('greedy-2', 'mc-cluster', 'ea-cluster'):
                    params = '{0}_'.format(max_iters2)
                file_name = time.strftime("data/community_{0}_{1}_{2}%y-%m-%d_%H-%M-%S.txt".format(data, alg, params))
                print("Writing {0}".format(file_name))
                with open(file_name, 'w') as f:
                    for i, c in enumerate(cluster.membership):
                        f.write("{0}\t{1}\n".format(i,c))
    
    if export:
        node_filename = "gephi_exports/nodes_{0}_{1}".format(dataset, algorithm)
        edge_filename = "gephi_exports/edges_{0}_{1}".format(dataset, algorithm)
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
                        choices=['eigenvector', 'walktrap', 'greedy-1', 'greedy-2', 'betweenness', 'mc-cluster', 'ea-cluster'],
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
    parser.add_argument('-m',
                        type=float,
                        help='alpha for mc_cluster algorithm',
                            default=1000)
    parser.add_argument('-c',
                        type=int,
                        help='max iterations with no change before assuming converged',
                            default=500)
    parser.add_argument('-w',
                        action='store_true',
                        help='write created clusters to disk',
                        default=False)
    parser.add_argument('-e',
                        action='store_true',
                        help='export Gephi spreadsheet csv file')
    parser.add_argument('-p',
                        type=str,
                        default='',
                        help='Run cProfile on main() function and store results in file provided.')

    args = parser.parse_args()

    if args.p:
        cProfile.run("""main(dataset=args.d,  algorithm=args.a,  verbose=args.v,  max_iters1=args.x1, max_iters2=args.x2, write_clusters=args.w, tries=args.t, max_no_progress=args.c, export=args.e, alpha=args.m)""", args.p)
    else:
        main(dataset=args.d, 
             algorithm=args.a, 
             verbose=args.v, 
             max_iters1=args.x1,
             max_iters2=args.x2,
             write_clusters=args.w,
             tries=args.t,
             max_no_progress=args.c,
             export=args.e,
             alpha=args.m,
        )
