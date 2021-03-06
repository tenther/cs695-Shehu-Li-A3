#!/usr/bin/python3 -u
from __future__ import division
import argparse
import cProfile
from collections import defaultdict, namedtuple
import copy
import datetime
import functools
import glob
import igraph as ig
import itertools
import math
import multiprocessing as mp
import numpy as np
import os
import pdb
import random
import re
import sys
import time
import json

VC = ig.VertexClustering
float_type = np.float64
f64_1 = float_type(1.0)
f64_2 = float_type(2.0)
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
        a           = np.zeros(no_of_comms, dtype=float_type)
        e           = np.zeros(no_of_comms, dtype=float_type)
        m           = np.zeros(no_of_comms, dtype=float_type)

        modularity  = 0.0
        edges       = [(edge.source, edge.target) for edge in graph.es]


        for v1, v2 in edges:
            c1 = membership[v1]
            c2 = membership[v2]
            if (c1==c2):
                e[c1] += f64_2
            a[c1] += f64_1
            a[c2] += f64_1

        no_of_edges = len(graph.es)
        f64_no_of_edges = float_type(no_of_edges)
        if no_of_edges > 0:
            for i in range(no_of_comms):
                tmp = a[i]/f64_2/f64_no_of_edges
                m[i] = e[i]/f64_2/f64_no_of_edges - tmp*tmp

        modularity = m.sum()

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

    def copy(self, other):
        self.graph       = other.graph
        self.membership  = other.membership.copy()
        self.edges       = other.edges
        self.no_of_edges = other.no_of_edges
        self.no_of_comms = other.no_of_comms
        self.modularity  = other.modularity
        self.e           = other.e.copy()
        self.a           = other.a.copy()
        self.m           = other.m.copy()
        self.previous    = other.previous
        return self

    def move_community(self, v, new_community):
        a               = self.a
        e               = self.e
        m               = self.m
        no_of_edges     = self.no_of_edges
        f64_no_of_edges = float_type(no_of_edges)
        no_of_comms     = self.no_of_comms
        membership      = self.membership
        adj             = self.adj
        modularity      = self.modularity


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

        if c1 == new_community:
            raise Exception("Moving node to it's current community ({0} -> {1}) will likely break move_community".format(c1, new_community))

        # Store previous from and to communities, and the prior a,e,m values.
        try:
            self.previous = (v, 
                             c1, 
                             a[c1],
                             e[c1], 
                             m[c1], 
                             new_community, 
                             a[new_community], 
                             e[new_community], 
                             m[new_community])
        except Exception as e:
            pdb.set_trace()
            raise
            
        for v2 in adj[v]:
            c2 = membership[v2]
            if c1 == c2:
                e[c1] -= f64_2
            if c2 == new_community:
                e[new_community] += f64_2
            a[c1] -= f64_1
            a[new_community] += f64_1

        # m array is used to track modularity component
        # of each community. Recalculate these for the
        # affected commununities.
        for i in [c1, new_community]:
            tmp = a[i]/f64_2/f64_no_of_edges
            m[i] = e[i]/f64_2/f64_no_of_edges - tmp*tmp

        membership[v] = new_community

        # Recalculate modularity
        self.modularity = m.sum()

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
        self.modularity = self.m.sum()
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
                if (target, source) not in E:
                    E.add((source, target))

    g = ig.Graph(directed=directed)
    g.add_vertices(sorted(list(V)))
    g.add_edges(list(E))
    return g

def export_gephi_csv(graph, membership, nodes_filename="nodes", edges_filename="edges"):
    g = graph
    atts = list(g.vs.attribute_names())
    atts.remove("name")
    with open(nodes_filename, 'w') as f:
        line = 'Id,Label,Community' + (atts and ',' or '') + ','.join(map(str, atts))
        f.write(line + "\n")
        for v in g.vs:
            line = "{0},{1},{2}".format(str(v.index),v["name"],membership[v.index])
            #We really only care about community, so I guess we don't necessarily need this
            if len(atts) > 0:
                temp = [str(v[att]) for att in atts]
                line += "," + ','.join(map(str,temp))
            f.write(line + "\n")
            
    with open(edges_filename, 'w') as f:
        f.write("source,target,type\n")
        temp = [e.tuple for e in g.es]
        for s,t in temp:
            line = "{0},{1},{2}".format(str(s),str(t),"undirected")
            f.write(line + "\n")

def do_clustering(graph, 
                  func, 
                  tries=100, 
                  max_iterations=5000, 
                  min_delta=0.0, 
                  max_no_progress=500, 
                  verbose=False,
                  alpha=None,
                  max_children=None,
                  population_size=None,
                  stats_rate=100
                  ):
    results = None
    if tries > 1:
        with mp.Pool(max_children) as p:
            results = p.starmap(func, [(graph, max_iterations, min_delta, verbose, max_no_progress, alpha, population_size, stats_rate) for _ in range(tries)])
    else:
        # When tries == 1 just run in this process to allow gathering of cProfile stats.
        results = [func(graph, max_iterations, min_delta, verbose, max_no_progress, alpha, population_size, stats_rate)]
    
    best_cluster = max(results, key=lambda r: r[0].modularity)[0]
    
    if func.__name__ == "ea_clustering":
        print(results)
        stats = results[0][1]
    else:
        data = [s for _,s in results]
        stats = stats_from_modularity_data(data, stats_rate)

    return best_cluster, stats

def stats_from_modularity_data(data, stats_rate):
    max_length = max([len(l) for l in data])
    for line in data:
        max_val = line[-1]
        line += [max_val] * (max_length - len(line))
    stats_prep = list(zip(*data))
    stats = []
    for i in range(len(stats_prep)):
        stats.append([i*stats_rate, max(stats_prep[i]), sum(stats_prep[i]) / len(stats_prep[i])])
    return stats

def write_stats_to_file(stats, filename):
    with open(filename, 'w') as f:
        for line in stats:
            f.write("{0},{1},{2}\n".format(line[0], line[1], line[2]))
            
def read_stats_from_file(filename):
    stats = []
    with open(filename, 'r') as f:
        for line in f:
            temp = line.split(",")
            stats.append([int(temp[0]),float(temp[1]),float(temp[2])])
    return stats

# Make communities indexed from 0
def normalize_membership(membership):
    communities = set(membership)
    old_to_new = dict(zip(sorted(list(communities)),
                          range(len(communities))))
    return [old_to_new[x] for x in membership]

# Expected data format - rows: run, column: iteration
def generate_random_membership(n, max_communities=float('inf')):
    membership = []
    if max_communities > n:
        max_communities = n
    for _ in range(n):
        membership.append(int(random.random() * max_communities))
    return normalize_membership(membership)

def greedy_clustering(graph, max_iterations=5000, min_delta=0.0, verbose=False, max_no_progress=500, alpha=0.0, population_size=None, stats_rate=100):
    # start with each vertex in its own commuanity
    mm = ModularityMaintainer(graph, [i for i, _ in enumerate(graph.vs)])
    modularity_vals_for_run = []

    partition_counts = defaultdict(int)
    for _, p in enumerate(mm.membership):
        partition_counts[p] = partition_counts[p] + 1

    previous_modularity = mm.modularity
    no_progress_counter = 0
    num_vertices = len(graph.vs)
    empty_partitions = set([])
    non_empty_communities = []
    
    for iteration in range(max_iterations):
        if no_progress_counter == max_no_progress:
            break
        selected_vertex    = int(random.random() * num_vertices)
        selected_community = mm.membership[selected_vertex]

        p_new_com = 1/(len(partition_counts)+1)
        r = random.uniform(0.0, 1.0)
        if p_new_com > r and empty_partitions:
            new_community = empty_partitions.pop()
            partition_counts[new_community] = 0
        else:
            new_community = mm.membership[int(random.random() * float(num_vertices))]
            if new_community == selected_community:
                continue

        mm.move_community(selected_vertex, new_community)
        delta = mm.modularity - previous_modularity
        if delta > min_delta:
            no_progress_counter = 0
            partition_counts[selected_community] -= 1
            if not partition_counts[selected_community]:
                del(partition_counts[selected_community])
                empty_partitions.add(selected_community)
            partition_counts[new_community] += 1
            if verbose:
                print("Greedy clustering. iteration={0} modularity:={1} delta={2}.".format(iteration, mm.modularity, mm.modularity - previous_modularity))
            previous_modularity = mm.modularity
        else:
            no_progress_counter += 1
            mm.revert()
        if iteration % stats_rate == 0:
            modularity_vals_for_run.append(mm.modularity)
    
    return_vc = VC(graph, normalize_membership(mm.membership))
    if verbose:
        print("""Finished greedy_clustering. 
    Clustered {0} communities into {1}. 
    ModularityMaintainer modularity = {2}. 
    VertexClustering communities={3}, 
    modularity={4}""".format(len(mm.membership), 
                             len(set(mm.membership)), 
                             mm.modularity,
                             len(set(return_vc.membership)),
                             return_vc.modularity))
    return return_vc, modularity_vals_for_run

def mc_clustering(graph, max_iterations=5000, min_delta=0.0, verbose=False, max_no_progress=500, alpha=1000.0, population_size=None, stats_rate=100):
    # start with each vertex in its own commuanity
    mm = ModularityMaintainer(graph, [i for i, _ in enumerate(graph.vs)])
    modularity_vals_for_run = []

    partition_counts = defaultdict(int)
    for i, p in enumerate(mm.membership):
        partition_counts[p] = partition_counts[p] + 1

    previous_modularity = mm.modularity
    no_progress_counter = 0
    accept_change = False
    num_vertices = len(graph.vs)
    best_ever_modularity = float('-inf')
    best_ever_membership = []
    empty_partitions = set([])
    non_empty_communities = []
    
    for iteration in range(max_iterations):
        if no_progress_counter == max_no_progress:
            break
        selected_vertex    = int(random.random() * num_vertices)
        selected_community = mm.membership[selected_vertex]
        
        p_new_com = 1/(len(partition_counts)+1)
        r = random.uniform(0.0, 1.0)
        if p_new_com > r and empty_partitions:
            new_community = empty_partitions.pop()
            partition_counts[new_community] = 0
        else:
            new_community = mm.membership[int(random.random() * float(num_vertices))]
            if new_community == selected_community:
                continue
                
        mm.move_community(selected_vertex, new_community)
        delta = mm.modularity - previous_modularity
        p = r = 0.0  # Declaring p and r outside if statement for reporting
        if delta > min_delta:
            accept_change = True
        else:
            p = math.exp(delta / alpha * iteration / max_iterations)
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
                empty_partitions.add(selected_community)
            partition_counts[new_community] += 1
            if verbose:
                print("mc_clustering. iteration={0} modularity={1} delta={2} p={3} r={4} went_back={5}".format(iteration, mm.modularity, delta, p, r, delta < 0.0))
            previous_modularity = mm.modularity
        else:
            no_progress_counter += 1
            mm.revert()
        if iteration % stats_rate == 0:
            modularity_vals_for_run.append(mm.modularity)
    
    return_vc = VC(graph, normalize_membership(best_ever_membership))
    if verbose:
        print("""Finished mc_clustering. 
    Clustered {0} communities into {1}. 
    ModularityMaintainer modularity = {2}. 
    VertexClustering communities={3}, 
    modularity={4}""".format(len(best_ever_membership), 
                             len(set(best_ever_membership)), 
                             best_ever_modularity,
                             len(set(return_vc.membership)),
                             return_vc.modularity))
    return return_vc, modularity_vals_for_run

class EA_Mutator():
    def __init__(self, graph, membership):
        self.graph = graph
        self.num_vertices = len(graph.vs)                                       
        self.mm = ModularityMaintainer(graph, membership)
        self.partition_counts = defaultdict(int)
        for p in self.mm.membership:
            self.partition_counts[p] += 1

        self.empty_partitions = []

    def copy(self, other):
        # explicitly copy the things that would have
        # only be reference copies
        self.mm.copy(other.mm)
        self.partition_counts   = other.partition_counts.copy()
        self.empty_partitions   = other.empty_partitions.copy()

    def mutate(self, n_mutations = 1):
        for _ in range(n_mutations):
            selected_vertex    = int(random.random() * self.num_vertices)
            selected_community = self.mm.membership[selected_vertex]

            while True:
                p_new_com = 1/(len(self.partition_counts)+1)
                r = random.uniform(0.0, 1.0)
                if p_new_com > r and self.empty_partitions:
                    new_community = self.empty_partitions.pop()
                    self.partition_counts[new_community] = 0
                else:
                    community_index    = int(random.random() * (len(self.mm.membership)))
                    new_community = self.mm.membership[community_index]

                if new_community == selected_community:
                    continue
                self.mm.move_community(selected_vertex, new_community)

                self.partition_counts[selected_community] -= 1
                if not self.partition_counts[selected_community]:
                    del(self.partition_counts[selected_community])
                    self.empty_partitions.append(selected_community)
                self.partition_counts[new_community] += 1
                break

def do_ea_work(m_target, m_source):
    m_target.copy(m_source)
    m_target.mutate(1)
    return m_target

def ea_clustering(graph, max_iterations=5000, min_delta=None, verbose=False, max_no_progress=None, alpha=None, population_size=100, stats_rate=100):
    # setup initial population
    population = []
    num_vertices = len(graph.vs)
    modularity_vals_for_run = []

    modularity_key = lambda m: m.mm.modularity

    # Preallocate 2n individuals in population, so we can resuse objects.
    if verbose:
        print("ea_clustering: Allocating population of size 2*population_size={0}".format(2*population_size))
    for i in range(2*population_size):
        population.append(EA_Mutator(graph, generate_random_membership(num_vertices)))

    for i in range(max_iterations):
        population.sort(reverse=True, key=modularity_key)

        if population[0].mm.modularity == population[population_size-1].mm.modularity:
            if verbose:
                print("ea_clustering: converged")
            break
        # copying whole objects is slow, so use custom copy functions and copy better
        # mutators over to existing ones that are poorer.

        for m_idx in range(population_size, 2*population_size):
            population[m_idx].copy(population[m_idx - population_size])
            population[m_idx].mutate(1)

        if i%100 == 0:
            if verbose:
                print("ea_clustering: iteration {0}/{1}. Best modularity {2}".format(i,max_iterations, population[0].mm.modularity))
        if i % stats_rate == 0:
            modularity_vals_for_run.append([i, population[0].mm.modularity,
                                            sum(m.mm.modularity for m in population[:population_size]) / population_size])

    population.sort(reverse=True, key=modularity_key)
    best = population[0].mm.membership.copy()
    if verbose:
        print("Finished ea_clustering. Clustered into {0} communities.".format(len(set(best))))
    return VC(graph, normalize_membership(best)), modularity_vals_for_run
    
valid_datasets = ['facebook','wikivote','collab', 'test', 'karate',]

def main(dataset=None, 
         algorithm=None, 
         verbose=False, 
         max_iters=30000, 
         write_clusters=False, 
         tries=1,
         max_no_progress=500,
         alpha=1000,
         export=False,
         display=False,
         write_report=False,
         stats_rate=100,
         do_dendro=False,
         ):

    for d in ['reports', 'gephi_exports']:
        if not os.path.isdir(d):
            os.makedirs(d)

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

    dataset_igraph_layout = {
        'facebook':  'lgl',
        'wikivote':  'lgl',
        'collab':    'lgl',
        'karate':    'kk',
        'test':      'kk',
    }

    algorithm_func = {
        'betweenness': lambda g, kw: g.community_edge_betweenness(),
        'eigenvector': lambda g, kw: (g.community_leading_eigenvector(), None),
        'walktrap':    lambda g, kw: g.community_walktrap(),
        'greedy':      lambda g, kw: do_clustering(g, greedy_clustering, **kw),
        'mc-cluster':  lambda g, kw: do_clustering(g, mc_clustering, **kw),
        'ea-cluster':  lambda g, kw: do_clustering(g, ea_clustering, tries=1, max_children=1, **kw),
    }
    
    dataset_algorithm_params = defaultdict(lambda: defaultdict(dict))
    for data in valid_datasets:
        dataset_algorithm_params[dataset]['greedy']     = dict(verbose=verbose, max_iterations=max_iters, tries=tries, max_no_progress=max_no_progress, stats_rate=stats_rate)
        dataset_algorithm_params[dataset]['mc-cluster'] = dict(verbose=verbose, max_iterations=max_iters, tries=tries, max_no_progress=max_no_progress, alpha=alpha, stats_rate=stats_rate)
        dataset_algorithm_params[dataset]['ea-cluster'] = dict(verbose=verbose, max_iterations=max_iters, population_size=tries, stats_rate=stats_rate)
        
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
            results = func(graphs[data], kw)
            if alg in ('betweenness', 'walktrap'):
                clusters[data][alg] = (results.as_clustering(), None)
                if do_dendro:
                    file_name = 'images/' + dataset + '_' + alg + '_dendrogram.png'
                    print("Writing {0}".format(file_name))
                    ig.plot(results, file_name, bbox=(1200, 1200))
            else:
                clusters[data][alg] = results
            dataset_algorithm_time[data][alg] = time.time() - t0
                

    for data, graph in graphs.items():
        print("Graph summary for dataset {0}: {1}".format(data, graph.summary()))
        for alg, results in clusters[data].items():
            cluster, stats = results
            print("Clusters summary for dataset {0}.{1}: {2}".format(data, alg, cluster.summary()))
            print("    modularity: {0}".format(cluster.modularity))
            print("    time: {0}".format(dataset_algorithm_time[data][alg]))
        print("")

    report_file_timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())

    node_filename = None
    edge_filename = None
    if export:
        node_filename = "gephi_exports/nodes_{0}_{1}_{2}.csv".format(dataset, algorithm, report_file_timestamp)
        edge_filename = "gephi_exports/edges_{0}_{1}_{2}.csv".format(dataset, algorithm, report_file_timestamp)
        export_gephi_csv(clusters[data][alg][0].graph,clusters[data][alg][0].membership,node_filename,edge_filename)
        print("Exporting Gephi spreadsheet csv files: {0}.csv, {1}.csv".format(node_filename,edge_filename))

    if write_clusters or write_report or display:
        for data in clusters.keys():
            for alg, results in clusters[data].items():
                cluster, stats = results
                params = ""
                if alg in ('greedy', 'mc-cluster', 'ea-cluster'):
                    params = '{0}_'.format(max_iters)
                file_name_base = time.strftime("reports/{0}_{1}_{2}%y-%m-%d_%H-%M-%S".format(data, alg, params))
                community_file_name = "{0}_community.txt".format(file_name_base)
                print("Writing {0}".format(community_file_name))
                with open(community_file_name, 'w') as f:
                    for i, c in enumerate(cluster.membership):
                        f.write("{0}\t{1}\n".format(i,c))
    
                if display:
                    print("Displaying results for dataset {0}, algorithm {1}, with layout {2}".format(data, alg, dataset_igraph_layout[data]))
                    executable = re.sub(r'[^/\\]+.py$', 'display_graph_communities.py', sys.argv[0])
                    os.system("{0} -g {1} -m {2} -y {3}".format(executable, dataset_file_name[data], community_file_name, dataset_igraph_layout[data]))

                if write_report:

                    report_file_name = "{0}_report.json".format(file_name_base)
                    print("Writing {0}".format(report_file_name))

                    report_fields = [['dataset', data],
                                     ['graph_data', dataset_file_name[data]],
                                     ['algorithm', algorithm],
                                     ['verbose', verbose],
                                     ['max_iters', max_iters],
                                     ['write_clusters', write_clusters],
                                     ['tries', tries],
                                     ['max_no_progress', max_no_progress],
                                     ['alpha', alpha],
                                     ['export', export],
                                     ['display', display],
                                     ['write_report', write_report],
                                     ['community_file', community_file_name],
                                     ['modularity', clusters[data][alg][0].modularity],
                                     ['elapsed_time', dataset_algorithm_time[data][alg]],
                                     ['stats_rate', stats_rate],
                                     ['cluster_summary', cluster.summary()],
                    ]

                    if node_filename:
                        report_fields.extend([['gephi_node_filename', node_filename],
                                              ['gephi_edge_filename', edge_filename]])
                        
                    if alg in ('greedy', 'mc-cluster', 'ea-cluster'):
                        stats_report_file_name = "{0}_stats_report.csv".format(file_name_base)
                        print("Writing {0}".format(stats_report_file_name))
                        write_stats_to_file(stats, stats_report_file_name)
                        report_fields.append(['stats_file', stats_report_file_name])

                    with open(report_file_name, 'w') as f:
                        f.write(json.dumps({k:v for k,v in report_fields}, indent=2, sort_keys=True))
                    
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        choices=valid_datasets,
                        help='Dataset to process',
                        default=None)
    parser.add_argument('-a',
                        choices=['eigenvector', 'walktrap', 'greedy', 'betweenness', 'mc-cluster', 'ea-cluster'],
                        help='Algorithm to run on dataset',
                            default=None)
    parser.add_argument('-v',
                        action='store_true',
                        help='Verbose mode',
                        default=False)
    parser.add_argument('-t',
                        type=int,
                        help='Number of tries to run with greedy, and mc-cluster algorithms. Population size with ea-cluster algorithm',
                            default=1)
    parser.add_argument('-x',
                        type=int,
                        help='max iterations to run',
                            default=500)
    parser.add_argument('-m',
                        type=float,
                        help='alpha for mc_cluster algorithm',
                            default=1000)
    parser.add_argument('-c',
                        type=int,
                        help='max iterations with no change before assuming converged',
                            default=0)
    parser.add_argument('-w',
                        action='store_true',
                        help='write created clusters to disk',
                        default=False)
    parser.add_argument('-e',
                        action='store_true',
                        help='export Gephi spreadsheet csv file')
    parser.add_argument('-y',
                        action='store_true',
                        help='Display graph results visually in igraph.')
    parser.add_argument('-p',
                        type=str,
                        default='',
                        help='Run cProfile on main() function and store results in file provided.')
    parser.add_argument('-r',
                        action='store_true',
                        default=False,
                        help='Write statistics and filenames to report file.')
    parser.add_argument('-dro',
                        action='store_true',
                        default=False,
                        help='Write dendrogram file.')
    parser.add_argument('-sr',
                        type=int,
                        help='Rate at which to store modularity statistics',
                            default=100)
    parser.add_argument('-z',
                        type=str,
                        default='',
                        help='Run a test whose parameters are given in the specified json file (with same field format as an output report.')

    args = parser.parse_args()

    if args.z:
        with open(args.z) as report_file:
            report = json.loads(report_file.read())
            args.d = report["dataset"]
            args.a = report["algorithm"]
            args.v = report["verbose"]
            args.t = report["tries"]
            args.x = report["max_iters"]
            args.m = report["alpha"]
            args.c = report["max_no_progress"]
            args.w = report["write_clusters"]
            args.r = report["write_report"]
            args.e = report["export"]
            args.y = report["display"]
            args.sr = report["stats_rate"]

            main(dataset=args.d, 
                 algorithm=args.a, 
                 verbose=args.v, 
                 max_iters=args.x,
                 write_clusters=args.w,
                 tries=args.t,
                 max_no_progress=args.c,
                 export=args.e,
                 alpha=args.m,
                 display=args.y,
                 write_report=args.r,
                 stats_rate=args.sr,
                 do_dendro=args.dro,
            )
    else:
        if args.c == 0:
            args.c = int(args.x * 0.01)

        if args.p:
            cProfile.run("""main(dataset=args.d,  algorithm=args.a,  verbose=args.v,  max_iters=args.x, write_clusters=args.w, tries=args.t, max_no_progress=args.c, export=args.e, alpha=args.m,display=args.y,write_report=args.r,stats_rate=args.sr)""", args.p)
        else:
            main(dataset=args.d, 
                 algorithm=args.a, 
                 verbose=args.v, 
                 max_iters=args.x,
                 write_clusters=args.w,
                 tries=args.t,
                 max_no_progress=args.c,
                 export=args.e,
                 alpha=args.m,
                 display=args.y,
                 write_report=args.r,
                 stats_rate=args.sr,
                 do_dendro=args.dro,
            )
