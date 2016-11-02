#!/usr/bin/python
import igraph as ig
import glob
import os
import pdb
import re
import functools
from collections import defaultdict
import random

comments_re = re.compile(r'#.*')

# Load one set of files 
def load_facebook(data_dir, file_names=None):
    data_file_path = functools.partial(os.path.join, data_dir)
    if not file_names:
        if not data_dir:
            raise Exception("Must provide data_dir with file_names")
        file_names = glob.glob(data_file_path("*.circles"))


    # It seems to be faster to loaded vertices and edges into sets and
    # create the graph with them all at once, instead of adding them
    # to the graph as we go. Didn't rigorously check this though.
    V = set()
    E = set()

    for circles_file_name in file_names:
        # Add each id from a file name to the vertices
        ego_id = int(circles_file_name.split('/')[-1].split('.')[0])
        V.add(ego_id)

        # Add vertices from feat file
        feat_file_name = os.path.join(data_dir, "{0}.feat".format(ego_id))
        with open(feat_file_name) as feat_file:
            for line in feat_file:
                fields = line.strip().split()
                id = int(fields[0])
                V.add(id)

        # Add edges
        edge_file_name = os.path.join(data_dir, "{0}.edges".format(ego_id))
        with open(edge_file_name) as edge_file:
            for line in edge_file:
                v1, v2 = [int(v) for v in line.strip().split()]
                V.add(v1)
                V.add(v2)
                E.add((v1, v2))

                # make sure links are symmetrical
                E.add((v2, v1))

                # Links from primary vertex to others is implicit. E is a set, so we won't have dups.
                E.add((ego_id, v1))
                E.add((ego_id, v2))

                # make sure links are symmetrical
                E.add((v1, ego_id))
                E.add((v2, ego_id))


    # It seems igraph will make all vertices a contiguous range, even
    # if there are gaps (which would cause the vertices and edges to
    # get out of sync.) So add them in here and warn.
    for i in xrange(max(V) + 1):
        if i not in V:
            print "Adding missing vertex {0} to V".format(i)
            V.add(i)
    g = ig.Graph(n=len(V), edges=list(E))

    return g

def load_wiki_vote(data_file_name):

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

    g = ig.Graph()
    g.add_vertices(list(V))
    g.add_edges(list(E))
    return g

# this file appears to be the same format as wikivote
load_collaboration = load_wiki_vote

def greedy_clustering(graph, max_iterations=5000, min_delta=0.0):
    VC = ig.VertexClustering
    # start with each vertex in its own commuanity
    vc = VC(graph, [i for i, _ in enumerate(graph.vs)])

#    pdb.set_trace()
    # shortest_paths = graph.shortest_paths_dijkstra(mode=ig.ALL)

    partition_vertexes = defaultdict(set)
    for i, p in enumerate(vc.membership):
        partition_vertexes[p].add(i)

    partition_counts = dict()
    for i, s in partition_vertexes.iteritems():
        partition_counts[i] = len(s)

    for iteration in xrange(max_iterations):
        # Copy membership, just to avoid odd errors. May not be necessary.
        membership         = list(vc.membership)
        selected_vertex    = random.randint(0, len(membership) - 1)
#        new_communities    = set([i for i in membership if i != membership[selected_vertex]])
        new_communities    = partition_counts.keys()
        random.shuffle(new_communities)
        found_one          = False
        found_community    = 0
        for new_community in new_communities:
            membership[selected_vertex] = new_community
            new_vc = VC(graph,membership)
            delta = new_vc.modularity - vc.modularity
            if delta > min_delta:
                found_community = new_community
                found_one = True
                break
        if found_one:
            partition_counts[vc.membership[selected_vertex]] -= 1
            if not partition_counts[vc.membership[selected_vertex]]:
                del(partition_counts[vc.membership[selected_vertex]])
            partition_counts[found_community] += 1
            vc = new_vc
            print "Greedy clustering. iteration={0} modularity:={1} delta={2}".format(iteration, vc.modularity, delta)
        # else:
        #     print "Greedy clustering converged at {0} iterations.".format(iteration)
        #     break
    return vc

def main():


    # algorithm_func = {
    #     'eigenvector': (Graph.community_leading_eigenvector, None),
    #     'walktrap':    (Graph.community_walktrap, Dendrogram.as_clusterin),
    #     'greedy':      (greedy, None),

    graphs = {}
    clusters = defaultdict(dict)

    #  create path to data in a way that will work with Windows
    path_to_fb_data = os.path.join(*"data/egonets-Facebook/facebook".split("/"))

    graphs['facebook']                     = load_facebook(data_dir=path_to_fb_data)
    clusters['facebook']['eigenvector'] = graphs['facebook'].community_leading_eigenvector()
    clusters['facebook']['walktrap']    = graphs['facebook'].community_walktrap().as_clustering()
    clusters['facebook']['greedy']      = greedy_clustering(graphs['facebook'])

    wiki_vote_file_name                    = os.path.join(*"data/wiki-Vote/wiki-Vote.txt".split("/"))
    graphs['wikivote']                     = load_wiki_vote(wiki_vote_file_name)
    clusters['wikivote']['eigenvector'] = graphs['wikivote'].community_leading_eigenvector()
    clusters['wikivote']['walktrap']    = graphs['wikivote'].community_walktrap().as_clustering()

    collaboration_file_name              = os.path.join(*"data/ca-GrQc/ca-GrQc.txt".split("/"))
    graphs['collab']                     = load_collaboration(collaboration_file_name)
    clusters['collab']['eigenvector'] = graphs['collab'].community_leading_eigenvector()
    clusters['collab']['walktrap']    = graphs['collab'].community_walktrap().as_clustering()

    for dataset, graph in graphs.items():
        print "Graph summary for dataset {0}: {1}".format(dataset, graph.summary())
#        print "    modularity: {0}".format(graph.modularity())
        for algorithm, cluster in clusters[dataset].items():
            print "Clusters summary for dataset {0}.{1}: {2}".format(dataset, algorithm, cluster.summary())
            print "    modularity: {0}".format(cluster.modularity)
        print ""

    return

if __name__ == '__main__':
    main()


