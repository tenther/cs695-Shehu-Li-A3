#!/usr/bin/python
import igraph as ig
import glob
import os
import pdb
import re
import functools

# Load one set of files 
def load_facebook(data_dir):
    data_file_path = functools.partial(os.path.join, data_dir)

    # It seems to be faster to loaded vertices and edges into sets and
    # create the graph with them all at once, instead of adding them
    # to the graph as we go. Didn't rigorously check this though.
    V = set()
    E = set()
            
    for circles_file_name in glob.glob(data_file_path("*.circles")):
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

                # Links from primary vertex to others is implicit. E is a set, so we won't have dups.
                E.add((ego_id, v1))
                E.add((ego_id, v2))


    # It seems igraph will make all vertices a contiguous range, even
    # if there are gaps (which would cause the vertices and edges to
    # get out of sync.) So add them in here and warn.
    for i in xrange(max(V) + 1):
        if i not in V:
            print "Adding missing vertex {0} to V".format(i)
            V.add(i)
    g = ig.Graph()
    g.add_vertices(list(V))
    g.add_edges(list(E))
    return g

def main():
    pdb.set_trace()
    path_to_data = os.path.join(*"data/egonets-Facebook/facebook".split("/"))
    fb_g = load_facebook(path_to_data)
    c = fb_g.community_edge_betweenness(directed=False)
    return

if __name__ == '__main__':
    main()



