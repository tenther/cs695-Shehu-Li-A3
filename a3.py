#!/usr/bin/python
import igraph as ig
import glob
import os
import pdb
import re
import functools

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
    g = ig.Graph()
    g.add_vertices(list(V))
    g.add_edges(list(E))
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

def main():
    #  create path to data in a way that will work with Windows
    path_to_fb_data = os.path.join(*"data/egonets-Facebook/facebook".split("/"))
    fb_g = load_facebook(data_dir=path_to_fb_data)

    fb_eig_dend  = fb_g.community_leading_eigenvector()
    fb_walk_dend = fb_g.community_walktrap()

    #  create path to data in a way that will work with Windows
    wiki_vote_file_name = os.path.join(*"data/wiki-Vote/wiki-Vote.txt".split("/"))
    wv_g = load_wiki_vote(wiki_vote_file_name)
    wv_eig_dend  = wv_g.community_leading_eigenvector()
    wv_walk_dend = wv_g.community_walktrap()

    pdb.set_trace()

    collaboration_file_name = os.path.join(*"data/ca-GrQc/ca-GrQc.txt".split("/"))
    co_g = load_collaboration(collaboration_file_name)
    co_eig_dend  = co_g.community_leading_eigenvector()
    co_walk_dend = co_g.community_walktrap()

    return

if __name__ == '__main__':
    main()



