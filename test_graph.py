#!/usr/bin/python

import pdb
import igraph as ig

def main():

    # Graph based on slide 44 of lecture notes 7
    V = set()

    E = [
    ('a', 'b'),
    ('b', 'c'),
    ('a', 'c'),
    ('c', 'd'),
    ('d', 'e'),
    ('d', 'i'),
    ('e', 'f'),
    ('e', 'g'),
    ('f', 'g'),
    ('g', 'h'),
    ('h', 'i'),
    ('h', 'j'),
    ('j', 'i'),
    ('j', 'k'),
    ('i', 'k')]

    for s,t in E:
        V.add(s)
        V.add(t)

    g = ig.Graph(directed=False)
    g.add_vertices(list(V))
    g.add_edges(list(E))

    fixed_community = {'a': 0,
                       'b': 0,
                       'c': 0,
                       'd': 0,
                       'e': 1,
                       'f': 1,
                       'g': 1,
                       'h': 2,
                       'i': 2,
                       'j': 2,
                       'k': 2,
    }
    
    community_color = ['purple', 'green', 'light blue', 'yellow', 
                       'red', 'orange', 'pink', 'white', 
                       'black', 'brown', 'gray',]

    communities = [0.0 for _ in range(len(g.vs))]

    for n, c in fixed_community.items():
        i = g.vs.select(name_eq=n)[0].index
        communities[i] = c

    layout = g.layout_kamada_kawai()

    clustering = {}
    clustering['betweenness'] = g.community_edge_betweenness().as_clustering()
    clustering['eigenvector'] = g.community_leading_eigenvector()
    clustering['walktrap']    = g.community_walktrap().as_clustering()
    clustering['fixed']       = ig.VertexClustering(g, communities)

    visual_style={}
    visual_style['vertex_label']=g.vs["name"]
    visual_style['bbox']=(600,600)
    visual_style['margin']=50
    visual_style['label_dist']=1

#    pdb.set_trace()

    for name, clustering in sorted(clustering.items()):
        visual_style['vertex_color']=[community_color[i] for i in clustering.membership]
        print "Displaying clustering of {0} algorithm. Modularity is {1}".format(name, clustering.modularity)
        ig.plot(g, layout=layout, **visual_style)

    return

if __name__=="__main__":
    main()
    
