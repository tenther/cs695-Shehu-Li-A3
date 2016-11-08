#!/usr/bin/python
import igraph as ig
import argparse
import a3
import pdb

def load_membership(membership_file_name):
    membership = []
    with open(membership_file_name) as membership_file:
        for line in membership_file:
            v, c = line.strip().split()
            membership.append(int(c))
    return membership

def main(graph_file_name, membership_file_name, directed):
    g          = a3.load_tsv_edges(graph_file_name, directed=directed)
    membership = load_membership(membership_file_name)
    visual_style={}
#    visual_style['vertex_label']=g.vs["name"]
#    visual_style['label_dist']=1
    visual_style['bbox']=(600,600)
    visual_style['margin']=50

#    layout = g.layout_kamada_kawai()
    layout = g.layout_lgl()

    community_color = ['purple', 'green', 'light blue', 'yellow', 
                       'red', 'orange', 'pink', 'white', 
                       'black', 'brown', 'gray',]

    visual_style['vertex_color']=[community_color[i%(len(community_color))] for i in membership]
    ig.plot(g, layout=layout, **visual_style)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g',
                        type=str,
                        help='file containing graph information')
    parser.add_argument('-m',
                        type=str,
                        help='file containing membership information')
    parser.add_argument('-d',
                        action='store_true',
                        help='Directed',
                        default=False)
    args = parser.parse_args()
    main(args.g, args.m, args.d)
