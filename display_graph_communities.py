#!/usr/bin/python3
import a3
import argparse
import igraph as ig
import json
import pdb

dataset_igraph_layout = {
    'facebook':  'lgl',
    'wikivote':  'grid',
    'collab':    'lgl',
    'karate':    'kk',
    'test':      'kk',
}

def load_membership(graph, membership_file_name):
    membership = [None for _ in graph.vs]
    with open(membership_file_name) as membership_file:
        for line in membership_file:
            v, c = line.strip().split()
            # idx = graph.vs.select(name_eq=v)[0].index
            # membership[idx] = int(c)
            membership[int(v)] = int(c)
    return membership

def main(graph_file_name, membership_file_name, directed, labels=False, layout_name='kk'):
#    pdb.set_trace()
    g          = a3.load_tsv_edges(graph_file_name, directed=directed)
    membership = load_membership(g, membership_file_name)
    visual_style={}
    if labels:
        visual_style['vertex_label']=g.vs["name"]
        visual_style['label_dist']=1
    visual_style['bbox']=(600,600)
    visual_style['margin']=50

    layout = g.layout(layout_name)

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
    parser.add_argument('-r',
                        type=str,
                        default='',
                        help='file report information')
    parser.add_argument('-l',
                        action='store_true',
                        help='Display labels on graph',
                        default=False)
    parser.add_argument('-y',
                        choices=['kk', 'lgl','drl'],
                        help='Layout to use',
                            default='kk')
    args = parser.parse_args()
    if args.r:
        with open(args.r) as report_file:
            report = json.loads(report_file.read())
            main(report["graph_data"], report["community_file"], False, args.l, dataset_igraph_layout[report["dataset"]])
    else:
        main(args.g, args.m, False, args.l, args.y)
