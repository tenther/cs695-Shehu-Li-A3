#!/usr/bin/python
import igraph as ig
import glob
import pdb
import re

def main():
    pdb.set_trace()
    g = ig.Graph()
    V = set()
    E = set()
    for fn in glob.glob("data/egonets-Facebook/facebook/*.edges"):
        v_ego = int(fn.split('/')[-1].split('.')[0])
        V.add(v_ego)
        with open(fn) as f:
            for line in f.xreadlines():
                v1, v2 = [int(v) for v in line.strip().split()]
                V.add(v1)
                V.add(v2)
                E.add((v1, v2))
                E.add((v_ego, v1))
                E.add((v_ego, v2))
        for v in V:
            if v != v_ego:
                E.add((v_ego, v))
    for i in xrange(max(V) + 1):
        V.add(i)
    g.add_vertices(list(V))
    g.add_edges(list(E))
    return g

if __name__ == '__main__':
    main()
    
