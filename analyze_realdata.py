import re
import networkx as nx
import matplotlib.pyplot as plt
import math
import Analyser.analyser as analyser
import csv
import os


def analyse():
    # G2 = G.subgraph(sorted(list(G.nodes))[0:10000])
    G2 = G
    common_lim = (0, 4, -7, 0)
    # analyser.draw_degree_dist(G2.degree, hold=False, lbl='度分布', fit_func=analyser.linear_log_fit, lim=common_lim, fit_range=(1.2, 2.5, 0),
    #                           G=G2)
    # analyser.draw_degree_dist(G2.in_degree, hold=False, lbl='入度分布', fit_func=analyser.linear_log_fit, lim=common_lim, fit_range=(1.2, 2.0, 0),
    #                           G=G2)
    # analyser.draw_degree_dist(G2.out_degree, hold=False, lbl='出度分布', fit_func=analyser.linear_log_fit, lim=common_lim, fit_range=(1.2, 2.5, 0),
    #                           G=G2)
    # print('Number of nodes: %r' % (G2.number_of_nodes()))
    # print('Number of edges: %r' % (G2.number_of_edges()))
    analyser.analyse_clustering_coefficient(G2)
    # analyser.analyse_gini_coefficient(G)
    # analyser.analyse_gini_coefficient(G, type='in')
    # analyser.analyse_gini_coefficient(G, type='out')
    # analyser.analyse_gini_coefficient(G, lbl='节点度-洛伦兹曲线', hold=False)
    # analyser.analyse_gini_coefficient(G, type='in', lbl='节点入度-洛伦兹曲线', hold=False, save_path=save_path)
    # analyser.analyse_gini_coefficient(G, type='out', lbl='节点出度-洛伦兹曲线', hold=False, save_path=save_path)
    # analyser.analyse_clustering_coefficient(G2, save_path=path, fit_range=(1.5, 3, -0.3), ticks=True, lim=(0, 3.5, -3, 0))


if __name__ == '__main__':
    # path = './data/stable/TWITTER'
    path = '/home/hviktortsoi/Documents/graph_data'
    limit = 8000000
    start = 40000000
    with open(path + '/soc-LiveJournal1.txt', 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        G = nx.Graph()
        for number, row in enumerate(reader):
            if number > start:
                G.add_edge(int(row[0]), int(row[1]))
                if number > start + limit:
                    print(G.number_of_nodes())
                    analyse()
                    # nx.write_gpickle(G, 'data/stable/LIVEJ/graph.gpickle')
                    break
