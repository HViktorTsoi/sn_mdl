# coding:utf-8
import snap
from collections import defaultdict
import json
import networkx as nx


def save():
    diameter_list[new_node_id] = d
    print("%d %d" % (new_node_id, d))
    if new_node_id % 4 == 0:
        with open("%s/diam_evo_data.json" % path, 'w') as file:
            file.write(json.dumps({
                "dia": diameter_list,
                "cc": [],
                "edge": []
            }))


# 加载图
path = '../data/20180130_204852_n20000_e74456_初始不连通d10k10'
G = snap.LoadPajek(snap.PNGraph, path + '/graph.paj')
# G_nx = nx.Graph(nx.read_pajek(path + '/graph.paj'))

# 两种方法的间断点
break_point = 15000
# 存储节点列表
node_vector = snap.TIntV()
node_list = []
# 存储网络直径的列表
diameter_list = {}
# 用精密方法计算
# for node_id in range(0, break_point, 1):
#     # 为估算方法的节点列表计数
#     node_list.append("%d" % node_id)
#     sub_graph = G_nx.subgraph(node_list)
#     new_node_id = sub_graph.number_of_nodes()
#     if int(new_node_id) > 0:
#         d = nx.diameter(sub_graph)
#         save()

for node_id in range(0, break_point):
    node_vector.Add(node_id)
# 用估计方法计算
for node_id in range(break_point, G.GetNodes(), 1):
    test_dis = 100
    node_vector.Add(node_id)
    # 取子图
    sub_graph = snap.GetSubGraph(G, node_vector)
    new_node_id = sub_graph.GetNodes()
    # 计算网络直径
    if new_node_id > 0:
        d = snap.GetBfsFullDiam(sub_graph, test_dis, True)
        save()
