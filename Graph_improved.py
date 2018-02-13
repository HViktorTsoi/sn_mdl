# coding: utf-8
"""
改进的算法
"""
import networkx as nx
import random
import math
import os
import time
import csv
import json
from collections import defaultdict
import pickle

import Analyser.analyser as analyser
from Defination import *


def log_status(obj, enable=False):
    if enable:
        print(obj)


def init_network(init_graph_size, init_weight, k):
    # 生成初始的完全图网络
    G = nx.DiGraph()
    init_edge_list = []
    for n in range(init_graph_size):
        # G.add_node(n, Os=0, Is=0, V=random.random(), D=generate_domin_vector(k))
        G.add_node(
            n, Os=0, Is=0, V=random.random(),
            D=[1 if (n == _) or (n + 1 == _) else 0 for _ in range(k)]
        )
    for f in range(init_graph_size - 1):
        # init_edge_list += [
        #     (f, t, {'w': init_weight}) for t in range(init_graph_size) if t != f
        # ]
        init_edge_list += [(f, f + 1, {'w': init_weight}),
                           (f + 1, f, {'w': init_weight})]

    G.add_edges_from(init_edge_list)

    # 初始化网络的出势和入势
    for node in G.nodes(data=False):
        # 出势
        out_power = 0
        for out_edge in G.out_edges(nbunch=node, data=True):
            out_power += out_edge[2]['w']
        G.nodes[node][Os] = out_power
        # 入势
        in_power = 0
        for in_edge in G.in_edges(nbunch=node, data=True):
            in_power += in_edge[2]['w']
        G.nodes[node][Is] = in_power
        # 势
        update_node_power(G, node_id=node)
    return G


def add_new_edge(G, f, t, weight=1):
    G.add_edge(f, t, w=weight)
    if param_delta == 1:
        G.add_edge(t, f, w=weight)
    update_node_power(G, f)
    update_node_power(G, t)


def update_node_power(G, node_id):
    # 求这个节点新的势
    sum_out_power = sum_in_power = 0
    for edge in G.out_edges(node_id, data=True):
        sum_out_power += edge[2]['w']
    for edge in G.in_edges(node_id, data=True):
        sum_in_power += edge[2]['w']
    # 更新节点势
    G.nodes[node_id][Os] = sum_out_power
    G.nodes[node_id][Is] = sum_in_power
    G.node[node_id][S] = sum_out_power + sum_in_power


def add_new_node(G, k):
    # 加入一个新节点
    new_node_id = G.number_of_nodes()

    # 添加新节点
    G.add_node(node_for_adding=new_node_id, Os=0, Is=0, V=random.random(), D=generate_domin_vector(k))

    # 从原网络中不包含新节点自身自身的节点中选一个建立连接
    origin_mtwk_node_id = choose_node(
        G,
        G.nodes,
        new_node_id,
        type=Is
    )
    log_status("当前新加的节点i: {}".format(new_node_id), enable=True)
    # 加边,并更新新旧节点的出、入势和势
    log_status("新节点选择建立连接的第一个节点: {}".format(origin_mtwk_node_id))
    add_new_edge(G, new_node_id, origin_mtwk_node_id)
    # add_new_edge(G, origin_mtwk_node_id, new_node_id)

    return new_node_id


def BSet(G, node_id):
    # 先求邻接节点Set
    _set = Set(G, node_id)
    b_set = set()
    for nid in _set:
        b_set = b_set.union(Set(G, nid))
    # BSeti = BSeti − (Seti ∪ {i})
    b_set = b_set.difference(_set.union([node_id]))
    return b_set


def Set(G, node_id):
    return set(G.successors(node_id)).union(set(G.predecessors(node_id)))


def roll(p):
    log_status("概率 1/|Seti+1|: {}".format(p))
    return True if random.random() < p else False


def calc_domain_suitability(a1, a2):
    k = len(a1)
    return sum(
        [a1[idx] == a2[idx] == 1 for idx in range(k)]
    ) / float(k)


def generate_domin_vector(k):
    return [int(random.random() < 0.5) for _ in range(k)]


def choose_node(G, node_id_list, new_node_id, type):
    # draw_graph(G, 18, 9)
    # by v
    nodes = dict(G.nodes(data=True))

    # 求 势*活跃度*领域匹配度 的和值
    '''nodes[node_id][type] * '''
    sum_power = sum(
        [nodes[node_id][type] * calc_domain_suitability(
            nodes[new_node_id][D],
            nodes[node_id][D]
        )
         for node_id in node_id_list if node_id != new_node_id]
    )

    # 随机选择
    rnd = random.uniform(0, sum_power)
    for node_id in node_id_list:
        # 舍弃node_id等于新节点自身的情况（初始新加节点时会出现这种情况）
        if node_id == new_node_id:
            continue
        choose_factor = nodes[node_id][type] * calc_domain_suitability(
            nodes[new_node_id][D],
            nodes[node_id][D]
        )
        if rnd <= choose_factor:
            log_status("选中了节点: {}".format(node_id))
            # 记录选边信息
            # dist.append({
            #     'cur': new_node_id,
            #     'adj_suit': {node_id: calc_domain_suitability(nodes[new_node_id][D], nodes[node_id][D])
            #                  for node_id in node_id_list if node_id != new_node_id},
            #     'chosen': node_id
            # })
            return node_id
        else:
            rnd -= choose_factor


def save_graph(G, info):
    save_path = "data/%s_n%d_e%d_%s" % (
        time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())),
        G.number_of_nodes(),
        G.number_of_edges(),
        info
    )
    # 创建目录
    os.mkdir(save_path)
    # 保存节点
    with open('%s/nodes.csv' % save_path, 'w') as file:
        hdr = ['nid', 'Is', 'Os', 'S', 'V', 'D']
        f_csv = csv.DictWriter(file, fieldnames=hdr)
        f_csv.writeheader()
        for node in G.nodes(data=True):
            row = node[1]
            row['nid'] = node[0]
            f_csv.writerow(row)
    # 保存边
    with open('%s/edges.csv' % save_path, 'w') as file:
        hdr = ['from', 'to', 'w']
        f_csv = csv.DictWriter(file, fieldnames=hdr)
        f_csv.writeheader()
        for edge in G.edges(data=True):
            row = edge[2]
            row['from'] = edge[0]
            row['to'] = edge[1]
            f_csv.writerow(row)
    return save_path


def start_evolution(init_graph_size, delta_origin, max_ntwk_size, k):
    G = init_network(init_graph_size=init_graph_size, init_weight=1, k=k)
    while G.number_of_nodes() < max_ntwk_size:
        # 添加一个节点，并且按照入势选择网络中的一个节点进行连接
        new_node_id = add_new_node(G, k)
        # 恢复delta的值
        delta = delta_origin
        while not (
                len(BSet(G, new_node_id).union(Set(G, new_node_id))) == G.number_of_nodes() - 1
                or delta == 1
        ):
            # 节点足够活跃 并且活跃度随时间不断下降
            decrease_factor = 0.1
            if G.nodes[new_node_id][V] - decrease_factor > 0:
                G.nodes[new_node_id][V] -= decrease_factor
            if roll(G.nodes[new_node_id][V]):
                log_status("δ={} 此时Set{}: {}".format(delta, new_node_id, Set(G, new_node_id)))
                p = float(len(Set(G, new_node_id)) + 1)
                # 以 1/p 或者1-1/p为概率添加结点
                if roll(1 / p):
                    log_status("从非邻接结点里选入势最大的")
                    # 从非 BSeti∪Seti 的节点中按概率选
                    chosen_origin_ntwk_node_id = choose_node(
                        G,
                        set(G.nodes(data=False)).difference(
                            BSet(G, new_node_id).union(Set(G, new_node_id)).union([new_node_id])
                        ),
                        new_node_id,
                        type=Is
                    )
                    # 选中后添加边，并更新节点的势
                    add_new_edge(G, new_node_id, chosen_origin_ntwk_node_id)

                else:
                    log_status("从邻接节点里选入势最大的点")
                    # 从 BSetu 中选择节点 v
                    chosen_origin_ntwk_node_id = choose_node(
                        G,
                        BSet(G, new_node_id),
                        new_node_id,
                        type=Is
                    )
                    # 加边
                    # add_new_edge(G, chosen_origin_ntwk_node_id, new_node_id)
                    add_new_edge(G, new_node_id, chosen_origin_ntwk_node_id)
                    # 求Setv∩Setu
                    connected_nodes_set = Set(G, chosen_origin_ntwk_node_id) \
                        .intersection(Set(G, new_node_id))
                    log_status("Setv∩Setu: {}".format(connected_nodes_set))

                    v = G.nodes[chosen_origin_ntwk_node_id]
                    # 更新 Setl∩Seti 中节点的权值
                    for node_id in connected_nodes_set:
                        g = G.nodes[node_id]
                        # 更新边权值
                        if G[chosen_origin_ntwk_node_id].get(node_id):
                            G[chosen_origin_ntwk_node_id][node_id]['w'] \
                                += (g[Is] / g[S]) * v[V]
                        if G[node_id].get(chosen_origin_ntwk_node_id):
                            G[node_id][chosen_origin_ntwk_node_id]['w'] \
                                += (v[Is] / v[S]) * g[V]
                        # 更新节点权值
                        update_node_power(G, node_id)
                    update_node_power(G, chosen_origin_ntwk_node_id)

                    # 选择出边
                    # 从邻接节点里选择出势最大的点
                    log_status("从邻接节点里选出势最大的点")
                    chosen_origin_ntwk_node_id = choose_node(
                        G,
                        # BSet(G, new_node_id).union(Set(G, new_node_id)),
                        BSet(G, new_node_id),
                        new_node_id,
                        type=Os
                    )
                    # 建立连接
                    add_new_edge(G, chosen_origin_ntwk_node_id, new_node_id)
                    if chosen_origin_ntwk_node_id in BSet(G, new_node_id):
                        connected_nodes_set = Set(G, chosen_origin_ntwk_node_id) \
                            .intersection(Set(G, new_node_id))
                        log_status("Setu∩Setv: {}".format(connected_nodes_set))
                        # 更新 Setu∩Setv 中节点的权值
                        u = G.nodes[new_node_id]
                        v = G.nodes[chosen_origin_ntwk_node_id]
                        for node_id in connected_nodes_set:
                            g = G.nodes[node_id]
                            # 更新边权值
                            if G[chosen_origin_ntwk_node_id].get(node_id):
                                G[chosen_origin_ntwk_node_id][node_id]['w'] \
                                    += (v[Os] / v[S]) * u[V]
                            if G[node_id].get(chosen_origin_ntwk_node_id):
                                G[node_id][chosen_origin_ntwk_node_id]['w'] \
                                    += (g[Os] / g[S]) * g[V]
                            # 更新节点权值
                            update_node_power(G, node_id)
                        update_node_power(G, chosen_origin_ntwk_node_id)
            delta -= 1
    return G


if __name__ == '__main__':
    # 存储演化过程特性
    dist = []
    param_delta = 5
    param_k = 10
    save_info = 'dta%d_%s' % (param_delta, input('演化关键信息: '))
    network_model = start_evolution(
        init_graph_size=param_k - 1,
        delta_origin=param_delta,
        max_ntwk_size=20000,
        k=param_k
    )
    # 保存图
    saved_path = save_graph(network_model, save_info)
    # 保存演化信息
    # pickle.dump(dist, file=open('/tmp/rst.pkl', 'wb'))
