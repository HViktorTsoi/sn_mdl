import networkx as nx
import matplotlib.pyplot as plt
import random
import math

# 常量定义
Is = 'Is'
Os = 'Os'
S = 'S'


def log_status(obj, enable=False):
    if enable:
        print(obj)


def init_network(init_graph_size, init_weight):
    # 生成初始的完全图网络
    G = nx.DiGraph()
    init_edge_list = []
    for n in range(init_graph_size):
        G.add_node(n, Os=0, Is=0)
    for f in range(init_graph_size):
        init_edge_list += [
            (f, t, {'w': init_weight}) for t in range(init_graph_size) if t != f
        ]
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


def draw_graph(G, width=18, height=9):
    fig = plt.gcf()
    fig.set_size_inches(width, height)
    nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=True)
    # nx.draw_networkx(G, pos=nx.shell_layout(G), with_labels=True)
    # plt.show()
    plt.savefig('out_img/graph.png')


def add_new_edge(G, f, t, weight=1):
    G.add_edge(f, t, w=weight)

    # optional
    # G.add_edge(t, f, w=weight)

    update_node_power(G, f)
    update_node_power(G, t)


def update_node_power(G, node_id):
    G.nodes[node_id][Is] = G.nodes[node_id][Os] = 0
    for edge in G.out_edges(node_id, data=True):
        G.nodes[node_id][Os] += edge[2]['w']
    for edge in G.in_edges(node_id, data=True):
        G.nodes[node_id][Is] += edge[2]['w']
    G.node[node_id][S] = G.node[node_id][Is] + G.node[node_id][Os]


def add_new_node(G):
    # 加入一个新节点
    new_node_id = G.number_of_nodes()
    # 总出度
    sum_Os = sum([node[1][Os] for node in G.nodes(data=True)])
    # 选择一个[0,sum_Os)之间的随机数
    rnd = random.uniform(0, sum_Os)
    log_status("\n新加节点rnd={}".format(rnd))
    for origin_mtwk_node_id, info in G.nodes(data=True):
        # 如果当前节点被选中
        if rnd < info[Os]:
            # 添加新节点
            G.add_node(n=new_node_id, Os=0, Is=0)
            log_status("当前新加的节点i: {}".format(new_node_id), enable=True)
            # 加边,并更新新旧节点的出、入势和势
            log_status("新节点选择建立连接的第一个节点: {}".format(origin_mtwk_node_id))
            add_new_edge(G, origin_mtwk_node_id, new_node_id)
            add_new_edge(G, new_node_id, origin_mtwk_node_id)
            # 结束
            break
        else:
            rnd -= info[Os]
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


def choose_node(node_id_list, new_node_id):
    # draw_graph(G, 18, 9)
    sum_power = sum([G.nodes[node_id][S] for node_id in node_id_list])
    # 随机选择
    rnd = random.uniform(0, sum_power)
    for node_id in node_id_list:
        if rnd < G.nodes[node_id][S]:
            log_status("选中了节点: {}".format(node_id))
            return node_id
        else:
            rnd -= G.nodes[node_id][S]


if __name__ == '__main__':
    G = init_network(init_graph_size=5, init_weight=1)
    delta_origin = 20
    max_ntwk_size = 3000
    while G.number_of_nodes() < max_ntwk_size:
        # 添加一个节点，并且按照出势选择网络中的一个节点进行连接
        new_node_id = add_new_node(G)
        # 恢复delta的值
        delta = delta_origin
        while not (
                len(BSet(G, new_node_id).union(Set(G, new_node_id))) == G.number_of_nodes() - 1
                or delta == 1
        ):
            log_status("δ={} 此时Set{}: {}".format(delta, new_node_id, Set(G, new_node_id)))
            p = float(len(Set(G, new_node_id)) + 1)
            # 以 1/p 或者1-1/p为概率添加结点
            if roll(1 / p):
                log_status("从非邻接结点里选")
                # 从非 BSeti∪Seti 的节点中按概率选
                chosen_origin_ntwk_node_id = choose_node(
                    set(G.nodes(data=False)).difference(
                        BSet(G, new_node_id).union(Set(G, new_node_id)).union([new_node_id])
                    ),
                    new_node_id
                )
                # 选中后添加边，并更新节点的势
                add_new_edge(G, chosen_origin_ntwk_node_id, new_node_id)
            else:
                log_status("从邻接节点里选")
                # 从 BSeti 中选择节点 l
                chosen_origin_ntwk_node_id = choose_node(
                    BSet(G, new_node_id),
                    new_node_id
                )
                # 加边
                add_new_edge(G, chosen_origin_ntwk_node_id, new_node_id)
                # 求Setl∩Seti
                connected_nodes_set = Set(G, chosen_origin_ntwk_node_id) \
                    .intersection(Set(G, new_node_id))
                log_status("Setl∩Seti: {}".format(connected_nodes_set))
                # 更新 Setl∩Seti 中节点的权值
                for node_id in connected_nodes_set:
                    l = G.nodes[chosen_origin_ntwk_node_id]
                    g = G.nodes[node_id]
                    try:
                        # 更新边权值
                        if G[chosen_origin_ntwk_node_id].get(node_id):
                            G[chosen_origin_ntwk_node_id][node_id]['w'] \
                                += (l[Os] / l[S])
                            # G[chosen_origin_ntwk_node_id][node_id]['w']=math.sqrt(G[chosen_origin_ntwk_node_id][node_id]['w'])
                        if G[node_id].get(chosen_origin_ntwk_node_id):
                            G[node_id][chosen_origin_ntwk_node_id]['w'] \
                                += (l[Is] / l[S])
                            # G[node_id][chosen_origin_ntwk_node_id]['w']=math.sqrt(G[node_id][chosen_origin_ntwk_node_id]['w'])
                        update_node_power(G, chosen_origin_ntwk_node_id)
                        update_node_power(G, node_id)
                    except Exception as e:
                        log_status(G[chosen_origin_ntwk_node_id])
                        log_status(G[node_id])
                        raise e
                    # 更新节点权值
            delta -= 1
    nodes = G.nodes(data=True)
    log_status(sorted(nodes, key=lambda n: n[1][Os], reverse=True), enable=True)
    log_status(sorted(nodes, key=lambda n: n[1][Is], reverse=True), enable=True)
    log_status(sorted(nodes, key=lambda n: n[1][S], reverse=True), enable=True)
    edges = G.edges(data=True)
    log_status(sorted(edges, key=lambda e: e[2]['w'], reverse=True), enable=True)
    in_degrees = G.in_degree(G.nodes)
    log_status(sorted(in_degrees, key=lambda d: d[1], reverse=True), enable=True)
    out_degrees = G.out_degree(G.nodes)
    log_status(sorted(out_degrees, key=lambda d: d[1], reverse=True), enable=True)

    # 求节点入度分布列表
    dist = dict()
    for nid, d in in_degrees:
        dist[d] = 0
    for nid, d in in_degrees:
        dist[d] += 1
    dist = sorted(dist.items(), key=lambda d: d[0], reverse=False)
    log_status(dist, enable=True)
    plt.scatter([math.log10(d[0]) for d in dist], [math.log10(d[1]) for d in dist])
    # plt.scatter([(d[0]) for d in dist], [(d[1]) for d in dist])
    plt.show()

    # 求节点出度分布列表
    dist = dict()
    for nid, d in out_degrees:
        dist[d] = 0
    for nid, d in out_degrees:
        dist[d] += 1
    dist = sorted(dist.items(), key=lambda d: d[0], reverse=False)
    log_status(dist, enable=True)
    plt.scatter([math.log10(d[0]) for d in dist], [math.log10(d[1]) for d in dist])
    # plt.scatter([(d[0]) for d in dist], [(d[1]) for d in dist])
    plt.show()
    # draw_graph(G, 150, 75)
# log_status(G.nodes(data=True))
# log_status(G.edges(data=True))
