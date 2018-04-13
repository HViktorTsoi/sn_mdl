import csv
import json
import math
from builtins import len

import numpy as np
import matplotlib.pyplot as plt
from clyent import color
from docutils.nodes import label, comment
from scipy.optimize import curve_fit
from scipy import stats
import networkx as nx
import networkx.algorithms as nxalgo
from networkx.algorithms import community

import Analyser.utils as utils
from Defination import *

import random
from collections import defaultdict
import pickle

# 全局变量
# 控制是否显示拟合曲线
FIT_CURVE = True


def lim_range(lim):
    # 标注以及坐标轴大小限制
    if lim:
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])


def draw_ticks(custom=None, direction='n'):
    # 设置自定tick
    if custom:
        plt.xticks(custom[0], custom[1])
        plt.yticks(custom[2], custom[3])
        pass
    else:
        plt.xticks([0, 1, 2, 3, 4, 5], ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$'])
        if direction == 'n':
            plt.yticks([-7, -6, -5, -4, -3, -2, -1, 0], ['$10^{-7}$', '$10^{-6}$', '$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^0$'])
        elif direction == 'p':
            plt.yticks([4, 3, 2, 1, 0], ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^0$'])


def draw_graph(G: nx.Graph, save_path: str = '', width: int = 18, height: int = 9, target: str = 's'):
    fig = plt.gcf()
    fig.set_size_inches(width, height)
    nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=True)
    # nx.draw_networkx(G, pos=nx.shell_layout(G), with_labels=True)
    if target == 's':
        plt.show()
    elif target == 'f':
        plt.savefig('%s/graph.png' % save_path)


@utils.destroy
def draw_degree_dist(
        degrees_list,
        lbl: str = "",
        save_path: str = "",
        hold: bool = True,
        process=lambda v: np.log10(v),
        lim=(0, 3.5, 0, 3.5),
        fit_func=None,
        fit_range=(0.1, 2, 0.3),
        fit_curve_range1=None,
        fit_func2=None,
        fit_range2=None,
        fit_curve_range2=None,
        G=None,
        ticks=None,
        style=None,
        **kwargs
):
    # 获取分布
    dist = {}
    for nid, d in degrees_list:
        dist[d] = dist.get(d, 0) + 1
    dist = process(np.array([
        (x, y / (G.number_of_nodes() if G else 1))
        for x, y in sorted(dist.items()) if x != 0]))
    # 画原始分布图
    plt.scatter(
        dist[:, 0],
        dist[:, 1],
        color=style[0] if style else 'g',
        marker=style[1] if style else 'x',
        label=lbl,
    )
    plt.legend(loc="upper right")
    plt.xlabel('节点度值')
    plt.ylabel('概率', labelpad=-4)
    if FIT_CURVE:
        # 曲线拟合
        if fit_func:
            fit_func(dist, fit_range, fit_curve_range1)
        # 第二段曲线拟合
        if fit_func2:
            fit_func(dist, fit_range2, fit_curve_range2)
    if ticks:
        draw_ticks(custom=ticks)
    else:
        draw_ticks()
    lim_range(lim)
    if not hold:
        if save_path:
            plt.savefig("%s/%s.png" % (save_path, lbl))
        plt.show()
    return dist


def linear_log_fit(dist, fit_range=None, fit_curve_range=None, lim=None, **kwargs):
    # 选择曲线拟合的数据范围
    dist = np.array(
        [(x, y) for x, y in dist
         if fit_range[0] <= x <= fit_range[1] and y
         ] if fit_range
        else
        dist
    )
    # 拟合的函数
    linear_fit_func = lambda x, a, b: a * x + b
    v, g = curve_fit(linear_fit_func, dist[:, 0], dist[:, 1])
    print(v)
    # 画拟合曲线
    if fit_curve_range:
        x = np.arange(fit_curve_range[0], fit_curve_range[1], 0.1)
    else:
        x = np.arange(0, lim[1] * 2 if lim else 100, 0.1)
    plt.plot(x, linear_fit_func(x, v[0] + (fit_range[3] if len(fit_range) > 3 else 0), v[1] + fit_range[2]), color='k', linestyle='--', linewidth=1)


def poisson_fit(dist, *args):
    x = dist[:, 0]
    plt.plot(x, stats.poisson.pmf(x, 1000), color='k', linestyle='--')


def norm_fit(dist, *args):
    x = dist[:, 0]
    plt.plot(x, stats.norm.pdf(x, 600, 10), color='k', linestyle='--')


def const_fit(dist, **kwargs):
    x = dist[:, 0]
    plt.plot((0.97 * min(x), 1.03 * max(x)), (kwargs['p'], kwargs['p']), color='k', linestyle='--', linewidth=1)


def draw_power_dist(
        node_list: list,
        type: str = None, fit_range=None,
        lbl: str = "", hold: bool = True, style=None
):
    dist = {}
    for node in node_list:
        power = int(node[1][type])
        if power:
            dist[power] = dist.get(power, 0) + 1
    dist = np.array(list(dist.items()))
    dist = np.log10(dist)
    plt.scatter(
        dist[:, 0],
        dist[:, 1],
        label=lbl,
        color=style[0],
        marker=style[1],
        # s=style[2]
    )
    # 曲线拟合
    if fit_range:
        linear_log_fit(dist, fit_range=fit_range)
    plt.xlabel('节点势值')
    plt.legend(loc="upper right")
    draw_ticks(direction='p')
    plt.xlim(0, 3)
    plt.ylim(0, 3.5)
    if not hold:
        plt.show()


def analyse_power_degree_relation(G, save_path):
    lim = (20000, 20000)
    param_list = [(G.in_degree, Is, 'g', '入'), (G.out_degree, Os, 'b', '出')]
    for idx, param in enumerate(param_list):
        dist = []
        for node_id in G.nodes:
            dist.append((
                param[0](node_id),
                G.nodes[node_id][param[1]]
            ))
        dist = np.array(dist)
        plt.subplot(2, 1, idx + 1)
        # 画度-势相关图
        plt.scatter(dist[:, 0], dist[:, 1], color=param[2])
        # 线性拟合
        k = np.polyfit(dist[:, 0], dist[:, 1], 1)[0]
        print(k)
        # 画拟合曲线
        plt.plot([0, lim[0]], [0, k * lim[0]], color='k', linestyle='--', linewidth=1)
        plt.xscale('symlog')
        plt.yscale('symlog')
        plt.xlabel('节点%s度' % (param[3]))
        plt.ylabel('节点信息传播%s势' % (param[3]))
        plt.xlim(0, lim[0])
        plt.ylim(0, lim[1])
    plt.gcf().tight_layout()
    plt.savefig('%s/节点度势相关性.png' % save_path)


def analyze_degree_power(G, save_path):
    fig = plt.gcf()
    fig.set_size_inches(30, 16)
    # 画节点入度分布
    plt.subplot(2, 3, 1)
    draw_degree_dist(list(G.in_degree), lbl="in degree")
    # 画节点出度分布
    plt.subplot(2, 3, 2)
    draw_degree_dist(list(G.out_degree), lbl="out degree")
    # 画节点度分布
    plt.subplot(2, 3, 3)
    draw_degree_dist(list(G.degree), "degree")

    # 画节点入势分布
    plt.subplot(2, 3, 4)
    draw_power_dist(G.nodes(data=True), type=Is, lbl='in power', style=['k', 'x', 10])
    # 画节点出势分布
    plt.subplot(2, 3, 5)
    draw_power_dist(G.nodes(data=True), type=Os, lbl='out power', style=['k', 'x', 10])
    # 画节点势分布
    plt.subplot(2, 3, 6)
    draw_power_dist(G.nodes(data=True), type=S, lbl='power', style=['k', 'x', 10])

    plt.savefig('%s/degree_and_power.png' % save_path)
    plt.show()


@utils.destroy
def analyse_clustering_coefficient(
        G: nx.Graph, save_path='', lim=None,
        sample_count=500,
        process=lambda v: np.log10(v),
        fit_func=None, fit_range=None, fit_curve_range=None,
        ticks=None, p=None, hold=True, **kwargs
):
    Gud = G.to_undirected() if G.is_directed() else G
    # 对网络中节点取样
    nodes_list = random.sample(Gud.nodes, sample_count)
    # 求聚集系数分布数据
    print('计算聚集系数')
    cc_dist = np.array([(Gud.degree(node_id), cc)
                        for node_id, cc in nxalgo.clustering(Gud, nodes_list).items() if cc > 0])
    # 计算原始数据平均聚集系数 因子为网络中总节点数
    ave_cc = sum(cc_dist[:, 1]) / sample_count
    print('平均聚集系数: %r' % ave_cc)
    # 取双对数坐标
    cc_dist = process(cc_dist)
    plt.scatter(cc_dist[:, 0], cc_dist[:, 1], marker='o', color='b', s=5, label='聚集系数分布')
    # 绘制处理后的平均聚集系数
    ave_cc_proccessed = process(ave_cc)
    plt.hlines(ave_cc_proccessed, lim[0], lim[1], colors='grey', linestyles='--', label='平均聚集系数')
    # 拟合曲线
    if fit_func:
        fit_func(cc_dist, fit_range=fit_range, fit_curve_range=fit_curve_range, lim=lim, p=p)
    plt.xlabel('节点度值')
    plt.ylabel('系数', labelpad=-4)
    plt.legend()
    if ticks:
        draw_ticks(custom=ticks)
    else:
        draw_ticks()
    lim_range(lim)
    if not hold:
        plt.show()
        if save_path:
            plt.savefig('%s/聚集系数分布.png' % (save_path))


@utils.destroy
def analyse_gini_coefficient(G: nx.Graph, calc_type=None, hold=True, lbl='', save_path=''):
    N = G.number_of_nodes()
    # 根据类型的不同选择出度、入度或者度
    print('当前正在计算%s' % (calc_type if calc_type else 'all'))
    raw_degree = G.degree if calc_type is None \
        else G.in_degree if calc_type == 'in' \
        else G.out_degree if calc_type == 'out' else None
    linestyle_dict = {None: '-', 'in': '-.', 'out': ':'}
    # 总度数
    print('计算总度数')
    sum_k = sum([raw_degree[node_id] for node_id in G.nodes])
    # 经过排序的节点度列表
    node_list = sorted(raw_degree, key=lambda n: n[1])
    # 计算洛伦兹曲线数据
    print('计算曲线数据')
    s = 0
    lorentz_curve_data = []
    for idx, node in enumerate(node_list):
        x = idx / N
        s += node[1] / sum_k
        lorentz_curve_data.append((x, s,))
    lorentz_curve_data = np.array(lorentz_curve_data)
    # 用微分计算基尼系数
    gini_coefficient = sum(v[0] - v[1] for v in lorentz_curve_data) / (N / 2 + 0.5)
    print('基尼系数:%f' % (gini_coefficient))
    plt.plot(lorentz_curve_data[:, 0], lorentz_curve_data[:, 1], label=lbl, linestyle=linestyle_dict[calc_type])
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    plt.vlines(1, 0, 1, colors='gray', linestyles='--', linewidths=1)
    plt.hlines(1, 0, 1, colors='gray', linestyles='--', linewidths=1)
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    # 面积标记
    # plt.text(0.8, 0.7, 'α')
    # plt.text(0.8, 0.1, 'β')
    plt.xlabel('累计节点数比')
    plt.ylabel('累计节点度值比')
    plt.legend(loc='upper left')
    if not hold:
        plt.show()
        if save_path:
            plt.savefig('%s/%s.png' % (save_path, lbl))


def analyse_core(G):
    core_list = nx.core_number(G)
    dist = np.array(list({G.degree[node_id]: core_list[node_id] for node_id in core_list.keys()}.items()))
    print(dist)
    plt.scatter(dist[:, 0], dist[:, 1])
    plt.xscale('symlog')
    plt.xlabel('节点度值')
    plt.ylabel('节点核度')
    plt.show()


def analyse_universe_param(G):
    # 分析网络聚集系数
    # analyse_clustering_coefficient(G)

    # 网络直径
    # print('网络直径: %d' % nx.diameter(G.to_undirected()))

    # 绘制洛伦兹曲线
    analyse_gini_coefficient(G)
    # analyse_gini_coefficient(G, type='in')
    # analyse_gini_coefficient(G, type='out')


def analyse(G: nx.Graph, save_path: str):
    # 分析网络出入度 出入势
    analyze_degree_power(G, save_path)

    # 分析网络通用的拓扑结构参数
    analyse_universe_param(G)


def save_graph(G, save_path):
    nx.write_gpickle(G, save_path + '/graph.gpickle')


def load_graph(save_path, graph_file='graph.gpickle'):
    return nx.read_gpickle(save_path + '/' + graph_file)


def load_from_csv(path):
    G = nx.DiGraph()
    with open('%s/nodes.csv' % path, 'r') as file:
        f_csv = csv.DictReader(file)
        field_types = [
            ('nid', int), (Is, float), (Os, float), (S, float), (V, float)
        ]
        for row in f_csv:
            # 转换类型
            row.update((key, conv(row[key])) for key, conv in field_types)
            row[D] = json.loads(row[D])
            node_id = row['nid']
            del row['nid']
            # 添加节点
            G.add_node(node_id, **row)
    with open('%s/edges.csv' % path, 'r') as file:
        f_csv = csv.DictReader(file)
        field_types = [
            ('from', int), ('to', int), ('w', float)
        ]
        for row in f_csv:
            row.update((key, conv(row[key])) for key, conv in field_types)
            G.add_edge(row['from'], row['to'], w=row['w'])
    return G


def analyse_communities(G, k=4):
    def calc_domain_suitability(a1, a2):
        k = len(a1)
        return sum(
            [a1[idx] == a2[idx] == 1 for idx in range(k)]
        ) / float(k)

    G_ud = nx.Graph()
    for edge in G.edges:
        f, t = edge[0], edge[1]
        if not G_ud.has_edge(f, t):
            factor = calc_domain_suitability(
                G.nodes[f][D], G.nodes[t][D]
            )
            G_ud.add_edge(f, t, suit=factor)
            print(edge)
    cms = community.k_clique_communities(G_ud, k)
    id = 0
    for cm in cms:
        nbr = set()
        for node_id in cm:
            nbr = nbr | set(G.successors(node_id))
        nbr = nbr - cm
        print(nbr)
        plt.gcf().set_size_inches(50, 25)
        pos = nx.spring_layout(G_ud.subgraph(cm | nbr))
        colors = [edge[2]['suit'] for edge in G_ud.subgraph(cm | nbr).edges(data=True)]
        nx.draw_networkx(
            G_ud.subgraph(cm | nbr), edge_cmap=plt.cm.get_cmap('RdYlBu'),
            edge_color=colors, pos=pos, width=0.5, node_size=50)
        plt.savefig('/tmp/tmp.png')
        # id += 1
        break


def analyse_evolution_status(G, save_path):
    diameter_list = {}
    cc_list = {}
    edge_list = {}
    node_list = list(G.nodes(data=False))
    step = [*range(1, 30), *range(30, 70, 2), 80, 90, 100, 200, 400, 800, 1000, 2000, *range(3000, 20000, 4000)]
    step = [*range(1, 1000, 2), *range(2000, 10000, 100)]
    for cnt in range(len(node_list)):
        # 指定间隔步数
        if step and cnt == step[0]:
            step.pop(0)
            # 计算演化到此时的网络直径、平均聚集系数、边数
            G_ud = G.subgraph(node_list[:cnt]).to_undirected()
            new_node_id = node_list[cnt]
            # d = nx.diameter(G_ud)
            # diameter_list[new_node_id] = d
            # print("直径： %r" % diameter_list)
            cc_list[new_node_id] = nx.average_clustering(G_ud)
            print("聚集系数： %r" % cc_list)
            # edge_list[new_node_id] = G_ud.number_of_edges()
        # 保存结果
        with open("%s/cc_evo_data.json" % save_path, 'w') as file:
            file.write(json.dumps({
                "dia": diameter_list,
                "cc": cc_list,
                "edge": edge_list
            }))


def analyse_edge_activity_map(G, ctof, save_path):
    # 变换方式
    def convert(v, cutoff=ctof):
        v = math.log10((v - 1)) + cutoff
        return v if v > 0 else 0

    # 计算各条边的活跃度分布
    dist = np.array([(edge[0], edge[1], convert(edge[2]['w'])) for edge in G.edges(data=True)
                     if edge[2]['w'] > 1])
    plt.plot(range(len(dist)), sorted(dist[:, 2]))
    plt.show()
    # 绘制活跃度分布点
    sc = plt.scatter(dist[:, 0], dist[:, 1], s=10, marker='v',
                     c=dist[:, 2] / max(dist[:, 2]),
                     cmap=plt.cm.get_cmap('RdYlBu' + '_r'))
    # 加入colorbar
    plt.colorbar(sc)
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.xlabel('节点标识n')
    plt.ylabel('节点标识n')
    plt.savefig('%s/%s.png' % (save_path, '边活跃度热力图'))
    plt.show()


def analyse_attach_trend(G):
    def draw_domian_attach(slt_value_domain):
        slt_value_domain = np.array(slt_value_domain)
        plt.gcf().set_size_inches(6.4 * 2, 4.8)
        # 画出相对分布图
        plt.subplot(121)
        x = np.arange(len(slt_value_domain))
        plt.scatter(x, slt_value_domain[:, 1], s=8, marker='x', color='g', label='被选择节点匹配度')
        plt.scatter(x, slt_value_domain[:, 0], s=5, label='平均匹配度')
        plt.ylim(0, 1.1)
        plt.xlabel('演化时间(边数)\n(a)', linespacing=2)
        plt.ylabel('领域匹配度')
        plt.legend()
        # 画出对比分布图
        plt.subplot(122)
        plt.scatter(slt_value_domain[:, 0], slt_value_domain[:, 1], s=5)
        plt.plot([0, 1], [0, 1], color='k', linestyle='--')
        plt.ylim(0, 1.1)
        plt.xlabel('平均匹配度\n(b)', linespacing=2)
        plt.ylabel('被选择节点匹配度')
        plt.gcf().tight_layout()
        plt.savefig(path + '/匹配度依附趋势.png')

    with open('/tmp/rst.pkl', 'rb') as pkl_file:
        dist = pickle.load(pkl_file)
        # 记录选择节点的值
        slt_value_domain = []
        for choose_process in dist:
            cur_node_id = choose_process['cur']
            adj_suit = choose_process['adj_suit']
            chosen_node_id = choose_process['chosen']
            # 计算平均值
            ave = sum(adj_suit.values()) / len(adj_suit)
            slt_value_domain.append((ave, adj_suit[chosen_node_id]))
            # 对势计数
            # slt_value_power[(G.nodes[chosen_node_id][S])] += 1
        # 画领域依附图
        draw_domian_attach(slt_value_domain)


def nme_gini_example_ticks():
    # 标注
    x1, y1 = 0.5, 0.13
    plt.hlines(y1, 0, x1, linestyles='--', linewidths=1)
    plt.vlines(x1, 0, y1, linestyles='--', linewidths=1)
    plt.annotate(r'$(\frac{i}{N},S_i)$', xy=(x1, y1), xytext=(0.4, 0.3),
                 arrowprops=dict(arrowstyle='->'))
    x2, y2 = 0.6, 0.19
    plt.hlines(y2, 0, x2, linestyles='--', linewidths=1)
    plt.vlines(x2, 0, y2, linestyles='--', linewidths=1)
    plt.annotate(r'$(\frac{i+1}{N},S_{i+1})$', xy=(x2, y2), xytext=(0.6, 0.4),
                 arrowprops=dict(arrowstyle='->'))
    plt.text(1.01, 0.01, 'C')
    plt.text(x1, y1 + 0.01, 'D')
    plt.text(x2 + 0.01, y2 - 0.05, 'E')
    plt.text(1, 1, 'F')
    plt.xticks([0, 1], ['O', 1])
    plt.yticks([1], [1])
    plt.savefig('../data/stable/基尼系数样例.png')
    plt.show()


@utils.destroy
def NME(existed=None, types=(3,), isHold=True, path='data/stable/NME_2', lbl=None, style=None, **kwargs):
    G = load_from_csv(path=path)
    '''分析NME模型'''
    # 类型
    for type in types:
        if type == 0:
            common_lim = 0, 3, -4.3, 0
            # 度数分布
            draw_degree_dist(
                G.in_degree, hold=isHold, lbl='入度分布',
                fit_func2=linear_log_fit, fit_range2=(0, 0.7, 0.30),
                fit_curve_range2=(0, 1.5),
                fit_func=linear_log_fit, fit_range=(0.5, 1.5, 0.5),
                fit_curve_range1=(0.6, 3),
                save_path=path,
                lim=common_lim,
                G=G
            )
            draw_degree_dist(
                G.out_degree, hold=isHold, lbl='出度分布',
                fit_func2=linear_log_fit, fit_range2=(0, 0.65, 0.25),
                fit_curve_range2=(0, 1.6),
                fit_func=linear_log_fit, fit_range=(0.4, 1.8, 0.6),
                fit_curve_range1=(0.6, 3),
                save_path=path,
                lim=common_lim,
                G=G
            )
            # draw_degree_dist(
            #     G.degree, hold=isHold, lbl=lbl if lbl else '度分布',
            #     fit_func2=linear_log_fit, fit_range2=(0, 0.7, 0.30),
            #     fit_curve_range2=(0, 1.7),
            #     fit_func=linear_log_fit, fit_range=(0.7, 2, 0.6),
            #     fit_curve_range1=(0.8, 3),
            #     save_path=path,
            #     lim=common_lim,
            #     style=style,
            #     G=G
            # )
        elif type == 1:
            analyse_clustering_coefficient(
                G, save_path=path, hold=isHold,
                fit_func=linear_log_fit, fit_range=(1, 3, 0),
                lim=(0, 3, -3.3, 0),
                sample_count=10000,
                ticks=None)
        elif type == 2:
            plt.subplot(1, 2, 1)
            draw_power_dist(G.nodes(data=True), type=Is, fit_range=(0.3, 1.5, 0.1), lbl='节点入势', style=('g', 'x', 10))
            plt.ylabel('概率')
            plt.subplot(1, 2, 2)
            draw_power_dist(G.nodes(data=True), type=Os, fit_range=(0.6, 1.9, 0.05), lbl='节点出势', style=('b', 'v', 10))
            if path:
                plt.savefig('%s/%s.png' % (path, '节点势分布'))
            plt.show()
        elif type == 3:
            analyse_gini_coefficient(G, save_path=path, lbl='度-洛伦兹曲线', hold=isHold)
            analyse_gini_coefficient(G, calc_type='in', save_path=path, lbl='入度-洛伦兹曲线', hold=isHold)
            analyse_gini_coefficient(G, calc_type='out', save_path=path, lbl='出度-洛伦兹曲线', hold=isHold)
            # nme_gini_example_ticks()
        elif type == 4:
            analyse_core(G)
        elif type == 5:
            analyse_attach_trend(G)
        elif type == 6:
            analyse_communities(G)
        elif type == 7:
            # 演化过程系数
            analyse_evolution_status(G, save_path=path)
        elif type == 8:
            from collections import defaultdict

            v_list = [node[1][V] for node in G.nodes(data=True)]
            dist = defaultdict(int)
            for v in v_list:
                dist[round(v, 2)] += 1
            dist = np.array(list(dist.items()))
            plt.scatter(dist[:, 0], dist[:, 1])
            plt.show()
        elif type == 9:
            analyse_edge_activity_map(G, ctof=2.6, save_path=path)
        elif type == 10:
            analyse_power_degree_relation(G, save_path=path)
        elif type == 11:
            FIT_CURVE = False
            NME(types=[0], isHold=True, path='../data/20180129_125317_n20000_e74780_delta10_k10', style='go', lbl='$\epsilon=10$')
            NME(types=[0], isHold=True, path='../data/20180202_002604_n20000_e110320_30_k10', style='b^', lbl='$\epsilon=30$')
            NME(types=[0], isHold=True, path='../data/20180131_122403_n20000_e39998_d1', style='rx', lbl='$\epsilon=1$')
            plt.savefig('%s/度随参数分布.png' % path)
        elif type == 12:
            # G = G.subgraph(nodes=random.sample(G.nodes, 2000))
            print(G.nodes)
            plt.gcf().set_size_inches(20, 10)
            nx.draw_networkx(G, nx.spring_layout(G), width=0.5, node_size=50, font_size=5)
            plt.savefig('/tmp/network.png')


# 单元测试
if __name__ == '__main__':
    # 载入网络数据
    # 效果比较好/stable/Nme
    # path = '../data/20180202_002604_n20000_e110320_30_k10'
    path = '../data/stable/NME_2'
    # path = '../data/20180128_172813_n6000_e27482_excellnt_不根据_20'
    NME(types=[12], isHold=True, path=path)
