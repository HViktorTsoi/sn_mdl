import json
import multiprocessing

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import Analyser.analyser as analyser
import Analyser.utils as utils
import gc

# 要进行benchmarke的拓扑结构类型
PROCESS_TYPE = 0
PROCESS_TYPE_DEGREE_TYPE = 0

TITLE_LIST = ['综合度分布', '综合聚集系数分布', '综合洛伦兹曲线', '网络拓扑图']


@utils.destroy
def er_model(existed=None, isHold=True):
    save_path = path + 'ER'
    if not existed:
        G = nx.fast_gnp_random_graph(20000, 0.05)
        analyser.save_graph(G, save_path)
    else:
        G = analyser.load_graph(save_path)
    if PROCESS_TYPE == 0:
        # 分析度分布
        analyser.draw_degree_dist(
            G.degree, process=lambda v: v,
            lim=(800, 1200, 0, 0.020),
            fit_func=analyser.poisson_fit,
            hold=isHold, lbl='度分布', save_path=save_path, G=G,
            ticks=(
                [800, 1000, 1200], ['$0.8x10^3$', '$10^3$', '$1.2x10^3$'],
                [0.001, 0.005, 0.01, 0.015, 0.02],
                ['$10^{-3}$', '$0.5x10^{-2}$', '$10^{-2}$', '$1.5x10^{-2}$', '$2x10^{-2}$']
            )
        )
    elif PROCESS_TYPE == 1:
        # 分析聚集系数
        analyser.analyse_clustering_coefficient(
            G, fit_func=analyser.const_fit, save_path=save_path,
            process=lambda v: v, lim=(800, 1200, 0.035, 0.07), p=0.05,
            sample_count=3000,
            ticks=(
                [800, 1000, 1200], ['$0.8x10^3$', '$10^3$', '$1.2x10^3$'],
                [0.03, 0.04, 0.05, 0.06, 0.07],
                ['$3x10^{-2}$', '$4x10^{-2}$', '$5x10^{-2}$', '$6x10^{-2}$', '$7x10^{-2}$']
            ),
            hold=isHold
        )
    elif PROCESS_TYPE == 2:
        analyser.analyse_gini_coefficient(G, save_path=save_path, lbl='度值', hold=True)
    elif PROCESS_TYPE == 3:
        analyser.draw_graph(G)


@utils.destroy
def ws_model(existed=None, isHold=True):
    save_path = path + 'WS'
    if not existed:
        G = nx.newman_watts_strogatz_graph(20000, 500, 0.2)
        analyser.save_graph(G, save_path)
    else:
        G = analyser.load_graph(save_path)
    if PROCESS_TYPE == 0:
        analyser.draw_degree_dist(
            G.degree, process=lambda v: v,
            lim=(500, 700, 0, 0.05),
            fit_func=analyser.norm_fit,
            hold=isHold, lbl='度分布', save_path=save_path, G=G,
            ticks=(
                [500, 600, 700], ['$0.5x10^3$', '$0.6x10^3$', '$0.7x10^3$'],
                [0.001, 0.01, 0.02, 0.03, 0.04, 0.05],
                ['$10^{-3}$', '$10^{-2}$', '$2x10^{-2}$', '$3x10^{-2}$', '$4x10^{-2}$', '$5x10^{-2}$']
            )
        )
    elif PROCESS_TYPE == 1:
        analyser.analyse_clustering_coefficient(
            G, fit_func=analyser.linear_log_fit,
            fit_range=(0, 1000, 0.003), fit_curve_range=(525, 680),
            save_path=save_path, process=lambda v: v,
            sample_count=4000,
            lim=(500, 700, 0.35, 0.7), ticks=(
                [500, 600, 700], ['$0.5x10^3$', '$0.6x10^3$', '$0.7x10^3$'],
                [0.3, 0.4, 0.5, 0.6, 0.7], ['$3x10^{-1}$', '$4x10^{-1}$', '$5x10^{-1}$', '$6x10^{-1}$', '$7x10^{-1}$']
            )
        )
    elif PROCESS_TYPE == 2:
        analyser.analyse_gini_coefficient(G, save_path=save_path, lbl='节点Degree', hold=False)
    elif PROCESS_TYPE == 3:
        analyser.draw_graph(G)


@utils.destroy
def ba_model(existed=None, isHold=True):
    save_path = path + 'BA'
    if not existed:
        G = nx.barabasi_albert_graph(20000, 5)
        analyser.save_graph(G, save_path)
    else:
        G = analyser.load_graph(save_path)
    if PROCESS_TYPE == 0:
        analyser.draw_degree_dist(
            G.degree, process=lambda v: np.log10(v),
            lim=(0.0, 3, -4.3, 0),
            fit_func=analyser.linear_log_fit,
            fit_range=(0.1, 2.2, 0.5),
            hold=isHold, lbl='度分布', save_path=save_path, G=G
        )
    elif PROCESS_TYPE == 1:
        analyser.analyse_clustering_coefficient(
            G, save_path=save_path, sample_count=7000,
            fit_func=analyser.linear_log_fit, fit_range=(0, 1.1, 0.20),
            ticks=None, lim=(0, 3, -3.3, 0))
    elif PROCESS_TYPE == 2:
        analyser.analyse_gini_coefficient(G, save_path=save_path, lbl='节点Degree', hold=False)
    elif PROCESS_TYPE == 3:
        analyser.draw_graph(G)
    elif PROCESS_TYPE == 4:
        analyser.analyse_community_evolution(G, 500)


@utils.destroy
def twitter(existed=None, isHold=True):
    save_path = path + 'TWITTER'
    G = analyser.load_graph(save_path)
    # 取子图
    G_sub = G.subgraph(sorted(list(G.nodes))[30000:50000])
    common_lim = (0, 3, -4.3, 0)
    if PROCESS_TYPE == 0:
        analyser.draw_degree_dist(
            G_sub.degree, hold=isHold, lbl='度分布',
            fit_func2=analyser.linear_log_fit, fit_range2=(0, 0.8, 0.35),
            fit_curve_range2=(0, 2),
            fit_func=analyser.linear_log_fit, fit_range=(1.2, 2.5, 0.5),
            fit_curve_range1=(0.8, 5),
            lim=common_lim,
            save_path=save_path, G=G_sub
        )
        if PROCESS_TYPE_DEGREE_TYPE == 1:
            analyser.draw_degree_dist(
                G_sub.in_degree, hold=isHold, lbl='入度分布',
                fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.30),
                fit_curve_range2=(0, 2),
                fit_func=analyser.linear_log_fit, fit_range=(1.5, 2.0, 0.5),
                fit_curve_range1=(0.8, 5),
                lim=common_lim,
                save_path=save_path, G=G_sub
            )
            analyser.draw_degree_dist(
                G_sub.out_degree, hold=isHold, lbl='出度分布',
                fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.35),
                fit_curve_range2=(0, 2),
                fit_func=analyser.linear_log_fit, fit_range=(1.2, 2.5, 0.5),
                fit_curve_range1=(0.8, 5),
                lim=common_lim,
                save_path=save_path, G=G_sub
            )
    elif PROCESS_TYPE == 1:
        analyser.analyse_clustering_coefficient(
            G_sub, save_path=save_path,
            fit_func=analyser.linear_log_fit, fit_range=(1.2, 3, -0.2),
            sample_count=15000,
            ticks=None, lim=(0, 3, -3.3, 0)
        )
    elif PROCESS_TYPE == 2:
        analyser.analyse_gini_coefficient(G_sub, save_path=save_path, lbl='度值', hold=True)
        analyser.analyse_gini_coefficient(G_sub, calc_type='out', save_path=save_path, lbl='入度', hold=True)
        analyser.analyse_gini_coefficient(G_sub, calc_type='in', save_path=save_path, lbl='出度', hold=True)
    elif PROCESS_TYPE == 4:
        # G = G.subgraph(sorted(list(G.nodes))[10000:20000])
        analyser.analyse_community_evolution(G, step=1000, path='')


@utils.destroy
def gplus(existed=None, isHold=True):
    save_path = path + 'GPlus'
    # 取子图
    # G_sub = G.subgraph(sorted(list(G.nodes))[50000:65000])
    G = analyser.load_graph(save_path, graph_file='sub_graph.gpickle')
    common_lim = (0, 3, -4.3, 0)
    if PROCESS_TYPE == 0:
        analyser.draw_degree_dist(
            G.degree, hold=isHold, lbl='度分布',
            fit_func2=analyser.linear_log_fit, fit_range2=(0, 0.8, 0.35),
            fit_curve_range2=(0, 2),
            fit_func=analyser.linear_log_fit, fit_range=(1.2, 2.5, 0.5),
            fit_curve_range1=(0.8, 5),
            lim=common_lim,
            save_path=save_path, G=G
        )
        if PROCESS_TYPE_DEGREE_TYPE == 1:
            analyser.draw_degree_dist(
                G.in_degree, hold=isHold, lbl='入度分布',
                fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.30),
                fit_curve_range2=(0, 2),
                fit_func=analyser.linear_log_fit, fit_range=(1.5, 2.0, 0.5),
                fit_curve_range1=(0.8, 5),
                lim=common_lim,
                save_path=save_path, G=G
            )
            analyser.draw_degree_dist(
                G.out_degree, hold=isHold, lbl='出度分布',
                fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.35),
                fit_curve_range2=(0, 2),
                fit_func=analyser.linear_log_fit, fit_range=(1.2, 2.5, 0.5),
                fit_curve_range1=(0.8, 5),
                lim=common_lim,
                save_path=save_path, G=G
            )
    elif PROCESS_TYPE == 1:
        analyser.analyse_clustering_coefficient(
            G, save_path=save_path,
            fit_func=analyser.linear_log_fit, fit_range=(0.7, 2.5, 0.00),
            sample_count=5000,
            ticks=None, lim=(0, 3, -3.3, 0)
        )
    elif PROCESS_TYPE == 2:
        analyser.analyse_gini_coefficient(G, save_path=save_path, lbl='度值', hold=True)
        analyser.analyse_gini_coefficient(G, calc_type='out', save_path=save_path, lbl='入度', hold=True)
        analyser.analyse_gini_coefficient(G, calc_type='in', save_path=save_path, lbl='出度', hold=True)
    elif PROCESS_TYPE == 4:
        analyser.analyse_community_evolution(G, step=200, path='')


@utils.destroy
def livej(existed=None, isHold=True):
    save_path = path + 'LIVEJ'
    G = analyser.load_graph(save_path)
    common_lim = (0, 4.5, -6.8, 0)
    if PROCESS_TYPE == 0:
        analyser.draw_degree_dist(
            G.degree, hold=isHold, lbl='度分布',
            fit_func2=analyser.linear_log_fit, fit_range2=(0.5, 1.5, 0.5),
            fit_curve_range2=(0, 2.5),
            fit_func=analyser.linear_log_fit, fit_range=(1.8, 2.5, 0.7),
            fit_curve_range1=(1.5, 5),
            lim=common_lim,
            save_path=save_path, G=G
        )
        if PROCESS_TYPE_DEGREE_TYPE == 1:
            analyser.draw_degree_dist(
                G.in_degree, hold=isHold, lbl='入度分布',
                fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.30),
                fit_curve_range2=(0, 2),
                fit_func=analyser.linear_log_fit, fit_range=(1.7, 2.7, 0.5),
                fit_curve_range1=(0.8, 5),
                lim=common_lim,
                save_path=save_path, G=G
            )
            analyser.draw_degree_dist(
                G.out_degree, hold=isHold, lbl='出度分布',
                fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.35),
                fit_curve_range2=(0, 2),
                fit_func=analyser.linear_log_fit, fit_range=(1.2, 2.5, 0.5),
                fit_curve_range1=(0.8, 5),
                lim=common_lim,
                save_path=save_path, G=G
            )
    elif PROCESS_TYPE == 1:
        analyser.analyse_clustering_coefficient(
            G, save_path=save_path,
            fit_func=analyser.linear_log_fit, fit_range=(1, 2, 0.6, -0.5),
            sample_count=10000,
            ticks=None, lim=(0, 3, -3.3, 0)
        )
    elif PROCESS_TYPE == 2:
        analyser.analyse_gini_coefficient(G, save_path=save_path, lbl='度值', hold=True)
        analyser.analyse_gini_coefficient(G, calc_type='in', save_path=save_path, lbl='入度', hold=True)
        analyser.analyse_gini_coefficient(G, calc_type='out', save_path=save_path, lbl='出度', hold=True)
    elif PROCESS_TYPE == 4:
        analyser.analyse_community_evolution(G, step=200, path='')


@utils.destroy
def pokec(existed=None, isHold=True):
    save_path = path + 'POKEC'
    G = analyser.load_graph(save_path)
    common_lim = (0, 3.5, -6.3, 0)
    if PROCESS_TYPE == 0:
        analyser.draw_degree_dist(
            G.degree, hold=isHold, lbl='度分布',
            fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.8, 0.4),
            fit_curve_range2=(0, 2.5),
            fit_func=analyser.linear_log_fit, fit_range=(2.0, 3, 0.9),
            fit_curve_range1=(1.8, 5),
            lim=common_lim,
            save_path=save_path, G=G
        )
        if PROCESS_TYPE_DEGREE_TYPE == 1:
            analyser.draw_degree_dist(
                G.in_degree, hold=isHold, lbl='入度分布',
                fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.3, 0.30),
                fit_curve_range2=(0, 2.5),
                fit_func=analyser.linear_log_fit, fit_range=(1.8, 2.7, 0.5),
                fit_curve_range1=(1.7, 5),
                lim=common_lim,
                save_path=save_path, G=G
            )
            analyser.draw_degree_dist(
                G.out_degree, hold=isHold, lbl='出度分布',
                fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.35),
                fit_curve_range2=(0, 2.5),
                fit_func=analyser.linear_log_fit, fit_range=(1.8, 2.5, 0.5),
                fit_curve_range1=(1.7, 5),
                lim=common_lim,
                save_path=save_path, G=G
            )
    elif PROCESS_TYPE == 1:
        analyser.analyse_clustering_coefficient(
            G, save_path=save_path,
            fit_func=analyser.linear_log_fit, fit_range=(0.5, 2.2, 1.0, -0.55),
            sample_count=10000,
            ticks=None, lim=(0, 3, -3.3, 0)
        )
    elif PROCESS_TYPE == 2:
        analyser.analyse_gini_coefficient(G, save_path=save_path, lbl='度值', hold=True)
        analyser.analyse_gini_coefficient(G, calc_type='in', save_path=save_path, lbl='入度', hold=True)
        analyser.analyse_gini_coefficient(G, calc_type='out', save_path=save_path, lbl='出度', hold=True)
    elif PROCESS_TYPE == 4:
        analyser.analyse_community_evolution(G, step=200, path='')


@utils.destroy
def zhihu(existed=None, isHold=True):
    save_path = path + 'ZHIHU'
    G = analyser.load_graph(save_path)
    common_lim = (0, 4.5, -6.0, 0)
    if PROCESS_TYPE == 0:
        analyser.draw_degree_dist(
            G.degree, hold=isHold, lbl='度分布',
            fit_func=analyser.linear_log_fit, fit_range=(0.5, 1.5, 0.8),
            fit_curve_range1=(0, 6),
            lim=common_lim,
            save_path=save_path, G=G
        )
        if PROCESS_TYPE_DEGREE_TYPE == 1:
            analyser.draw_degree_dist(
                G.in_degree, hold=isHold, lbl='入度分布',
                fit_func2=analyser.linear_log_fit, fit_range2=(0.5, 1.3, 0.1),
                fit_curve_range2=(0, 2.5),
                fit_func=analyser.linear_log_fit, fit_range=(1.8, 2.7, 0.5),
                fit_curve_range1=(1.7, 5),
                lim=common_lim,
                save_path=save_path, G=G
            )
            analyser.draw_degree_dist(
                G.out_degree, hold=isHold, lbl='出度分布',
                fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.35),
                fit_curve_range2=(0, 2.5),
                fit_func=analyser.linear_log_fit, fit_range=(1.8, 2.5, 0.5),
                fit_curve_range1=(1.7, 5),
                lim=common_lim,
                save_path=save_path, G=G
            )
    elif PROCESS_TYPE == 1:
        analyser.analyse_clustering_coefficient(
            G, save_path=save_path,
            fit_func=analyser.linear_log_fit, fit_range=(0.5, 2.2, 0.5, -0.2),
            sample_count=10000,
            ticks=None, lim=(0, 3, -3.3, 0),
            hold=isHold
        )
    elif PROCESS_TYPE == 2:
        analyser.analyse_gini_coefficient(G, save_path=save_path, lbl='度值', hold=True)
        analyser.analyse_gini_coefficient(G, calc_type='in', save_path=save_path, lbl='入度', hold=True)
        analyser.analyse_gini_coefficient(G, calc_type='out', save_path=save_path, lbl='出度', hold=True)
    elif PROCESS_TYPE == 4:
        analyser.analyse_community_evolution(G, step=200, path='')


def degree_layout():
    # 设置画布大小
    # 度值和聚集系数
    # models = [er_model, ws_model, ba_model, analyser.NME, zhihu, twitter, gplus, pokec, livej]
    # names = ['Erdos-Renri模型', 'Watts Strogatz模型', 'Barabási-Albert模型',
    #          'NME模型', 'Zhihu', 'Twitter', 'Google+', 'Pokec', 'LiveJournal']
    # fig_width, fig_height = 4.8, 3.2
    # rows, cols = 3, 3
    # 拓扑图
    # models = [er_model, ws_model, ba_model, analyser.NME]
    # names = ['Erdos-Renri模型', 'Watts Strogatz模型', 'Barabási-Albert模型', 'NME模型']
    # fig_width, fig_height = 7.2, 5
    # rows, cols = 2, 2
    # 基尼系数
    models = [er_model, analyser.NME, zhihu, twitter, gplus, pokec, None, livej, None]
    names = ['Erdos-Renri模型', 'NME模型', 'Zhihu', 'Twitter', 'Google+', 'Pokec', None, 'LiveJournal', None]
    fig_width, fig_height = 4.8, 3.2
    rows, cols = 3, 3
    # 去掉大数据集
    # models = [analyser.NME, analyser.NME, analyser.NME, analyser.NME, analyser.NME, analyser.NME, analyser.NME,
    #           analyser.NME, analyser.NME, ]
    # names = ['NME模型', 'Zhihu', 'Twitter', 'Google+', 'Pokec', 'LiveJournal', 'Erdos-Renri模型', 'Erdos-Renri模型',
    #          'Erdos-Renri模型', ]
    # fig_width, fig_height = 4.8, 3.2
    # rows, cols = 3, 3

    lbls = ['(%s) %s' % (chr(ord('a') + idx), names[idx]) for idx in range(len(models))]

    plt.gcf().set_size_inches(fig_width * cols, fig_height * rows)
    for idx, lbl in enumerate(lbls):
        # 配置子图
        if models[idx] is not None:
            # 当设置当前子图不为空时才进行绘制
            plt.subplot(rows, cols, idx + 1)
            # 调用相应的模型分析
            models[idx](existed=True)
            # 重新设置label
            origin_lbl = plt.gca().xaxis.get_label().get_text()
            plt.xlabel('%s\n%s' % (origin_lbl, lbl), linespacing=2)
    plt.gcf().tight_layout()
    plt.savefig('data/stable/%s.png' % TITLE_LIST[PROCESS_TYPE])
    plt.show()


@utils.destroy
def degree_in_out_layout():
    """
    分析节点数相近的数据集的出入度分布
    :return:
    """
    # 载入数据集
    ba = analyser.load_graph(save_path='./data/stable/BA')
    nme = analyser.load_from_csv(path='./data/stable/NME_2')
    twitter = analyser.load_graph(save_path='./data/stable/TWITTER')
    gplus = analyser.load_graph(save_path='./data/stable/GPlus', graph_file='sub_graph.gpickle')
    common_lim = (0, 3, -4.3, 0)
    isHold = True
    save_path = ''
    data_list = (
        [nme.in_degree, twitter.in_degree, gplus.in_degree, ba.degree, '(a) 入度'],
        [nme.out_degree, twitter.out_degree, gplus.out_degree, ba.degree, '(b) 出度'],
    )
    analyser.FIT_CURVE = False
    plt.gcf().set_size_inches(6.4 * 2, 5)
    for idx, data in enumerate(data_list):
        plt.subplot(1, 2, idx + 1)
        analyser.draw_degree_dist(
            data[0], hold=isHold, lbl='NME模型',
            lim=common_lim,
            style='rx',
            save_path=save_path, G=nme
        )
        analyser.draw_degree_dist(
            data[1], hold=isHold, lbl='Twitter',
            lim=common_lim,
            style='go',
            save_path=save_path, G=twitter
        )
        analyser.draw_degree_dist(
            data[2], hold=isHold, lbl='Google+',
            lim=common_lim,
            style='b<',
            save_path=save_path, G=gplus
        )
        analyser.draw_degree_dist(
            data[3], hold=isHold, lbl='Barabási-Albert',
            lim=common_lim,
            style='c>',
            save_path=save_path, G=gplus
        )
        plt.xlabel('节点度值\n%s' % data[4], linespacing=2)
        plt.legend(loc='upper right', fontsize=16)
    plt.gcf().tight_layout()
    plt.savefig('data/stable/出入度分布对比.png')
    plt.show()


@utils.destroy
def evolution_community_analyse():
    """
    分析社区演化趋势
    :return:
    """
    # # 社区演化趋势随着参数和初始网络的变化
    # path_list = [
    #     # 初始200
    #     'data/20180516_133412_n10000_e21440_dta3_200',
    #     'data/20180518_183009_n10000_e29397_dta5_200',
    #     'data/20180516_133731_n10000_e33353_dta7_200',
    #
    #     # 初始600
    #     'data/20180514_113435_n10000_e21482_dta3_600',
    #     'data/20180510_220335_n10000_e28688_dta5_600',
    #     'data/20180514_121502_n10000_e33191_dta7_600',
    # ]
    # graph_list = [analyser.load_from_csv('./' + path) for path in path_list]
    # # 标签列表
    # lbl_list = [r'$N_0=%d,\epsilon=%d$' % (n, e) for n in [200, 600] for e in [3, 5, 7]]
    # title = '不同参数社区演化趋势'

    # 社区演化趋势随着数据集的变化
    gplus = nx.read_gpickle('data/stable/GPlus/graph.gpickle')
    graph_list = [
        nx.read_gpickle('./data/stable/ER/graph.gpickle'),
        nx.read_gpickle('./data/stable/WS/graph.gpickle'),
        nx.read_gpickle('./data/stable/BA/graph.gpickle'),
        analyser.load_from_csv('./data/20180510_220335_n10000_e28688_dta5_600'),
        nx.read_gpickle('data/stable/TWITTER/graph.gpickle'),
        nx.DiGraph(gplus.subgraph(sorted(list(gplus.nodes))[0:40000])),
    ]
    del gplus
    gc.collect()
    # 标签列表
    lbl_list = ['BA模型', 'ER模型', 'WS模型', 'NME模型', 'Twitter', 'Google+']
    title = '不同数据集社区演化趋势'

    # 使用多进程进行计算
    pool = multiprocessing.Pool(processes=10)
    # 社区分段大小以及分析步长
    common_segment_range = (0, 3, 5, 10, 100, 500)
    common_step = 500
    cm_status_list = []
    # 设置图像大小
    plt.gcf().set_size_inches(6.4 * 3, 4 * len(graph_list))
    for idx, G in enumerate(graph_list):
        # 将结果加入
        cm_status_list.append(
            pool.apply_async(
                analyser.analyse_community_evolution,
                kwds={
                    'G': G, 'step': common_step, 'max_analyse_size': 10000,
                    'segment_range': common_segment_range,
                }
            )
        )
    pool.close()
    pool.join()
    # 保存中间结果
    with open('./data/stable/community_%s.json' % title, 'w') as file:
        file.write(
            json.dumps([rst.get() for rst in cm_status_list])
        )
    # 获取多线程执行结果并绘图
    for idx, cm_status in enumerate(cm_status_list):
        # 对于每一个的计算结果 绘制相应行的图像
        cm_status = json.loads(cm_status.get())
        # 绘图
        analyser.draw_community_status(
            cm_cof=cm_status['cof'],
            # 因为这里使用了jsonencode keys都变成字符串了 所以要转型
            cm_size_dist={int(k): v for k, v in cm_status['size_dist'].items()},
            segment_range=common_segment_range,
            total_rows=len(graph_list),
            cur_row=idx,
            step=common_step,
            is_hold=True,
            lbl='(%d) %s' % (idx + 1, lbl_list[idx])
        )
    plt.gcf().tight_layout()
    plt.savefig('./data/stable/%s.png' % title)
    # plt.show()


if __name__ == '__main__':
    path = './data/stable/'
    matplotlib.rcParams.update({'font.size': 18})
    PROCESS_TYPE = 2
    PROCESS_TYPE_DEGREE_TYPE = 0
    '''分析ER模型'''
    # er_model()
    # er_model(existed=True, isHold=False)
    '''分析WS模型'''
    # ws_model()
    # ws_model(existed=True, isHold=False)
    '''分析BA模型'''
    # ba_model()
    # ba_model(existed=True, isHold=False)
    '''分析twitter数据集'''
    # twitter(existed=True, isHold=False)
    '''分析glus数据集'''
    # gplus(existed=True, isHold=False)
    '''分析livejournal数据集'''
    # livej(existed=True, isHold=False)
    '''分析pokec数据集'''
    # pokec(existed=True, isHold=False)
    '''分析zhihu数据集'''
    # zhihu(existed=True, isHold=False)

    '''综合分析'''
    # degree_layout()
    '''分析出度入度分布'''
    # degree_in_out_layout()
    '''分析社区演化趋势'''
    evolution_community_analyse()
