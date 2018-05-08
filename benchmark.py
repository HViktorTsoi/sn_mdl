import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import Analyser.analyser as analyser
import Analyser.utils as utils


@utils.destroy
def er_model(existed=None, isHold=True):
    save_path = path + 'ER'
    if not existed:
        G = nx.fast_gnp_random_graph(20000, 0.05)
        analyser.save_graph(G, save_path)
    else:
        G = analyser.load_graph(save_path)
    # 分析度分布
    analyser.draw_degree_dist(
        G.degree, process=lambda v: v,
        lim=(800, 1200, 0, 0.020),
        fit_func=analyser.poisson_fit,
        hold=isHold, lbl='度分布', save_path=save_path, G=G,
        ticks=(
            [800, 1000, 1200], ['$0.8x10^3$', '$10^3$', '$1.2x10^3$'],
            [0.001, 0.005, 0.01, 0.015, 0.02], ['$10^{-3}$', '$0.5x10^{-2}$', '$10^{-2}$', '$1.5x10^{-2}$', '$2x10^{-2}$']
        )
    )
    # 分析聚集系数
    # analyser.analyse_clustering_coefficient(
    #     G, fit_func=analyser.const_fit, save_path=save_path,
    #     process=lambda v: v, lim=(800, 1200, 0.035, 0.07), p=0.05,
    #     sample_count=3000,
    #     ticks=(
    #         [800, 900, 1000, 1100, 1200], ['$0.8x10^3$', '$0.9x10^3$', '$10^3$', '$1.1x10^3$', '$1.2x10^3$'],
    #         [0.03, 0.04, 0.05, 0.06, 0.07], ['$3x10^{-2}$', '$4x10^{-2}$', '$5x10^{-2}$', '$6x10^{-2}$', '$7x10^{-2}$']
    #     ),
    #     hold=isHold
    # )
    # analyser.analyse_gini_coefficient(G, save_path=save_path, lbl='Degree', hold=True)


@utils.destroy
def ws_model(existed=None, isHold=True):
    save_path = path + 'WS'
    if not existed:
        G = nx.newman_watts_strogatz_graph(20000, 500, 0.2)
        analyser.save_graph(G, save_path)
    else:
        G = analyser.load_graph(save_path)
    analyser.draw_degree_dist(
        G.degree, process=lambda v: v,
        lim=(500, 700, 0, 0.05),
        fit_func=analyser.norm_fit,
        hold=isHold, lbl='度分布', save_path=save_path, G=G,
        ticks=(
            [500, 600, 700], ['$0.5x10^3$', '$0.6x10^3$', '$0.7x10^3$'],
            [0.001, 0.01, 0.02, 0.03, 0.04, 0.05], ['$10^{-3}$', '$10^{-2}$', '$2x10^{-2}$', '$3x10^{-2}$', '$4x10^{-2}$', '$5x10^{-2}$']
        )
    )
    # analyser.analyse_clustering_coefficient(
    #     G, fit_func=analyser.linear_log_fit,
    #     fit_range=(0, 1000, 0.003), fit_curve_range=(525, 680),
    #     save_path=save_path, process=lambda v: v,
    #     sample_count=4000,
    #     lim=(500, 700, 0.35, 0.7), ticks=(
    #         [500, 550, 600, 650, 700], ['$0.5x10^3$', '$0.55x10^3$', '$0.6x10^3$', '$0.65x10^3$', '$0.7x10^3$'],
    #         [0.3, 0.4, 0.5, 0.6, 0.7], ['$3x10^{-1}$', '$4x10^{-1}$', '$5x10^{-1}$', '$6x10^{-1}$', '$7x10^{-1}$']
    #     )
    # )
    # analyser.analyse_gini_coefficient(G, save_path=save_path, lbl='节点Degree', hold=False)


@utils.destroy
def ba_model(existed=None, isHold=True):
    save_path = path + 'BA'
    if not existed:
        G = nx.barabasi_albert_graph(20000, 5)
        analyser.save_graph(G, save_path)
    else:
        G = analyser.load_graph(save_path)
    analyser.draw_degree_dist(
        G.degree, process=lambda v: np.log10(v),
        lim=(0.0, 3, -4.3, 0),
        fit_func=analyser.linear_log_fit,
        fit_range=(0.1, 2.2, 0.5),
        hold=isHold, lbl='度分布', save_path=save_path, G=G
    )
    # analyser.analyse_clustering_coefficient(
    #     G, save_path=save_path, sample_count=7000,
    #     fit_func=analyser.linear_log_fit, fit_range=(0, 1.1, 0.20),
    #     ticks=None, lim=(0, 3, -3.3, 0))
    # analyser.analyse_gini_coefficient(G, save_path=save_path, lbl='节点Degree', hold=False)


@utils.destroy
def twitter(existed=None, isHold=True):
    save_path = path + 'TWITTER'
    G = analyser.load_graph(save_path)
    # 取子图
    G_sub = G.subgraph(sorted(list(G.nodes))[30000:50000])
    common_lim = (0, 3, -4.3, 0)
    analyser.draw_degree_dist(
        G_sub.in_degree, hold=isHold, lbl='入度分布',
        fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.30),
        fit_curve_range2=(0, 2),
        fit_func=analyser.linear_log_fit, fit_range=(1.5, 2.0, 0.5),
        fit_curve_range1=(0.8, 5),
        lim=common_lim,
        save_path=save_path, G=G_sub
    )
    # analyser.draw_degree_dist(
    #     G_sub.out_degree, hold=isHold, lbl='出度分布',
    #     fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.35),
    #     fit_curve_range2=(0, 2),
    #     fit_func=analyser.linear_log_fit, fit_range=(1.2, 2.5, 0.5),
    #     fit_curve_range1=(0.8, 5),
    #     lim=common_lim,
    #     save_path=save_path, G=G_sub
    # )
    # analyser.draw_degree_dist(
    #     G_sub.degree, hold=isHold, lbl='度分布',
    #     fit_func2=analyser.linear_log_fit, fit_range2=(0, 0.8, 0.35),
    #     fit_curve_range2=(0, 2),
    #     fit_func=analyser.linear_log_fit, fit_range=(1.2, 2.5, 0.5),
    #     fit_curve_range1=(0.8, 5),
    #     lim=common_lim,
    #     save_path=save_path, G=G_sub
    # )
    # analyser.analyse_clustering_coefficient(
    #     G_sub, save_path=save_path,
    #     fit_func=analyser.linear_log_fit, fit_range=(1.2, 3, -0.2),
    #     sample_count=15000,
    #     ticks=None, lim=(0, 3, -3.3, 0)
    # )
    # analyser.analyse_gini_coefficient(G_sub, save_path=save_path, lbl='Degree', hold=True)
    # analyser.analyse_gini_coefficient(G_sub, calc_type='out', save_path=save_path, lbl='In Degree', hold=True)
    # analyser.analyse_gini_coefficient(G_sub, calc_type='in', save_path=save_path, lbl='Out Degree', hold=True)


@utils.destroy
def gplus(existed=None, isHold=True):
    save_path = path + 'GPlus'
    # 取子图
    # G_sub = G.subgraph(sorted(list(G.nodes))[50000:65000])
    G_sub = analyser.load_graph(save_path, graph_file='sub_graph.gpickle')
    common_lim = (0, 3, -4.3, 0)
    analyser.draw_degree_dist(
        G_sub.in_degree, hold=isHold, lbl='入度分布',
        fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.30),
        fit_curve_range2=(0, 2),
        fit_func=analyser.linear_log_fit, fit_range=(1.5, 2.0, 0.5),
        fit_curve_range1=(0.8, 5),
        lim=common_lim,
        save_path=save_path, G=G_sub
    )
    # analyser.draw_degree_dist(
    #     G_sub.out_degree, hold=isHold, lbl='出度分布',
    #     fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.35),
    #     fit_curve_range2=(0, 2),
    #     fit_func=analyser.linear_log_fit, fit_range=(1.2, 2.5, 0.5),
    #     fit_curve_range1=(0.8, 5),
    #     lim=common_lim,
    #     save_path=save_path, G=G_sub
    # )
    # analyser.draw_degree_dist(
    #     G_sub.degree, hold=isHold, lbl='度分布',
    #     fit_func2=analyser.linear_log_fit, fit_range2=(0, 0.8, 0.35),
    #     fit_curve_range2=(0, 2),
    #     fit_func=analyser.linear_log_fit, fit_range=(1.2, 2.5, 0.5),
    #     fit_curve_range1=(0.8, 5),
    #     lim=common_lim,
    #     save_path=save_path, G=G_sub
    # )
    # analyser.analyse_clustering_coefficient(
    #     G_sub, save_path=save_path,
    #     fit_func=analyser.linear_log_fit, fit_range=(0.7, 2.5, 0.00),
    #     sample_count=5000,
    #     ticks=None, lim=(0, 3, -3.3, 0)
    # )
    # analyser.analyse_gini_coefficient(G_sub, save_path=save_path, lbl='Degree', hold=True)
    # analyser.analyse_gini_coefficient(G_sub, calc_type='out', save_path=save_path, lbl='In Degree', hold=True)
    # analyser.analyse_gini_coefficient(G_sub, calc_type='in', save_path=save_path, lbl='Out Degree', hold=True)


@utils.destroy
def livej(existed=None, isHold=True):
    save_path = path + 'LIVEJ'
    G_sub = analyser.load_graph(save_path)
    common_lim = (0, 4.5, -6.8, 0)
    analyser.draw_degree_dist(
        G_sub.degree, hold=isHold, lbl='度分布',
        fit_func2=analyser.linear_log_fit, fit_range2=(0.5, 1.5, 0.5),
        fit_curve_range2=(0, 2.5),
        fit_func=analyser.linear_log_fit, fit_range=(2.0, 3, 0.7),
        fit_curve_range1=(1.5, 5),
        lim=common_lim,
        save_path=save_path, G=G_sub
    )
    # analyser.draw_degree_dist(
    #     G_sub.in_degree, hold=isHold, lbl='入度分布',
    #     fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.30),
    #     fit_curve_range2=(0, 2),
    #     fit_func=analyser.linear_log_fit, fit_range=(1.7, 2.7, 0.5),
    #     fit_curve_range1=(0.8, 5),
    #     lim=common_lim,
    #     save_path=save_path, G=G_sub
    # )
    # analyser.draw_degree_dist(
    #     G_sub.out_degree, hold=isHold, lbl='出度分布',
    #     fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.35),
    #     fit_curve_range2=(0, 2),
    #     fit_func=analyser.linear_log_fit, fit_range=(1.2, 2.5, 0.5),
    #     fit_curve_range1=(0.8, 5),
    #     lim=common_lim,
    #     save_path=save_path, G=G_sub
    # )
    # analyser.analyse_clustering_coefficient(
    #     G_sub, save_path=save_path,
    #     fit_func=analyser.linear_log_fit, fit_range=(1, 2, 0.6, -0.5),
    #     sample_count=10000,
    #     ticks=None, lim=(0, 3, -3.3, 0)
    # )
    # analyser.analyse_gini_coefficient(G_sub, save_path=save_path, lbl='Degree', hold=True)
    # analyser.analyse_gini_coefficient(G_sub, calc_type='in', save_path=save_path, lbl='In Degree', hold=True)
    # analyser.analyse_gini_coefficient(G_sub, calc_type='out', save_path=save_path, lbl='Out Degree', hold=True)


@utils.destroy
def pokec(existed=None, isHold=True):
    save_path = path + 'POKEC'
    G_sub = analyser.load_graph(save_path)
    common_lim = (0, 3.5, -6.3, 0)
    analyser.draw_degree_dist(
        G_sub.degree, hold=isHold, lbl='度分布',
        fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.8, 0.4),
        fit_curve_range2=(0, 2.5),
        fit_func=analyser.linear_log_fit, fit_range=(2.0, 3, 0.9),
        fit_curve_range1=(1.8, 5),
        lim=common_lim,
        save_path=save_path, G=G_sub
    )
    # analyser.draw_degree_dist(
    #     G_sub.in_degree, hold=isHold, lbl='入度分布',
    #     fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.3, 0.30),
    #     fit_curve_range2=(0, 2.5),
    #     fit_func=analyser.linear_log_fit, fit_range=(1.8, 2.7, 0.5),
    #     fit_curve_range1=(1.7, 5),
    #     lim=common_lim,
    #     save_path=save_path, G=G_sub
    # )
    # analyser.draw_degree_dist(
    #     G_sub.out_degree, hold=isHold, lbl='出度分布',
    #     fit_func2=analyser.linear_log_fit, fit_range2=(0, 1.0, 0.35),
    #     fit_curve_range2=(0, 2.5),
    #     fit_func=analyser.linear_log_fit, fit_range=(1.8, 2.5, 0.5),
    #     fit_curve_range1=(1.7, 5),
    #     lim=common_lim,
    #     save_path=save_path, G=G_sub
    # )
    # analyser.analyse_clustering_coefficient(
    #     G_sub, save_path=save_path,
    #     fit_func=analyser.linear_log_fit, fit_range=(0.5, 2.2, 1.0, -0.55),
    #     sample_count=10000,
    #     ticks=None, lim=(0, 3, -3.3, 0)
    # )
    # analyser.analyse_gini_coefficient(G_sub, save_path=save_path, lbl='Degree', hold=True)
    # analyser.analyse_gini_coefficient(G_sub, calc_type='in', save_path=save_path, lbl='In Degree', hold=True)
    # analyser.analyse_gini_coefficient(G_sub, calc_type='out', save_path=save_path, lbl='Out Degree', hold=True)


@utils.destroy
def email(existed=None, isHold=True):
    save_path = path + 'EMAIL'
    G_sub = analyser.load_graph(save_path)
    analyser.analyse_gini_coefficient(G_sub, save_path=save_path, lbl='Degree', hold=True)
    analyser.analyse_gini_coefficient(G_sub, calc_type='in', save_path=save_path, lbl='In Degree', hold=True)
    analyser.analyse_gini_coefficient(G_sub, calc_type='out', save_path=save_path, lbl='Out Degree', hold=True)


def degree_layout():
    import matplotlib
    matplotlib.rcParams.update({'font.size': 18})
    # 设置画布大小
    # 度值和聚集系数
    models = [er_model, ws_model, ba_model, analyser.NME, twitter, gplus, pokec, livej]
    names = ['Erdos-Renri model', 'Watts Strogatz model',
             'Barabási-Albert Model', 'NMC Model', 'Twitter', 'Google+', 'Pokec', 'LiveJournal']
    # 基尼系数
    # models = [er_model, analyser.NME, twitter, gplus, pokec, livej]
    # names = ['Erdos-Renri model', 'NMC Model', 'Twitter', 'Google+', 'Pokec', 'LiveJournal']
    # 去掉大数据集
    # models = [analyser.NME, analyser.NME, analyser.NME, analyser.NME, analyser.NME, analyser.NME, ]
    # names = ['Erdos-Renri model', 'NMC Model', 'Twitter', 'Google+', 'Pokec', 'LiveJournal']

    lbls = ['(%s) %s' % (chr(ord('a') + idx), names[idx]) for idx in range(len(models))]
    rows = 4
    cols = 2
    plt.gcf().set_size_inches(4.8 * cols, 3.2 * rows)
    for idx, lbl in enumerate(lbls):
        # 配置子图
        plt.subplot(rows, cols, idx + 1)
        # 调用相应的模型分析
        models[idx](existed=True)
        # 重新设置label
        origin_lbl = plt.gca().xaxis.get_label().get_text()
        plt.xlabel('%s\n%s' % (origin_lbl, lbl), linespacing=2)
    plt.gcf().tight_layout()
    plt.savefig('data/stable/i18n_综合度分布曲线.png')
    plt.show()


@utils.destroy
def degree_in_out_layout():
    import matplotlib
    matplotlib.rcParams.update({'font.size': 18})
    # 载入数据集
    ba = analyser.load_graph(save_path='./data/stable/BA')
    nme = analyser.load_from_csv(path='./data/stable/NME_2')
    twitter = analyser.load_graph(save_path='./data/stable/TWITTER')
    gplus = analyser.load_graph(save_path='./data/stable/GPlus', graph_file='sub_graph.gpickle')
    common_lim = (0, 3, -4.3, 0)
    isHold = True
    save_path = ''
    data_list = (
        [nme.in_degree, twitter.in_degree, gplus.in_degree, ba.degree, '(a) In Degree'],
        [nme.out_degree, twitter.out_degree, gplus.out_degree, ba.degree, '(b) Out Degree'],
    )
    analyser.FIT_CURVE = False
    plt.gcf().set_size_inches(6.4 * 2, 4.8)
    for idx, data in enumerate(data_list):
        plt.subplot(1, 2, idx + 1)
        analyser.draw_degree_dist(
            data[0], hold=isHold, lbl='NMC Model',
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
        plt.xlabel('Node Degree\n%s' % data[4], linespacing=2)
        plt.legend(loc='upper right', fontsize=14)
    plt.gcf().tight_layout()
    plt.savefig('data/stable/i18n_出入度分布对比.png')
    plt.show()


if __name__ == '__main__':
    path = "./data/stable/"
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
    '''分析email数据集'''
    # email(existed=True, isHold=False)

    # degree_layout()
    degree_in_out_layout()
