import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import Analyser.analyser as analyser


def draw_diameter(data):
    # 网络半径演化过程
    dia = np.array([(int(k), v) for k, v in data['dia'].items() if k and v])
    plt.plot(dia[:, 0], dia[:, 1], color='red')
    plt.xscale('symlog')
    plt.xlabel('演化时间(节点数)')
    plt.ylabel('网络直径')
    plt.xlim(1, 3 * 10 ** 4)
    plt.ylim(0, 11)
    plt.hlines(9, 0, 10, colors='gray', linestyles='--', linewidths=1, label='最大直径')
    plt.hlines(6, 0, 2 * 10 ** 4, colors='gray', linestyles='-.', linewidths=1, label='演化稳定直径')
    plt.legend()
    plt.savefig(data_path + '/diameter.png')
    plt.show()


def draw_cc(data):
    # 网络聚集系数演化过程
    cc = np.array([(int(k), v) for k, v in data['cc'].items() if k and v])
    plt.plot(cc[:, 0], cc[:, 1], 'b')
    plt.xscale('symlog')
    plt.xlabel('演化时间(节点数)')
    plt.ylabel('聚集系数')
    plt.hlines(0.185, 0, 2 * 10 ** 4, colors='gray', linestyles='--', linewidths=1, label='演化稳定聚集系数')
    plt.xlim(5, 4 * 10 ** 4)
    plt.ylim(0, 0.5)
    plt.legend()
    plt.savefig(data_path + '/cc.png')
    plt.show()


def cc_edge_table(data):
    cc = np.array([(int(k), v) for k, v in data['cc'].items()])
    edges = np.array([(int(k), v) for k, v in data['edge'].items()])
    tbl = zip(np.array(cc[:, 0], dtype=np.int), cc[:, 1], edges[:, 1])
    with open(data_path + '/cc_dege.csv', 'w') as file:
        csv_f = csv.writer(file)
        csv_f.writerow(['节点数', '聚集系数', '边数'])
        for row in tbl:
            csv_f.writerow(row)


if __name__ == '__main__':
    data_path = '../data/20180130_204852_n20000_e74456_初始不连通d10k10'
    with open(data_path + '/cc_evo_data.json') as file:
        data = json.loads(file.read())
        draw_diameter(data)
        # draw_cc(data)
        # cc_edge_table(data)
