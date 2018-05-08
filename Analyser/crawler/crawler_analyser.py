# %% run
import pymongo
from bson.son import SON
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import networkx
import gc

users = pymongo.MongoClient('127.0.0.1', 27017).get_database('zhihu').get_collection('relationship')
G = networkx.DiGraph()
cur = users.find()
for edge in cur:
    G.add_edge(edge['f'], edge['t'])
dist = defaultdict(int)
for id, count in list(G.in_degree):
    if int(count) > 0:
        dist[int(count)] += 1
in_dist = np.array(list(dist.items()))
plt.scatter(in_dist[:, 0], in_dist[:, 1])
plt.xscale('symlog')
plt.yscale('symlog')
plt.show()

dist = defaultdict(int)
for id, count in list(G.out_degree):
    if int(count) > 0:
        dist[int(count)] += 1
out_dist = np.array(list(dist.items()))
plt.scatter(out_dist[:, 0], out_dist[:, 1])
plt.xscale('symlog')
plt.yscale('symlog')
plt.show()

dist = defaultdict(int)
for id, count in list(G.degree):
    if int(count) > 0:
        dist[int(count)] += 1
dist = np.array(list(dist.items()))
plt.scatter(dist[:, 0], dist[:, 1])
plt.xscale('symlog')
plt.yscale('symlog')
plt.show()

del dist
gc.collect()

print(networkx.average_clustering(G.to_undirected()))
