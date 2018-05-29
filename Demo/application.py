from flask import Flask, request
import redis
import json
import uuid
from Analyser.crawler.zhihu_spider import RedisQueue

app = Flask(__name__)
# host是redis主机，需要redis服务端和客户端都起着 redis默认端口是6379
pool = redis.ConnectionPool(host='127.0.0.1', port=6379, decode_responses=True)
rds = redis.Redis(connection_pool=pool)


@app.route('/start')
def start():
    # 为此次演化生成唯一id
    uid = uuid.uuid1()
    print('本次演化ID', uid)
    # 发布演化开始消息
    rds.publish('NME_SIGNAL_START', json.dumps({
        'init_graph_size': 10,
        'delta_origin': 10,
        'max_ntwk_size': 10000,
        'k': 7,
        'analyse_community': False,
        'uuid': str(uid)
    }))
    return str(uid)


@app.route('/fetch/<uid>')
def fetch(uid):
    print(f'获取{uid}的网络拓扑信息')
    rq = RedisQueue(db_name=f'NME_{uid}')
    return json.dumps(
        rq.pop()
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
