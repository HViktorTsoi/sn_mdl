from flask import Flask, request
import redis
import json
import uuid
from Analyser.crawler.zhihu_spider import RedisQueue

app = Flask(__name__)
# host是redis主机，需要redis服务端和客户端都起着 redis默认端口是6379
pool = redis.ConnectionPool(host='127.0.0.1', port=6379, decode_responses=True)
rds = redis.Redis(connection_pool=pool)


@app.route('/start', methods=['POST'])
def start():
    # 为此次演化生成唯一id
    uid = uuid.uuid1()
    print('本次演化ID', uid)
    # 获取网络演化参数
    evo_params = json.loads(request.data.decode())
    # 设置演化唯一ID
    evo_params['uuid'] = str(uid)
    print(evo_params)
    # 发布演化开始消息
    rds.publish('NME_SIGNAL_START', json.dumps(evo_params))
    return str(uid)


@app.route('/fetch/<uid>')
def fetch(uid):
    print(f'获取{uid}的网络拓扑信息')
    rq = RedisQueue(db_name=f'NME_{uid}')
    graph_raw_data = rq.pop()
    if graph_raw_data is not None:
        graph = json.loads(graph_raw_data)
        graph_in_echarts_format = {}
        graph_in_echarts_format['nodes'] = [
            {'id': node_id, 'name': node_id, 'itemStyle': {}}
            for node_id in graph['nodes']] if graph['nodes'] else None
        graph_in_echarts_format['links'] = [
            {'source': edge[0], 'target': edge[1]} for edge in graph['links']] if graph['links'] else None
        graph_in_echarts_format['dist'] = graph['dist']
        print(graph_in_echarts_format['dist'])
        return json.dumps(graph_in_echarts_format)
    else:
        return ''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
