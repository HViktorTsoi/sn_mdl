import redis
import json
import pprint
import NME
import multiprocessing
import sys


# 监听
def listening_task(handler_id):
    for event in p.listen():
        if event['type'] in ['pmessage', 'message']:
            if event['channel'] == 'NME_SIGNAL_START':
                # 获取参数
                params = json.loads(event['data'])
                pprint.pprint(params)
                try:
                    # 开始演化过程
                    NME.start_evolution(
                        init_graph_size=params['init_graph_size'],
                        delta_origin=params['delta_origin'],
                        max_ntwk_size=params['max_ntwk_size'],
                        k=params['k'],
                        analyse_community=params['analyse_community'],
                        uuid=params['uuid']
                    )
                except Exception as e:
                    print('演化过程中出现错误：\n', e, file=sys.stderr)


if __name__ == '__main__':

    # host是redis主机，需要redis服务端和客户端都起着 redis默认端口是6379
    pool = redis.ConnectionPool(host='127.0.0.1', port=6379, decode_responses=True)
    rds = redis.Redis(connection_pool=pool)

    # 订阅处理器
    p = rds.pubsub()
    p.subscribe(['NME_SIGNAL_START'])

    # 使用进程池来进行多个监听任务的处理
    pool = multiprocessing.Pool(processes=5)
    # 开启多个处理进程
    task_executor_number = 5
    for idx in range(task_executor_number):
        pool.apply_async(
            func=listening_task,
            kwds={
                'handler_id': idx
            }
        )
    # 关闭并等待进程结束
    pool.close()
    pool.join()
