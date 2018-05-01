import sys

import requests
import redis
import pymongo
from collections import Iterable
import json
from Analyser.crawler.ProxiesService import ProxiesService


class RedisQueue:
    def __init__(self, db_name, host='localhost', port=6379, decode_responses=True):
        pool = redis.ConnectionPool(host=host, port=port, decode_responses=decode_responses)  # host是redis主机，需要redis服务端和客户端都起着 redis默认端口是6379
        self._r = redis.Redis(connection_pool=pool)
        self._db_name = db_name

    def push(self, value):
        if self.size() > 4294967295:
            raise Exception('队列长度超过最大值: ', 4294967295)
        # 判断是不是列表或者元组 以选择插入方式
        if isinstance(value, (list, tuple)):
            self._r.rpush(self._db_name, *value)
        else:
            self._r.rpush(self._db_name, value)

    def pop(self):
        # 如果长度不为0
        if self.size():
            self._r.lpop(self._db_name)

    @property
    def top(self):
        if self.size():
            return self._r.lrange(self._db_name, 0, 1)[0]
        else:
            print("[队列为空]")
            return None

    def size(self):
        return self._r.llen(self._db_name)


class RequestExecutor:
    def __init__(self):
        self._headers = {
            'authorization': 'oauth c3cef7c66a1843f8b3a9e6a1e3160e20',
            'Connection': 'keep-alive',
            'Host': 'www.zhihu.com',
            'origin': 'https://www.zhihu.com',
            'Referer': 'https://www.zhihu.com/topic/19550638/hot',
            'User-Agent': '',
        }
        # 连接到MongoDB
        client = pymongo.MongoClient(host='127.0.0.1', port=27017)
        db = client.get_database(name='zhihu')
        self._user_col = db.get_collection('user')
        self._relation_col = db.get_collection('relationship')
        self._breakpoint_col = db.get_collection('breakpoint')
        self._proxy_provider = ProxiesService()
        self.page_limit = 20

    def check_and_save_user(self, user):
        return self._user_col.update_one(
            filter={'id': user['id']},
            update={'$set': user},
            upsert=True
        ).upserted_id

    def save_relation(self, f, t):
        return self._relation_col.update_one(
            filter={'f': f, 't': t},
            update={'$set': {'f': f, 't': t}},
            upsert=True
        ).upserted_id

    def fetch(self, url, offset):
        params = {
            'include':
                'data[*].gender,answer_count,articles_count,follower_count,is_following,is_followed',
            'limit': self.page_limit,
            'offset': offset
        }
        # 随机选取用户代理
        self._headers['User-Agent'] = self._proxy_provider.fetch_a_user_agent()
        # 发出请求
        content = requests.get(
            url=url,
            params=params,
            headers=self._headers,
            proxies=self._proxy_provider.fetch_a_proxy()
        ).content.decode()
        return json.loads(content)


if __name__ == '__main__':
    # 起始用户
    start_user = 'zhouyuan'
    followees_url = 'https://www.zhihu.com/api/v4/members/{user}/followees'
    # 初始化请求执行器
    request_executor = RequestExecutor()
    # 初始化队列
    q = RedisQueue(db_name='zhihu_user_queue')
    # 如果队列为空 则加入初始化用户
    if q.size() < 1:
        q.push(start_user)
    while q.size():
        cur_user_token = q.top
        cur_offset = 0
        # 获取全部条目数
        total_item = request_executor.fetch(url=followees_url.format(user=cur_user_token), offset=0)['paging']['totals']
        print(cur_user_token, '全部条目:', total_item)
        # 开始爬取
        while cur_offset <= total_item:
            # 获取一页
            user_list = request_executor.fetch(
                url=followees_url.format(user=cur_user_token),
                offset=cur_offset
            )
            # 判断是不是真读到了数据
            if user_list.get('paging') is None:
                print(user_list, file=sys.stderr)
                raise Exception('错误：用户列表中没有paging字段 请检查是否ip受限')
            print(cur_user_token, '%d/%d' % (cur_offset, total_item), len(user_list['data']))
            # 处理当前页的用户列表
            for user in user_list['data']:
                # 保存访问到的关系
                request_executor.save_relation(cur_user_token, user['url_token'])
                # 去重
                upserted_id = request_executor.check_and_save_user(user)
                # 如果有插入结果 就是不重复 没有访问过这个节点
                if upserted_id:
                    q.push(user['url_token'])

            # 增加并保存offset
            cur_offset += request_executor.page_limit
            # breakpoint_col.update_one({}, {'$set': {'offset': offset}}, upsert=True)
        q.pop()
