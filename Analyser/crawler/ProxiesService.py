import requests
import requests.adapters
import re
import json
import random
import pymongo


class ProxiesService:
    def __init__(self):
        self._url = "http://www.gatherproxy.com/zh/proxylist/country/?c=China"
        self._proxy_set = pymongo.MongoClient('127.0.0.1', 27017) \
            .get_database('zhihu') \
            .get_collection('proxy')
        # 代理池
        self._user_agent_pool = [
            'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6',
            'Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER',
            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11',
            'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; 360SE)',
            'Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20',
            'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6',
            'Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1',
            'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)',
            'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E; LBBROWSER)',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.3 Mobile/14E277 Safari/603.1.30',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
        ]
        requests.adapters.DEFAULT_RETRIES = 2

    def _verify(self, proxy):
        try:
            print('正在验证', proxy)
            rep = requests.get(
                'http://ip.cn',
                headers={'User-Agent': 'curl/7.52.1'},
                proxies=proxy,
                timeout=3
            )
            print(proxy, rep)
            return rep.ok
        except Exception as e:
            print('连接失败：', e)
            return False

    def refresh_proxies(self):
        # 遍历20页
        for page_id in range(1, 20):
            print('正在请求第%d页' % page_id)
            for proxy in self._proxy_set.find({}, {'http': 1, '_id': 0}):
                if not self._verify(proxy):
                    print('代理已失效 删除: ', proxy)
                    self._proxy_set.remove(proxy)
            res = requests.post(
                url=self._url,
                proxies={'http': 'socks5://127.0.0.1:1080'},
                data={
                    'Filter': 'elite',
                    'PageIdx': page_id,
                    'Uptime': 50,
                    'Country': 'china'
                }
            ).content.decode()
            proxy_list = re.findall(
                r"document.write\('(.*?)'\)[\s\S]*?gp.dep\('(.*?)'\)",
                res,
            )
            proxy_list = [{'http': "http://%s:%s" % (item[0], int(item[1], 16))}
                          for item in proxy_list]
            for proxy in proxy_list:
                if self._verify(proxy):
                    print('加入代理：', proxy)
                    self._proxy_set.update_one(
                        filter=proxy,
                        update={
                            '$set': proxy
                        },
                        upsert=True
                    )

    def fetch_a_proxy(self):
        proxy = self._proxy_set.aggregate([
            {'$project': {'http': 1, '_id': 0}},
            {'$sample': {'size': 1}}
        ]).next()
        # print('|== 使用代理: ', proxy)
        return proxy

    def fetch_a_user_agent(self):
        return random.choice(self._user_agent_pool)


if __name__ == '__main__':
    ps = ProxiesService()
    while True:
        ps.refresh_proxies()
