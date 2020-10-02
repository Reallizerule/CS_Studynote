import requests
from collections import deque
import json
from lxml import etree
import hashlib
from pybloom import BloomFilter
import threading
import time
import re
import os


class CrawlBSF:

    downloaded_urls = []

    du_md5_file_name = 'E:/stockcrawl/download.txt'
    bloom_downloaded_urls = BloomFilter(1024 * 1024 * 16, 0.01)
    cur_queue = deque()

    def __init__(self):

        try:
            self.dumd5_file = open(self.du_md5_file_name, 'r')
            self.downloaded_urls = self.dumd5_file.readlines()
            self.dumd5_file.close()
            for urlmd5 in self.downloaded_urls:
                self.bloom_downloaded_urls.add(urlmd5[:-1])
        except IOError:
            print("File not found")
        finally:
            self.dumd5_file = open(self.du_md5_file_name, 'a+')

    def enqueueUrl(self, url):
        md5url = url[1].encode('utf8')
        if hashlib.md5(md5url).hexdigest(
        ) not in crawler.bloom_downloaded_urls:
            self.cur_queue.append(url)

    def dequeuUrl(self):
        try:
            url = self.cur_queue.popleft()
            return url
        except IndexError:
            return None

    def close(self):
        self.dumd5_file.close()


def get_text(url):
    try:

        proxyHost = "http-dyn.abuyun.com"
        proxyPort = "9020"
        proxyMeta = "http://%(user)s:%(pass)s@%(host)s:%(port)s" % {
            "host": proxyHost,
            "port": proxyPort,
            "user": 'HC3DA8L507Z8PU8D',
            "pass": '2D8CA64EF80460DF',
        }

        proxies = {
            "http": proxyMeta,
            "https": proxyMeta,
        }

        html_text = requests.get(url, proxies=proxies, timeout=10)
        print(html_text.status_code)

        return html_text.text
    except BaseException:

        print("内容读取读取")
        return ""


def get_page_content(url, stock, model):

    try:
        talk_html2 = get_text(url[1])
        stock_page = etree.HTML(talk_html2)
        notice = stock_page.xpath('//div[@class="detail-body"]/div[1]')[0].text
        print(stock[:-1], url[0][:4])
        path = 'E:/stockcrawl/股票/%s/%s' % (stock[:-1], url[0][0:4])

        print(path)
        i = 0

        isExists = os.path.exists(path)

        if not isExists:
            os.makedirs(path)
            print(path + ' 创建成功')
        while True:
            if stock[:-1] + url[0][:4] + url[0][5:7] + \
                    url[0][8:11] + "%02d" % i in model.keys():
                i += 1
            else:
                break

        model[stock[:-1] + url[0][:4] + url[0]
              [5:7] + url[0][8:11] + "%02d" % i] = notice

        mdurl = url[1].encode('utf8')

        new_md5 = hashlib.md5(mdurl).hexdigest()

        crawler.dumd5_file.write(new_md5 + "\n")
    except BaseException:
        return ""


crawler = CrawlBSF()
threads = []
max_threads = 20
CRAWL_DELAY = 0.2


dum = open('E:/stockcrawl/stocks.csv', 'r')
stocklist = dum.readlines()

dum.close()

for stock in stocklist[1756:]:

    for i in range(1, 20):

        stockurl = ("http://data.eastmoney.com/notices/getdata.ashx?"
                    "StockCode=%s&CodeType=1&PageIndex=%d&PageSize=50&jsObj="
                    "pcYdswTt&SecNodeType=0&FirstNodeType=0&rt=50214727") % (stock[2:-1], i)
        print(stockurl)

        html_page = get_text(stockurl)

        talk_url = re.findall(
            r'"ENDDATE":"(201[5678].*?)T.*?"Url":"(.*?)"}',
            html_page)
        if talk_url == []:
            break

        else:

            for num in talk_url:

                crawler.enqueueUrl(num)

    model = {}
    # Go on next level, before that, needs to wait all current level crawling
    # done
    while True:
        url = crawler.dequeuUrl()

        if url is None:
            for t in threads:
                t.join()

            break

        else:

            while True:

                 # first remove all finished running threads
                for t in threads:
                    if not t.is_alive():
                        threads.remove(t)
                if len(threads) >= max_threads:
                    time.sleep(CRAWL_DELAY)
                    continue
                try:

                    t = threading.Thread(
                        target=get_page_content, name=None, args=(
                            url, stock, model))
                    threads.append(t)
                    # set daemon so main thread can exit when receives ctrl-c
                    t.setDaemon(True)
                    t.start()
                    time.sleep(CRAWL_DELAY)
                    break
                except Exception:
                    print("Error: unable to start thread")

    for talk_time in model.keys():
        with open('E:/stockcrawl/股票/%s/%s/%s.json' % (stock[:-1], talk_time[8:12], talk_time), 'w', encoding='utf-8') as json_file:
            infodict = {}
            infodict[talk_time] = model[talk_time]
            json.dump(infodict, json_file, ensure_ascii=False)
    print(stock + "完成")

crawler.close()
