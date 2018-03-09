#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    _/      _/    _/_/_/  _/    _/    _/_/_/
   _/_/    _/  _/        _/    _/  _/
  _/  _/  _/  _/        _/_/_/_/  _/
 _/    _/_/  _/        _/    _/  _/
_/      _/    _/_/_/  _/    _/    _/_/_/
https://github.com/TW-NCHC/functionality-scenario-test-A-2018

This program will get streaming images and speed data from
http://tisvcloud.freeway.gov.tw/cctv_value.xml.gz defined url.
@author August Chao <AugustChao@narlabs.org.tw>
"""

from __future__ import print_function
import sys
import os
import cv2
import urllib2
import StringIO
import gzip
import logging
import time
import socket
import signal
import pandas as pd
import numpy as np
import subprocess
import types
import threading
import ctypes
from threading import Event, Thread
from bs4 import BeautifulSoup
from optparse import OptionParser
from geopy.distance import great_circle
from functools import wraps

socket.setdefaulttimeout(3)

def retry(ExceptionToCheck, tries=4, delay=3, backoff=2, logger=None):
    """Retry calling the decorated function using an exponential backoff.

    http://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
    original from: http://wiki.python.org/moin/PythonDecoratorLibrary#Retry

    :param ExceptionToCheck: the exception to check. may be a tuple of
        exceptions to check
    :type ExceptionToCheck: Exception or tuple
    :param tries: number of times to try (not retry) before giving up
    :type tries: int
    :param delay: initial delay between retries in seconds
    :type delay: int
    :param backoff: backoff multiplier e.g. value of 2 will double the delay
        each retry
    :type backoff: int
    :param logger: logger to use. If None, print
    :type logger: logging.Logger instance
    """
    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck, e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    if logger:
                        logger.warning(msg)
                    else:
                        print (msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry

def getGZFile2Soup(fn = "http://tisvcloud.freeway.gov.tw/cctv_value.xml.gz"):
    logging.info("Geting %s"%fn)
    response = urllib2.urlopen(fn)
    compressedFile = StringIO.StringIO(response.read())
    decompressedFile = gzip.GzipFile(fileobj=compressedFile, mode='rb')

    return BeautifulSoup(decompressedFile.read(), 'xml')

def getUrlOpen(url):
    return urllib2.urlopen(url, timeout=1)

def getConnection(url):
    stream = None
    while type(stream)==type(None):
        stream = getUrlOpen(url)
    return stream


@retry(Exception, tries=100)
def getImg(url):
    stream=getConnection(url)
    _ttl = 60*1.
    prev_time = time.time()
    idx = 0
    bytes=''
    while True:
        if (time.time() - prev_time) > _ttl:
            raise socket.timeout('got no image data from source')

        try:
            bytes += stream.read(1024)

            a = bytes.find('\xff\xd8') # JPEG start
            b = bytes.find('\xff\xd9') # JPEG end
            if a!=-1 and b!=-1:
                jpg = bytes[a:b+2] # actual image
                bytes= bytes[b+2:] # other informations
                idx += 1

                if (idx%50) == 0:
                    stream=getConnection(url)
                    bytes=''

                prev_time = time.time()
                yield cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
        except (socket.timeout, TypeError):
            logging.info(">> Read() error, reconnecting")
            try:
                stream=getConnection(url)
            except (socket.timeout, TypeError):
                try:
                    stream=getConnection(url)
                except (socket.timeout, TypeError):
                    logging.error("!!! tried 3 times, can't reconnect... ")

            bytes=''

def getRetrivedImgCnt(exportPath, token):
    DIR = "%s/%s"%(exportPath, token)
    return len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])


def getImgStream(csv_fn, exportPath, token):
    outPath = "%s/%s"%(exportPath, token)

    try:
        df = pd.read_csv(csv_fn)
    except:
        logging.error("CSV file not found: %s. Run `./feed.py -c` first !"%csv_fn)
        return None

    if not os.path.isdir(outPath):
        subprocess.call(["mkdir", "-p", outPath])

    cctv_url = df.cctv_url[df.cctvid == token].iloc[0]
    idx = 0
    for img in getImg( cctv_url ):
        fn = time.time()
        logging.info ("Imgs:%s"%getRetrivedImgCnt(exportPath, token))
        cv2.imwrite("%s/%s.jpg"%(outPath, fn), img)
        idx += 1
        if (idx%50) == 0:
            logging.info("Writed image over %s"%idx)

def getDist(frm, to):
    return great_circle(frm, to).km

def mkCSV(csv_fn):
    soup = getGZFile2Soup("http://tisvcloud.freeway.gov.tw/vd_info.xml.gz")
    vds = {}
    for info in soup.find_all("Info"):
        vds[ info.attrs['vdid'] ] = (float(info.attrs['px']), float(info.attrs['py']))

    soup_value = getGZFile2Soup("http://tisvcloud.freeway.gov.tw/cctv_value.xml.gz")

    cctv_urls = {}
    for info in soup_value.find_all("Info"):
        cctv_urls[ info.attrs['cctvid'] ] = info.attrs['url']

    soup = getGZFile2Soup("http://tisvcloud.freeway.gov.tw/cctv_info.xml.gz")
    cctvs = {}
    for info in soup.find_all("Info"):
        cctvs[ info.attrs['cctvid'] ] = (float(info.attrs['px']), float(info.attrs['py']))


    logging.info("Calculating Geo-distance:")
    cctv_2_vd = {}
    for t_cctv in cctvs.keys():
        min_dist = 9999
        min_vdid = ""
        pxy_frm = cctvs[t_cctv]
        for vdid in vds.keys():
            if not t_cctv.split("-")[2] == vdid.split("-")[2]:
                continue
            dist = getDist(pxy_frm, vds[vdid])
            if min_dist > dist:
                min_vdid = vdid
                min_dist = dist
        cctv_2_vd[ t_cctv ] = { min_vdid: min_dist }

    logging.info("Preparing DataFrame...")
    df = pd.DataFrame(columns=['cctvid', 'vdid', 'dist', 'cctv_url'])
    for i in range(len(cctv_2_vd.keys())):
        cctv = cctv_2_vd.keys()[i]
        vd = cctv_2_vd[cctv].keys()[0]
        dist = cctv_2_vd[cctv][vd]
        url = cctv_urls[cctv]
        df.loc[i] = [cctv, vd, dist, url ]

    df.to_csv(csv_fn, encoding="UTF8")
    logging.info("Done write to %s "%csv_fn)

def list_token(fn):
    df = pd.read_csv(fn)
    print (df.cctvid)


def getSpeedFile(dest):
    url = "http://tisvcloud.freeway.gov.tw/vd_value.xml.gz"
    fn = int(time.time())
    outPath = "%s/speed"%(dest)
    if not os.path.isdir(outPath):
        subprocess.call(["mkdir", "-p", outPath])
    dest_fn = "%s/%s_%s"%(outPath, fn, url.split("/")[-1])
    print (dest_fn, url)
    subprocess.call(["wget", "-O", dest_fn, url])

def main():
    parser = OptionParser()
    parser.add_option('-d', '--dest_path', dest='dest',
            default="./cctv_imgs",
            help='destination path for export jpeg files w/ auto-mkdir, default="./cctv_imgs"')
    parser.add_option('-t', '--cctv_token', dest='token',
            default="nfbCCTV-N1-N-90.01-M",
            help='token for cctvid in tisv xml, default="nfbCCTV-N1-N-90.01-M"')
    parser.add_option('-f', '--csv_file', dest='file',
            default="cctvid-vd-dest-cctv_url.csv",
            help="read csv file for current cctvid-vd-dest-cctv_url information, default=cctvid-vd-dest-cctv_url.csv")
    parser.add_option('-s', '--speed_file', dest='speed',
            default=False, action="store_true",
            help='Periodically get 1min average speed stats from "http://tisvcloud.freeway.gov.tw/vd_value.xml.gz" and add timestamp on it, default: store to "speed/" under -d path')
    parser.add_option('-l', '--list_only', dest='list_only',
            default=False, action="store_true",
            help='list cctvid in "cctvid-vd-dest-cctv_url.csv"')
    parser.add_option('-c', '--csv_only', dest='csv_only',
            default=False, action="store_true",
            help='making "cctvid-vd-dest-cctv_url.csv" file (see -f) and NOT work with -d -t, default=False')

    (options, args) = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if options.csv_only:
        logging.info ("Making csv to %s"%(options.file))
        mkCSV(options.file)
    elif options.list_only:
        list_token(options.file)
    elif options.speed:
        print ("get speed file, save to %s"%options.dest)
        starttime=time.time()
        interval = 60.0
        while True:
            getSpeedFile(options.dest)
            time.sleep(interval - ((time.time() - starttime) % interval))
	signal.signal(signal.SIGINT, signal_handler)
        signal.pause()
    else:
        print ("get image stream from CCTVID: %s. Please use while loop in bash to keep retrieving images. \neg: while true; do ./feed.py; done;"%(options.token))
        getImgStream(options.file, options.dest, options.token)
        pass


if __name__ == "__main__":
    main()
