#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    _/      _/    _/_/_/  _/    _/    _/_/_/
   _/_/    _/  _/        _/    _/  _/
  _/  _/  _/  _/        _/_/_/_/  _/
 _/    _/_/  _/        _/    _/  _/
_/      _/    _/_/_/  _/    _/    _/_/_/
https://github.com/TW-NCHC/functionality-scenario-test-A-2018

This program prepares dataset for training CRNN model

@author August Chao <AugustChao@narlabs.org.tw>
"""

from __future__ import print_function
from __future__ import print_function
import warnings
warnings.simplefilter(action="ignore",category=DeprecationWarning)
warnings.simplefilter(action="ignore",category=FutureWarning)

import sys
import os
import gzip
import logging
import numpy as np
import pandas as pd
import dill as pickle
import subprocess
import time
from tqdm import tqdm
from bs4 import BeautifulSoup
from cStringIO import StringIO
from dateutil import parser
from joblib import Parallel, delayed
from optparse import OptionParser
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

os.environ["JOBLIB_TEMP_FOLDER"]="./tmp"   # see issue https://goo.gl/4YZJUH

time_2_stamp = lambda x: time.mktime(parser.parse(x).timetuple())

def getGZFile2Soup(fn):
    with open(fn, 'r') as speed_gz:
        compressedFile = StringIO(speed_gz.read())
        decompressedFile = gzip.GzipFile(fileobj=compressedFile)
        try:
            str_xml = decompressedFile.read()
            return BeautifulSoup(str_xml, 'xml')
        except:
            return None

def getAvgSpeed(opath, token, vdid, spgz):
    try:
        soup = getGZFile2Soup("%s/speed/%s"%(opath, spgz))
    except IOError:
        soup = None

    if type(soup) == type(None):
        return (None, None, None)

    for token_vd in soup.find_all("Info", attrs={"vdid":vdid}):
        if token_vd.attrs['status'] == "0":
            timestamp = time_2_stamp(token_vd.attrs["datacollecttime"])
            speeds = []
            for lane in token_vd.find_all("lane"):
                avg_speed = lane.attrs['speed']
                vol = 0
                for car in lane.find_all("cars"):
                    vol += int(car.attrs['volume'])
                if vol > 0:
                    speeds.append(int(avg_speed))
            if len(speeds)==0:
                return (None, None, None)
            else:
                return (vdid, timestamp, np.average(speeds))
    return (None, None, None)

def prepareData(opath, token, vdid, isTest):
    speed_path = "%s/speed/"%(opath)
    fns = sorted(os.listdir(speed_path))
    if isTest:
        fns = fns[:10]
    vdid_time_speed = Parallel(n_jobs=-1)(
            delayed(getAvgSpeed)(opath, token, vdid, fns[i]) for i in tqdm( range(len(fns)),
                                                                            ascii=True,
                                                                            desc="Loading Speed", ) )
    return vdid_time_speed

lots = []
def inRange(point):
    global lots
    try:
        for (tkn, tm, speed) in lots:
            try:
                if (float(tm) - float(point))>0 and (float(tm) - float(point)) < 60.0:
                    return (point, tm)
            except:
                pass
    except:
        pass
    return (point, None)

def preparing(csv_fn = "./cctvid-vd-dest-cctv_url.csv", opath = "./cctv_imgs", token = "nfbCCTV-N1-N-90.01-M", dest = "./datasets", isTest=False):
    global lots
    DIR = "%s/%s"%(opath, token)

    # get vdid info.
    try:
        df_csv = pd.read_csv(csv_fn)
        vdids = df_csv[df_csv.cctvid == token].vdid
        if len(vdids)==0:
            logging.error("Error token='%s' is not found in ./cctvid-vd-dest-cctv_url.csv,\n check again or run `./feed.py -c`."%(token))
            return None
        vdid = vdids.iloc[0]

        logging.info("Preparing data for vdid: %s"%(vdid))
    except Exception as inst:
        print (inst)
        print ("!!!")
        logging.error("error while reading: %s"%(csv_fn))
        return None

    # scanning image files
    logging.info("Scanning image files in directory: %s"%(DIR))
    all_images = dict([( float(name.replace(".jpg", "")), name) for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    logging.info("Found image files: %s"%(len(all_images)))

    # get real speed data
    lots = prepareData(opath, token, vdid, isTest)
    lots = [ (x, y, z) for (x, y, z) in lots if not type(z)==type(None)]

    # with 60 secs
    img_keys = all_images.keys()
    imgCate = Parallel(n_jobs=-1)(delayed(inRange)(img_keys[i]) for i in tqdm(range(len(img_keys)),
                                                                              ascii=True,
                                                                              desc="Find Images"))
    cate_2_img = {}
    for i in tqdm(range(len(imgCate)), ascii=True, desc="Mapping Speed"):
        ele = imgCate[i]
        fn_img = ele[0]
        cate = ele[1]
        if type(cate) == type(None):
            pass
        else:
            if not cate in cate_2_img:
                cate_2_img[cate] = []
            cate_2_img[cate].append(fn_img)

    df = pd.DataFrame(columns=['vdid', 'vd_timestamp', 'speed', "image_count", "image_file"])
    idx = 0
    cate_2_img_set = set(cate_2_img.keys())
    saw_timestamp = set()
    for i in tqdm(range(len(lots)), ascii=True, desc="Make DataFrame"):
        lot = lots[i]
        tkn, tm, speed = lot
        if not tm in saw_timestamp and tm in cate_2_img_set:
            df.loc[idx] = [tkn, tm, speed, len(cate_2_img[tm]), ", ".join([ "%s"%x for x in cate_2_img[tm]])]
            idx += 1
            saw_timestamp.add(tm)

    print ("Image Count in DataFrame")
    print (df.image_count.describe())


    datasets = []

    speed_hig = lambda x: float(x - (x%10))
    speed_mid = lambda x: float(x - (x%10))
    speed_low = lambda x: float(x - (x%20))

    _exp_low_speed_, _exp_mid_speed_, _exp_hig_speed_ = True, True, True
    _exp_low_ratio_, _exp_mid_ratio_, _exp_hig_ratio_ = 1., .2, .1


    _size_for_min_ = 30
    _spd_bl_low_ = 80
    _spd_ov_hig_ = 100

    _exp_max_ = 10
    getExpLot = lambda y: _exp_max_*_exp_low_ratio_ if y <= _spd_bl_low_ else _exp_max_*_exp_mid_ratio_ if y <= _spd_ov_hig_ else _exp_max_*_exp_hig_ratio_

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], ascii=True, desc="Mapping X&y"):
        y = float(row['speed'])

        y = speed_low(y) if y <= _spd_bl_low_ else speed_mid(y) if y <= _spd_ov_hig_ else speed_hig(y)

        if not y == 0:
            img_fns = sorted(row['image_file'].split(", "))[:_size_for_min_+_exp_max_] ## dont use too much
            img_fns_size = len(img_fns)

            if img_fns_size < _size_for_min_+_exp_max_:
                continue

            is_exp = False
            if y <= _spd_bl_low_ and _exp_low_speed_:
                is_exp = True
            elif y <= _spd_ov_hig_ and _exp_mid_speed_:
                is_exp = True
            elif _exp_hig_speed_:
                is_exp = True
            else:
                is_exp = False

            if is_exp:
                #logging.info ("!! EXPENING IMAGES for idx: %s, X imgs size:%s, Y :%s"%(idx, img_fns_size, y))
                max_size = int(_size_for_min_+getExpLot(y))+1
                for idy in range(max_size-_size_for_min_):
                    this_lot = img_fns[idy:idy+_size_for_min_]
                    X = Parallel(n_jobs=100, backend="threading")(delayed(getImgFrmFn)(x, DIR) for x in this_lot)
                    datasets.append( (X, y))
            else:
                this_lot = img_fns[:_size_for_min_]
                X = Parallel(n_jobs=100, backend="threading")(delayed(getImgFrmFn)(x, DIR) for x in this_lot)
                datasets.append( (X, y))

    train_x = np.array([x for (x, y) in datasets])
    train_y = np.array([y for (x, y) in datasets ])
    logging.info ("\t --== dimensions ==-- \n"+"X dim:%s, Y dim:%s"%(train_x.shape, train_y.shape))
    print ("Speed(y) distributions: \n", "%s"%pd.Series(train_y).value_counts())


    # check if dest exists
    m_dest = dest + "/" + token
    if not os.path.isdir(m_dest):
        subprocess.call(["mkdir", "-p", m_dest])

    vdid_time_speed = Parallel(n_jobs=-1)(
        delayed(save_xy)(i, train_x[i], train_y[i], dest_dir=m_dest) for i in tqdm( range(train_x.shape[0]),
                                                                            ascii=True,
                                                                            desc="Saving Sample", ) )

def save_xy(i, x, y, dest_dir):
    export_fn = "%s/%02d.pkl"%(dest_dir, i)
    with open(export_fn, 'wb') as handle:
        pickle.dump( (x, y), handle )

_catch_img_ = {}
_catch_img_fn_ = set()

def getImgFrmFn(fn, mdir):
    global _catch_img_
    global _catch_img_fn_
    is_gray = True
    if fn in _catch_img_fn_:
        return _catch_img_[fn]
    else:
        _catch_img_fn_.add(fn)
        _catch_img_[fn] = img_to_array(load_img("%s/%s.jpg"%(mdir, fn),
                                target_size=(120, 176),
                                grayscale=is_gray,
                                interpolation="hamming" ))/255.
        return _catch_img_[fn]


def main():
    parser = OptionParser()
    parser.add_option('-i', '--image_path', dest='img_path',
            default="./cctv_imgs",
            help='this path stores all cctv images, default="./cctv_imgs"')
    parser.add_option('-d', '--datasets_path', dest='datasets_path',
            default="./datasets",
            help='destination path for pickled(dill) datasets, default="./datasets"')
    parser.add_option('-f', '--csv_file', dest='file',
	            default="cctvid-vd-dest-cctv_url.csv",
	            help="read csv file for current cctvid-vd-dest-cctv_url information, default=cctvid-vd-dest-cctv_url.csv")
    parser.add_option('-c', '--cctvid', dest='cctvid',
            default="nfbCCTV-N1-N-90.01-M",
            help='preparing datasets for "nfbCCTV-N1-N-90.01-M", default="nfbCCTV-N1-N-90.01-M"')
    parser.add_option('-t', '--test', dest='isTest',
            default=False, action="store_true",
            help='test run under 10 speed xml.gz files, default=False')

    (options, args) = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    preparing(options.file,options.img_path, options.cctvid, options.datasets_path, options.isTest)


if __name__ == "__main__":
    main()
