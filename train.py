#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    _/      _/    _/_/_/  _/    _/    _/_/_/
   _/_/    _/  _/        _/    _/  _/
  _/  _/  _/  _/        _/_/_/_/  _/
 _/    _/_/  _/        _/    _/  _/
_/      _/    _/_/_/  _/    _/    _/_/_/
https://github.com/TW-NCHC/functionality-scenario-test-A-2018

This program can train a CRNN (convolutional recurrent neural network)
and provide web service for predicting

@author August Chao <AugustChao@narlabs.org.tw>
"""

from __future__ import print_function
import warnings
warnings.simplefilter(action="ignore",category=DeprecationWarning)
warnings.simplefilter(action="ignore",category=FutureWarning)

import logging
import dill as pickle
from optparse import OptionParser

import time
import datetime
import os
import re
import subprocess
import numpy as np
import pandas as pd
import keras.backend as K
from tqdm import tqdm
from joblib import Parallel, delayed
from keras.models import Model
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import RMSprop
from keras.layers import MaxPooling2D, Conv2D, RepeatVector, LSTM, multiply, Permute
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Input
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.utils import np_utils
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.preprocessing import LabelBinarizer

os.environ["JOBLIB_TEMP_FOLDER"]="./tmp"   # see issue https://goo.gl/4YZJUH
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from flask import Flask
app = Flask("NCHC_Visual_Speed")

def create_model(input_shape=(30, 120, 176, 1), rnn_units=128, cnn_units=32, num_gpu=1, nb_category=10):
    # define our time-distributed setup
    inp = Input(shape=input_shape)

    x = TimeDistributed(Conv2D(cnn_units, (3, 3), padding='same', activation='relu'))(inp)
    x = TimeDistributed(Conv2D(cnn_units, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = Dropout(.5)(x)

    x = TimeDistributed(Conv2D(cnn_units, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(cnn_units, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = Dropout(.5)(x)

    x = TimeDistributed(Flatten())(x)
    x = GRU(units=rnn_units, return_sequences=True)(x)
    x = Dropout(.5)(x)

    x = TimeDistributed(Flatten())(x)
    x = GRU(units=rnn_units, return_sequences=True)(x)
    x = Dropout(.5)(x)

    x = GRU(units=rnn_units, return_sequences=False)(x)

    x = Dropout(.5)(x)
    x = Dense(nb_category, activation='softmax')(x)

    opt_adm = Adam()
    model = Model(inp, x)
    if num_gpu > 1:
        model = multi_gpu_model(model, gpus=num_gpu) # gpus
    model.compile(optimizer=opt_adm, loss='categorical_crossentropy', metrics=['accuracy'])
    print ("model paerms: %s"%model.count_params())
    return model

def get_worker(mdir, fn):
    with open("%s/%s"%(mdir, fn), 'r') as handle:
        return pickle.load(handle)

def getDataSet(dataset_path = "./datasets", size=100):
    fns = os.listdir(dataset_path)
    np.random.shuffle(fns)
    if int(size) > 0 :
        fns = fns[:int(size)]

    datasets = Parallel(n_jobs=-1)(
        delayed(get_worker)(dataset_path, fns[i]) for i in tqdm( range(len(fns)),
                                                                 ascii=True,
                                                                 desc="Loading DS", ) )
    return datasets


def train( dataset_path = "./datasets",
           data_size = 100,
           nb_epochs=2,
           weight_path="./weights"):

    all_data = getDataSet(dataset_path, data_size)
    train_x = np.array([ x for (x,y) in all_data ])
    pre_train_y = np.array([ y for (x,y) in all_data ])
    print ("Speed(y) distributions: \n", "%s"%pd.Series(pre_train_y).value_counts())

    encoder = LabelBinarizer()
    train_y = encoder.fit_transform([ "%s"%(x) for x in pre_train_y])
    print( train_y.shape)

    np.random.seed(6813)
    K.set_image_dim_ordering('tf')

    # define some run parameters
    batch_size = 8
    maxToAdd = train_x.shape[1]
    hidden_units = 50
    pool_size = (2, 2)
    size_x, size_y = train_x.shape[2], train_x.shape[3]

    print("--"*5, "=="*5, "Dataset Info", "=="*5, "--"*5)
    print('X_train_raw shape:', train_x.shape)
    print(train_x.shape[0], 'train samples')
    print(len(train_y), 'test samples', train_y.shape)

    # creates weight_path
    if not os.path.isdir(weight_path):
        subprocess.call(["mkdir", "-p", weight_path])
    callbacks = [
        ModelCheckpoint(filepath="%s/weights.{epoch:02d}-{acc:.4f}.hdf5"%(weight_path), save_best_only=True, monitor='acc', verbose=0),
        CSVLogger(os.path.join(".", "training.log")),
    ]

    m_model = create_model(input_shape=train_x.shape[1:], nb_category=train_y.shape[1], rnn_units=128, cnn_units=32, num_gpu=1)
    m_model.fit(train_x, train_y, batch_size=batch_size, epochs=nb_epochs, validation_split=0.1, shuffle=True, verbose=1, callbacks=callbacks)

    print("--"*5, "=="*5, "Model Results", "=="*5, "--"*5)
    scores = m_model.evaluate(train_x, train_y, verbose=0)
    print("Model %s: %.2f%%" % (m_model.metrics_names[1], scores[1]*100))

    model_json = m_model.to_json()
    model_fn = "%s/model.json"%(weight_path)
    with open( model_fn, "w") as json_file:
        json_file.write(model_json)
    print("Model Saved: %s"%(model_fn))

    encoder_fn = "%s/encoder.pkl"%(weight_path)
    with open( encoder_fn, "w") as fp:
        pickle.dump(encoder, fp)
    print("Target Encoder Saved: %s"%(encoder_fn))

    print("Model Weights are: \n", ",\n ".join(sorted(os.listdir(weight_path))))

def getImgFrmFn(fn, mdir):
    return img_to_array(load_img("%s/%s"%(mdir, fn),
                        target_size=(120, 176),
                        grayscale=True,
                        interpolation="hamming" ))/255.

def getBestModel():
    global _weight_path_
    global _img_path_
    try:
        # load json and create model
        json_file = open("%s/model.json"%(_weight_path_), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
    except:
        print("Error while loading model from directory: %s"%(_weight_path_))
        return None

    # load weights into new model
    best_weight = .0
    best_weight_fn = ""
    for wfn in os.listdir(_weight_path_):
        if re.search("hdf5$", wfn):
            acc_val = float(wfn.split("-")[1].replace(".hdf5", ""))
            if acc_val > best_weight:
                best_weight = acc_val
                best_weight_fn = wfn
    print("Loading Best Acc-Model: %s"%(best_weight_fn))
    loaded_model.load_weights(_weight_path_ + "/" + best_weight_fn)
    return loaded_model

def predictor(weight_path="./weights", opath="./cctv_imgs", token = "nfbCCTV-N1-N-90.01-M"):
    global _model_

    DIR = "%s/%s"%(opath, token)
    logging.info("Scanning image files in directory: %s"%(DIR))
    all_images = sorted([ name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))], reverse=True)[:30]

    last_img_time = datetime.datetime.fromtimestamp(float(all_images[0].replace(".jpg", ""))).strftime('%Y-%m-%d %H:%M:%S')

    pre_set = Parallel(n_jobs=-1)(delayed(getImgFrmFn)(all_images[i], DIR) for i in tqdm( range(len(all_images)),
                                                                 ascii=True,
                                                                 desc="Loading IMGs", ) )
    pre_set = np.array([pre_set])
    print ("dataset for predicting: ",pre_set.shape)
    res = _model_.predict(pre_set)

    encoder_fn = "%s/encoder.pkl"%(weight_path)
    with open( encoder_fn, "r") as fp:
        encoder = pickle.load(fp)

    return "Speed Predicting Result for (%s) is %s."%(last_img_time, encoder.inverse_transform(res)[0])


@app.route("/")
def getCurrentSpeed():
    global _weight_path_
    global _img_path_
    start = time.time()
    res = predictor(weight_path=_weight_path_, opath=_img_path_)
    end = time.time()
    print( "Model building time: %.4f seconds."%(end - start) )
    return res

def main():
    parser = OptionParser()
    parser.add_option('-i', '--image_path', dest='img_path',
            default="./cctv_imgs",
            help='this path stores all cctv images, default="./cctv_imgs"')
    parser.add_option('-w', '--weight_path', dest='weight_path',
            default="./weights",
            help='destination path for weights, default="./weights"')
    parser.add_option('-d', '--datasets_path', dest='dataset_path',
            default="./datasets",
            help='destination path for pickled(dill) datasets, default="./datasets"')
    parser.add_option('-e', '--epochs_num', dest='epochs_num',
            default=20,
            help='epochs_num for training, default=20')
    parser.add_option('-l', '--lot_size', dest='lot_size',
            default=0,
            help='training data lot 0 for all, any number >0 will be limited to that size, default=0')
    parser.add_option('-p', '--serve_port', dest='serve_port',
            default=80,
            help='python flask bind port for requesting model results, default=80')
    parser.add_option('-a', '--address_ip', dest='address_ip',
            default='172.17.0.2',
            help='python flask bind ip address, default="172.17.0.2"')
    parser.add_option('-t', '--cctv_token', dest='token',
            default="nfbCCTV-N1-N-90.01-M",
            help='token for cctvid in tisv xml, default="nfbCCTV-N1-N-90.01-M"')
    parser.add_option('-s', '--is_serve', dest='isServe',
            default=False, action="store_true",
            help='picking best model weight to predict by latest 30 images. \n-i can define img_path for detecting new images, default=False')

    (options, args) = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if options.isServe:
        print ("-"*5, "="*5, "SERVING MODEL", "="*5, "-"*5)
        global _weight_path_
        global _img_path_
        global _model_

        _weight_path_= options.weight_path
        _img_path_   = options.img_path
        _model_ = getBestModel()

        app.run(host=options.address_ip, port=options.serve_port)

    else:
        print ("-"*5, "="*5, "TRAINING MODEL", "="*5, "-"*5)
        start = time.time()
        train(  options.dataset_path + "/" + options.token,
                data_size = options.lot_size,
                nb_epochs=int(options.epochs_num),
                weight_path=options.weight_path )
        end = time.time()
        print( "Model building time: %.4f seconds."%(end - start) )


if __name__ == "__main__":
    main()
