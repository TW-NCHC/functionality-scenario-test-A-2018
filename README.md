# NCHC scenario test "Task A:Visual-Speed Detection"

![Diagram](https://snag.gy/0QITS9.jpg)

目標

This is toolkits for deploy **Task A:Visual-Speed Dection**, including following scripts
- (T1) [feed.py](feed.py) 取得即時路況資料影像，並轉換存為以時間戳記為檔名的即時影像
- (T2) [prepare.py](prepare.py) 影像資料集轉換，此外針對低速資料集會進行擴展訓練資料工作。
- (T3) [train.py](train.py) 可用於建模及預測，為 CRNN (Convolutional Recurrent Neural Network) 模型。
- (T4) [pre-train model weight](https://drive.google.com/file/d/1sOK32x8buCID0mxAoz8g5o5p-YoXmjSe/view?usp=sharing) (keras save model)
- (T5) [image samples](https://drive.google.com/file/d/1c7woFGnUlsdYvAyZ6SLR0rNyqEt4y18q/view?usp=sharing) (600+ 分鐘影像資料集，每分鐘有30張維度為 240, 352 JPEG 照片)、[cctvid-vd-dest-cctv_url.csv](cctvid-vd-dest-cctv_url.csv) (即時影像及車速偵測器地理對應表)、[requirements.txt](requirements.txt)。

## 整合性服務測試目標：

- 處理 路況資料庫 http://tisvcloud.freeway.gov.tw/ 中 cctv_info.xml.gz 所有即時影像之各路段車速現況預測。
- 整合系統必需能自動佈署並架設及掛載必要的運算及儲存資源。

## 整合性服務測試步驟：

### Step 1: 取得資料 使用(T1) 取得即時路況影像資料及車速資料。
- 透過(T1) 工具，取得即時影像資料轉換後存放在 object storage 並以 HDFS 儲存協定存取JPEG 影像，供後續車速建模及預測使用。
- 每 1 分鐘至「即時資料庫」 中取得各車速偵測器之平均車速數值。
- (非必要) 「大資料分析平台」可以在資料尚未進行車速建模之前進行任何資料清理與轉換等工作，以降低車速建模時的處理時間。

### Step 2: 影像資料轉換
-  以 HDFS 儲存協定存取JPEG 影像，並使用 (T2) 工具將影像資料轉換為訓練資料集，放入高性能檔案系統。
- 資料轉換動作必需以 container 方式執行，並在深層類神經模型訓練之前完成。

### Step 3: 依即時路況影像資料進行車速建模
- 訓練周期為每12小時建模一次。
- 資料完成搬移後，進行深層類神經模型訓練。
- 使用 (T3) 工具進行車速預測之深層類神經模型建立，並留存 20 epoch 中最佳準確結果之類神經權重，做為車速預測結果。
- 最佳準確結果之類神經權重必需存放於 object storage 中，供後續預測時取用。

### Step 4:進行車速預測
- 以 HDFS 儲存協定存取JPEG 影像，並使用 (T2) 工具將影像資料轉換為 inference 資料集。
- 使用最佳準確結果之類神經權重進行車速預測，並以 Restful API 方式即時回傳預測結果。

----

### 工具說明

#### (T1) feed.py
使用說明 `./feed.py -h`

```
Usage: feed.py [options]

Options:
  -h, --help            show this help message and exit
  -d DEST, --dest_path=DEST
                        destination path for export jpeg files w/ auto-mkdir,
                        default="./cctv_imgs"
  -t TOKEN, --cctv_token=TOKEN
                        token for cctvid in tisv xml,
                        default="nfbCCTV-N1-N-90.01-M"
  -f FILE, --csv_file=FILE
                        read csv file for current cctvid-vd-dest-cctv_url
                        information, default=cctvid-vd-dest-cctv_url.csv
  -s, --speed_file      Periodically get 1min average speed stats from
                        "http://tisvcloud.freeway.gov.tw/vd_value.xml.gz" and
                        add timestamp on it, default: store to "speed/" under
                        -d path
  -l, --list_only       list cctvid in "cctvid-vd-dest-cctv_url.csv"
  -c, --csv_only        making "cctvid-vd-dest-cctv_url.csv" file (see -f) and
                        NOT work with -d -t, default=False

```

使用流程：
1. 以 `./feed.py -c` 先建立影像(CCTVID)對車速(VDID)的資料表 `cctvid-vd-dest-cctv_url.csv`
1. 使用 `./feed.py -l` 查詢現有的串流影像代號
1. `while true; do ./feed.py -t $TOKEN -d $DEST; done; ` 進行抓取來自$TOKEN代號的即時影像並存到 `$DEST` 目錄中的 `$TOKEN` 子目錄
1. 使用 `./feed.py -s` 每 60秒抓取一次車速資料存到 `$DEST` 目錄中的 `speed/` 子目錄

輸出結果如下：
```
# ./feed.py
get image stream from CCTVID: nfbCCTV-N1-N-90.01-M. Please use while loop in bash to keep retrieving images.
eg: while true; do ./feed.py; done;
INFO:root:Imgs:201727
INFO:root:Imgs:201728
INFO:root:Imgs:201729
...
# ./feed.py -s
get speed file, save to ./cctv_imgs
./cctv_imgs/speed/1520550609_vd_value.xml.gz http://tisvcloud.freeway.gov.tw/vd_value.xml.gz
--2018-03-09 07:10:09--  http://tisvcloud.freeway.gov.tw/vd_value.xml.gz
Resolving tisvcloud.freeway.gov.tw (tisvcloud.freeway.gov.tw)... 210.241.131.253
Connecting to tisvcloud.freeway.gov.tw (tisvcloud.freeway.gov.tw)|210.241.131.253|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: unspecified [application/gzip]
Saving to: './cctv_imgs/speed/1520550609_vd_value.xml.gz'

./cctv_imgs/speed/1520     [ <=>                      ]  84.91K  --.-KB/s    in 0.01s

2018-03-09 07:10:09 (8.58 MB/s) - './cctv_imgs/speed/1520550609_vd_value.xml.gz' saved [86943]
```


#### (T2) prepare.py
使用說明 `./prepare.py -h`
```
Usage: prepare.py [options]

Options:
  -h, --help            show this help message and exit
  -i IMG_PATH, --image_path=IMG_PATH
                        this path stores all cctv images,
                        default="./cctv_imgs"
  -d DATASETS_PATH, --datasets_path=DATASETS_PATH
                        destination path for pickled(dill) datasets,
                        default="./datasets"
  -c CCTVID, --cctvid=CCTVID
                        preparing datasets for "nfbCCTV-N1-N-90.01-M",
                        default="nfbCCTV-N1-N-90.01-M"
  -t, --test            test run under 10 speed xml.gz files, default=False
```

1. 測試功能 `prepare.py -t ` 會先處理10分鐘的車速及影像資料。
1. `prepare.py -i $IMG_PATH -d $DATASETS_PATH -c $CCTVID` 整理在 `$IMG_PATH/$CCTVID` 目錄中的影像照片並產出訓練資料集到`$DATASETS_PATH` 中。

輸出結果如下：
```
# ./prepare.py -t
Using TensorFlow backend.
INFO:root:Preparing data for vdid: nfbVD-N1-N-89.990-M-LOOP
INFO:root:Scanning image files in directory: ./cctv_imgs/nfbCCTV-N1-N-90.01-M
INFO:root:Found image files: 201727
Loading Speed: 100%|#####################################| 10/10 [00:00<00:00, 57.60it/s]
Find Images: 100%|############################| 201727/201727 [00:08<00:00, 23674.47it/s]
Mapping Speed: 100%|########################| 201727/201727 [00:00<00:00, 1478373.79it/s]
Make DataFrame: 100%|###################################| 10/10 [00:00<00:00, 404.07it/s]
Image Count in DataFrame
count       4
unique      4
top       551
freq        1
Name: image_count, dtype: int64
Mapping X&y: 100%|#########################################| 4/4 [00:01<00:00,  2.17it/s]
INFO:root:       --== dimensions ==--
X dim:(12, 30, 120, 176, 1), Y dim:(12,)
Speed(y) distributions:
 90.0     9
100.0    3
dtype: int64
Saving Sample: 100%|#####################################| 12/12 [00:00<00:00, 44.43it/s]
```

#### (T3) train.py
使用說明 `./train.py -h`
本項程式會使用到 GPU 運算。

```
Usage: train.py [options]

Options:
  -h, --help            show this help message and exit
  -i IMG_PATH, --image_path=IMG_PATH
                        this path stores all cctv images,
                        default="./cctv_imgs"
  -w WEIGHT_PATH, --weight_path=WEIGHT_PATH
                        destination path for weights, default="./weights"
  -d DATASET_PATH, --datasets_path=DATASET_PATH
                        destination path for pickled(dill) datasets,
                        default="./datasets"
  -e EPOCHS_NUM, --epochs_num=EPOCHS_NUM
                        epochs_num for training, default=20
  -l LOT_SIZE, --lot_size=LOT_SIZE
                        training data lot 0 for all, any number >0 will be
                        limited to that size, default=0
  -p SERVE_PORT, --serve_port=SERVE_PORT
                        python flask bind port for requesting model results,
                        default=80
  -a ADDRESS_IP, --address_ip=ADDRESS_IP
                        python flask bind ip address, default="172.17.0.2"
  -t TOKEN, --cctv_token=TOKEN
                        token for cctvid in tisv xml,
                        default="nfbCCTV-N1-N-90.01-M"
  -s, --is_serve        picking best model weight to predict by latest 30
                        images.  -i can define img_path for detecting new
                        images, default=False

```

##### 訓練部份
1. 測試訓練功能 `./train.py -e 2 -l 10` 會從預設資料集 $DATASET_PATH/$TOKEN 中取出預先整理好的資料集共 10 筆，並進行訓練 2次 (epochs)
1. 完整資料集訓練資料集 `./train.py -e 20 -l 0 -d $DATASET_PATH -t $TOKEN -w $WEIGHT_PATH`

輸出結果如下：
```
----- ===== TRAINING MODEL ===== -----
Loading DS: 100%|#######################################################################################################################| 6276/6276 [00:46<00:00, 134.88it/s]
Speed(y) distributions:
 90.0     1644
80.0     1628
100.0    1077
40.0      781
60.0      759
110.0     182
20.0      143
120.0      54
130.0       8
dtype: int64
(6276, 9)
---------- ========== Dataset Info ========== ----------
X_train_raw shape: (6276, 30, 120, 176, 1)
6276 train samples
6276 test samples (6276, 9)
model paerms: 16496297
Train on 5648 samples, validate on 628 samples
Epoch 1/20
2018-03-09 06:50:36.280698: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-03-09 06:50:36.591673: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1344] Found device 0 with properties:
name: Tesla P100-PCIE-12GB major: 6 minor: 0 memoryClockRate(GHz): 1.3285
pciBusID: 0000:03:00.0
totalMemory: 11.91GiB freeMemory: 11.62GiB
2018-03-09 06:50:36.591750: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1423] Adding visible gpu devices: 0
2018-03-09 06:50:36.906156: I tensorflow/core/common_runtime/gpu/gpu_device.cc:911] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-03-09 06:50:36.906270: I tensorflow/core/common_runtime/gpu/gpu_device.cc:917]      0
2018-03-09 06:50:36.906309: I tensorflow/core/common_runtime/gpu/gpu_device.cc:930] 0:   N
2018-03-09 06:50:36.906730: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 11250 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-12GB, pci bus id: 0000:03:00.0, compute capability: 6.0)
5648/5648 [==============================] - 244s 43ms/step - loss: 1.6489 - acc: 0.3360 - val_loss: 1.2298 - val_acc: 0.4379
Epoch 2/20
5648/5648 [==============================] - 238s 42ms/step - loss: 1.1393 - acc: 0.5239 - val_loss: 1.0184 - val_acc: 0.5557
Epoch 3/20
5648/5648 [==============================] - 238s 42ms/step - loss: 0.9595 - acc: 0.6000 - val_loss: 1.0691 - val_acc: 0.5557
Epoch 4/20
5648/5648 [==============================] - 239s 42ms/step - loss: 0.8312 - acc: 0.6650 - val_loss: 0.7383 - val_acc: 0.6799
Epoch 5/20
5648/5648 [==============================] - 238s 42ms/step - loss: 0.7010 - acc: 0.7190 - val_loss: 0.7217 - val_acc: 0.7309
Epoch 6/20
5648/5648 [==============================] - 238s 42ms/step - loss: 0.5784 - acc: 0.7762 - val_loss: 0.6117 - val_acc: 0.7309
Epoch 7/20
5648/5648 [==============================] - 241s 43ms/step - loss: 0.4826 - acc: 0.8205 - val_loss: 0.5847 - val_acc: 0.7739
...
```

##### 預測部份
1. 測試預測功能 `./train.py -s -i $IMG_PATH` 會自動載入在 $WEIGHT_PATH 目錄中最好的模型結果，從 `$IMG_PATH` 取得最新的車速影像(使用 T1 工具取得)的 30張進行預測。預測結果會 `http://$ADDRESS_IP:%SERVE_PORT` 以文字顯示。

僅使用 CPU 進行預測的輸出結果 command line 如下：
```
# ./train.py -s
Using TensorFlow backend.
----- ===== SERVING MODEL ===== -----
Loading Best Acc-Model: weights.20-0.9628.hdf5
2018-03-09 09:23:59.318723: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-03-09 09:23:59.600408: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1344] Found device 0 with properties:
name: Tesla P100-PCIE-12GB major: 6 minor: 0 memoryClockRate(GHz): 1.3285
pciBusID: 0000:03:00.0
totalMemory: 11.91GiB freeMemory: 11.62GiB
2018-03-09 09:23:59.600471: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1423] Adding visible gpu devices: 0
2018-03-09 09:23:59.863239: I tensorflow/core/common_runtime/gpu/gpu_device.cc:911] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-03-09 09:23:59.863325: I tensorflow/core/common_runtime/gpu/gpu_device.cc:917]      0
2018-03-09 09:23:59.863358: I tensorflow/core/common_runtime/gpu/gpu_device.cc:930] 0:   N
2018-03-09 09:23:59.863703: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 11250 MB memory) -> physical GPU (device: 0, name: Tesla P100-PCIE-12GB, pci bus id: 0000:03:00.0, compute capability: 6.0)
INFO:werkzeug: * Running on http://172.17.0.2:80/ (Press CTRL+C to quit)
INFO:root:Scanning image files in directory: ./cctv_imgs/nfbCCTV-N1-N-90.01-M
Loading IMGs: 100%|##########################################################################################################################| 30/30 [00:00<00:00, 60.60it/s]
dataset for predicting:  (1, 30, 120, 176, 1)
Model building time: 2.6704 seconds.
INFO:werkzeug:172.17.0.2 - - [09/Mar/2018 09:24:12] "GET / HTTP/1.0" 200 -

```

以 w3m 連接 http://172.17.0.2:80/ 輸出結果如下：
```
Speed Predicting Result for (2018-03-09 07:09:36) is 90.0.
```



### 環境參數
- python 函式庫，請參考 `requirements.txt`
- Tesla P100-PCIE + CUDA V9.0.176

```
# /usr/local/cuda/bin/nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176
# nvidia-smi
Fri Mar  9 07:38:40 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.81                 Driver Version: 384.81                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:03:00.0 Off |                    0 |
| N/A   41C    P0    37W / 250W |  11763MiB / 12193MiB |     88%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla P100-PCIE...  Off  | 00000000:82:00.0 Off |                    0 |
| N/A   28C    P0    24W / 250W |     10MiB / 12193MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+

```

### 系統效能參數

- X dim: (6276, 30, 120, 176, 1); Y dim: (6276, 9)
- model paerms: 16496297
- (5648/238)*30=711.9 images/sec 

![NCHC Infra-Serv Team](https://snag.gy/WpiKMB.jpg "NCHC Infra-Serv Team")


