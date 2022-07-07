# Real-RawVSR: Real-World Raw Video Super-Resolution with a Benchmark Dataset (Real-RawVSR)

This repository contains official implementation of Real-RawVSR: Real-World Raw Video Super-Resolution with a Benchmark Dataset, by Huanjing Yue, Zhiming Zhang, and Jingyu Yang.

<p align="center">
  <img width="800" src="https://github.com/zmzhang1998/Real-RawVSR/blob/main/images/framework.jpg">
</p>

## Paper

## Dataset

### Real-RawVSR Dataset

<p align="center">
  <img width="600" src="https://github.com/zmzhang1998/Real-RawVSR/blob/main/images/dataset.jpg">
</p>

You can download our dataset from Baidu Netdisk. For each magnification scale, there are 150 video pairs and each video contains about 50 frames (To make the movements between neighboring frames more obvious, for each video, we extract frames from the original 150 frames with a step size of three, resulting in a 50 frame sequence). The Bayer pattern of raw data is RGGB, the black level is 2047, the white level is 16200.

### Copyright ###

The Real-RawVSR dataset is available for the academic purpose only. Any researcher who uses the CRVD dataset should obey the licence as below:

All of the Real-RawVSR Dataset (data and software) are copyright by [Intelligent Imaging and Reconstruction Laboratory](http://tju.iirlab.org/doku.php), Tianjin University and published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License. This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license.

This dataset is for non-commercial use only. However, if you find yourself or your personal belongings in the data, please contact us, and we will immediately remove the respective images from our servers.

## Code

### Installation

Our model is trained and tested through the environment in the file [requirements.txt](https://github.com/zmzhang1998/Real-RawVSR/blob/main/requirements.txt) on Ubuntu, run the following command to install:
  ```
  pip install -r requirements.txt
  ```
  
Deformable convlution setup,run:
  ```
  cd ./models/dcn/
  python setup.py develop
  ```

### Prepare Data

Download Real-RawVSR dataset. Put them in the dataset folder.

### Test

Download trained model from Baidu Netdisk. Put them in the weight_checkpoints folder.
Test 4X data, run:
  ```
  python test.py --gpu_id 0 --scale 4 --save_image True
  ```

### Train
Train 4X data, run:
  ```
  python train.py --gpu_id 0 --scale 4 --continue_train False
  ```

## Acknowledgement

Our work and implementations are inspired by following projects:<br/>
[EDVR](https://github.com/xinntao/EDVR)<br/>
[RViDeNet](https://github.com/cao-cong/RViDeNet)<br/>
[RawVSR](https://github.com/proteus1991/RawVSR)<br/>
[EBSR](https://github.com/Algolzw/EBSR)<br/>
