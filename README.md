# Real-RawVSR: Real-World Raw Video Super-Resolution with a Benchmark Dataset (Real-RawVSR)

This repository contains official implementation of Real-RawVSR: Real-World Raw Video Super-Resolution with a Benchmark Dataset in ECCV 2022, by Huanjing Yue, Zhiming Zhang, and Jingyu Yang.

<p align="center">
  <img width="800" src="https://github.com/zmzhang1998/Real-RawVSR/blob/main/images/framework.jpg">
</p>

## Paper

[https://arxiv.org/pdf/2209.12475.pdf](https://arxiv.org/pdf/2209.12475.pdf)
## Dataset

### Real-RawVSR Dataset

<p align="center">
  <img width="600" src="https://github.com/zmzhang1998/Real-RawVSR/blob/main/images/dataset.jpg">
</p>

You can download our dataset from [Baidu Netdisk](https://pan.baidu.com/s/1G5_zCt_L_POzwb_mWgpuDA)(key: hxyl). For each magnification scale, there are 150 video pairs and each video contains about 50 frames (To make the movements between neighboring frames more obvious, for each video, we extract frames from the original 150 frames with a step size of three, resulting in a 50 frame sequence). The Bayer pattern of raw data is RGGB, the black level is 2047, the white level is 16200.

### Copyright ###

The Real-RawVSR dataset is available for the academic purpose only. Any researcher who uses the dataset should obey the licence as below:

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

Download Real-RawVSR dataset from [Baidu Netdisk](https://pan.baidu.com/s/1G5_zCt_L_POzwb_mWgpuDA)(key: hxyl). Put them in the dataset folder.

### Test

Download trained model including our network and other networks from [Google Drive](https://drive.google.com/drive/folders/1zBMWiRq352HvurnVDxG0t-_OPVXAwtcQ?usp=sharing). Put them in the weight_checkpoints folder. In particular, the link also contains the pwcnet weight required in the DBSR. Please put it in the models_DBSR folder.

Test our model on $4\times$ data, run:
  ```
  python test.py --model model --gpu_id 0 --scale 4 --save_image True
  ```
  
You can also test other networks on our dataset, such as:

Test EDVR model on $4\times$ data, run:
  ```
  python test_EDVR.py --model model_EDVR --gpu_id 0 --scale 4 --save_image True
  ```

The test commands of other models are similar. Note that you may need to install the packages that are dependent on other models.
### Train
Train 4X data, run:
  ```
  python train.py --model model --gpu_id 0 --scale 4
  ```

## Acknowledgement

Our work and implementations are inspired by following projects:<br/>
[EDVR] (https://github.com/xinntao/EDVR)<br/>
[TDAN] (https://github.com/YapengTian/TDAN-VSR-CVPR-2020)<br/>
[mmediting] (https://github.com/open-mmlab/mmediting)<br/>
[DBSR] (https://github.com/goutamgmb/deep-burst-sr)<br/>
[RViDeNet] (https://github.com/cao-cong/RViDeNet)<br/>
[RawVSR] (https://github.com/proteus1991/RawVSR)<br/>
[EBSR] (https://github.com/Algolzw/EBSR)<br/>

## Citation
If you use our dataset and models in your research, please cite:
  ```
  @article{yue2022real,
  title={Real-RawVSR: Real-World Raw Video Super-Resolution with a Benchmark Dataset},
  author={Yue, Huanjing and Zhang, Zhiming and Yang, Jingyu},
  journal={arXiv preprint arXiv:2209.12475},
  year={2022}
  }
  ```
