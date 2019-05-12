# OCT Segmentation Guide

### Zhongyang Sun  szy019@gmail.com



## Environment

### Hardware

CPU AMD Threadripper 1900X

RAM 32GB（8GB * 4）

GPU NVIDIA GeForce 1080TI & NVIDIA GeForce 1060

### System

Linux Ubuntu 18.04

Python 3.6.6

Caffe 1.0.0

Cuda 9.0.176

opencv 3.4.1

cuDNN 7.0.5

NVIDIA Driver 390.116

## Directory Structure

```
fcn
 |
data - oct - Labels - *.mat
 |              |
 |           originalImages - *.jpg
 |              |
 |           test.txt
 |              |
 |           trainval.txt
 | 
FCN_16s - models - OCT_Segmentation.caffemodel
            |         |
            |      siftflow-fcn16s-heavy.caffemodel
            |       
          results - *.jpg
            |
          infer.py
            |
          infer_collection.py
            |
          oct_layers.py
            |
          score.py
            |
          solve.py
            |
          surgery.py
            |
          solver.prototxt
            |
          train.prototxt
            |
          test.prototxt

```



## Files



| data/oct       |                                                              |
| :------------- | :----------------------------------------------------------- |
| Labels         | Store label of each pixel in mat format for all images in training set |
| originalImages | Store data of training and test images in jpg format         |
| test.txt       | Includes all image names of images from test set             |
| train.txt      | Includes all image names of images from training set         |



| Fcn_16s             |                                                              |
| ------------------- | ------------------------------------------------------------ |
| models              | Store trained models                                         |
| results             | Store the Segmentated OCT image in jpg format                |
| infer.py            | Used to segmentation single OCT image  to estimate the performance of model |
| Infer_collection.py | Used to segmentation all images in tests, and store the segmentated OCT images in results folder |
| oct_layers.py       | Used to replace caffe's default input layer                  |
| solve.py            | Including the specific procedure of the training process, such as source model of  migration training, or the number of training iterations, etc. |
| solver.prototxt     | Includes mainly parameters in the training procedure, such as training rate, batch size, etc. |
| train.prototxt      | Define network structure for training                        |
| test.prototxt       | Define network structure for testing                         |

## Instructions

### Training

1.Copy images in training set to ```originalImages``` folder

2.Generate label files for every image in training set, and store in ```Labels``` folder

3.Write name of images in trainset set in ```test.txt```

4.Modify ```slove.py```, ```solver.protxt```, identify training processes and parameters

5.Execute ```python3 solve.py``` ，start training and store trained model in ```moldels``` folder

### Testing

1.Copy images in test set to ```originalImage``` folder

2.Write name of images in test set in ```train.txt```

3.Execute ```python3 infer_collection.py``` ，segmentation images in test set, and store the segmentation result in ```results``` folder



## THOCT1800 Dataset

This dataset consists of 1800 preprocessed retinal SD-OCT B-scans (600 AMD, 600 DME, and 600 NOR), all images are intended for use in research and education situations, and every use of this dataset should include citation in their corresponding papers.
