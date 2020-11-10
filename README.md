# Top-View-Re-Identificaton

## Introduction:

This network is used for top-view person re-identification, it has been tested on RealShop and TVPR2 datasets. The network is a custom version of RCFusion. [link](https://github.com/MRLoghmani/rcfusion)

## Requirements:

1. *Python 3.6*
2. *Keras 2.2.4*
3. *Tensorflow 1.10*
4. *Scikit-image 0.14.2*
5. *Scikit-learn 0.19.2*
6. *Cuda 9 + CudNN*

## This repository contains:
 
1. Labeller
2. csv and txt creator and manager
3. csv and txt for tests
4. Preprocessing script
5. Custom RCFusion network
6. Open World

And other utility files that helped us to get the job done.

Now a brief description of the most important files.

## 1. Labeller

1. *main.py*: given a dataset folder, it's used to exclude bad frames and unify ids of the same person, his output will be a csv file cointaining "original id", "new id", "warning", "images"
2. *move_files.py*: it takes the csv file produced by main.py and the dataset folder, it copies the frames from their original directory to a new one, frames with same "new id" will be placed inside the same folder, frames ignored in the previous step will not be considered

## 2. csv and txt creator and manager:

1. *make_csv.py*: it can be used if the the dataset is already labelled, given the dataset path, it generates a csv file that is needed to use the make_txt_from_csv.py files
3. *make_txt_from_csv.py*: given the path to the generated csv, those kind of files produce the txt files used to divide the dataset in train, validation, test and gallery sets

## 3. csv and txt for tests

Those files can be found in the csv and txt folders. They are the csv and txt we produced and used, and can be used for tests.

## 4. Preprocessing script

*preprocess_cv.py*: before feeding the network a preprocess phase can is needed: given a set of images, this script find the biggest subject inside each frames and then apply a black mask on the background in order to remove noise. For depth images colormap jet is applied. 

## 5. Custom RCFusion network

The network needs:
1. Path to ResNet-18 pre-trained weights for initialization [link](https://data.acin.tuwien.ac.at/index.php/s/RueHQUbs2JtoHeJ)
2. Path to txt files generated with csv and txt crerator and manager
3. Path to dataset

There are three variations of the original RCFusion network: 
1. RcFusion folder contains the modified network, it can be tested running train_and_eval_and_test.py
2. RcFusionDataAug folder contains a variation with data augmentation applied, it can be tested running train_and_eval_and_test.py
3. RcFusionGalleryDataAug folder cointains a variation with data augmentation and gallery, it can be tested running train_and_eval_and_test.py

To run the network in colab run Colab.ipynb: this file contains the instructions to run the network in google colab.

Before starting the network some paramethers have to be specified:
- If you are using .png frames (such as RealShop dataset), you have to be sure to import ImageDataHandler from image_data_handler_joint_multimodal_png
- If you are using .jpg frames (such as TVPR2 dataset), you have to be sure to import ImageDataHandler from image_data_handler_joint_multimodal_jpg

## 6. Open World
In the folder Create txt openworld text there are some script, to use when train, validation and gallery are alreary created, to create txt test files with a selecatble number of intruders get randomically by the dataset csv.
In the same folder there are also scripts to create test files with intruders from zero, without having train, validation and gallery alreary created.
There are 2 script to plot TTR/FTR graphic and plot histogram comparsion between gallery and test set with intruders.
In particular there is utils.py which contains all the functions used to generate adaptive and standard thresholds and only_gallery_and_test.py to make open world test with intruders and compute ttr and ftr.


## RealShop path settings:

dataset_root_dir = '/path/to/RealShop/'

dataset_train_dir_rgb = dataset_root_dir
dataset_val_dir_rgb = dataset_root_dir
dataset_test_dir = dataset_root_dir

train_file = dataset_train_dir_rgb + '/path/to/train.txt'
val_file = dataset_val_dir_rgb + '/path/to/val.txt'
test_file = dataset_test_dir + '/path/to/test.txt'
gallery_file = dataset_train_dir_rgb + '/path/to/gallery.txt'

## TVPR2 path settings:

dataset_root_dir = '/path/to/TVPR2'

dataset_train_dir_rgb = dataset_root_dir + '/train/'
dataset_val_dir_rgb = dataset_root_dir + '/train/'
dataset_test_dir = dataset_root_dir + '/test/'

train_file = dataset_train_dir_rgb + '/path/to/train.txt'
val_file = dataset_val_dir_rgb + '/path/to/val.txt'
test_file = dataset_test_dir + '/path/to/test.txt'
gallery_file = dataset_train_dir_rgb + '/path/to/gallery.txt'

## Specify the following paths:

params_root_dir = '/path/to/resnet18_ocid_params'

params_dir_rgb = params_root_dir + '/resnet18_ocid_rgb++_params.npy'

params_dir_depth = params_root_dir + '/resnet18_ocid_surfnorm++_params.npy'

pathToSavedModel = '/path/to/model.ckpt'

checkpoint_dir = '/tmp/my_caffenet/'

simpleLog = 'path/to/Log.txt'
