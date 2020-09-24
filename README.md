# TesinaCv-DL
This network is used for top-view person re-identification, it has been tested on RealShop and TVPR2 datasets.

Dependencies:

Python 3.6
Keras 2.2.4
Tensorflow 1.10
Scikit-image 0.14.2
Scikit-learn 0.19.2
Cuda 9 + CudNN

This repository contain: 
-A. Labeller engine
-B. Txt creator
-C. Preprocessing script
-D. Custom RCFusion network  https://github.com/MRLoghmani/rcfusion
-E. Run Custom RCFusion network in Colab https://colab.research.google.com/

- A. Labeller engine

1a Main2.py: can be found inside Labellatore folder, given a dataset folder, it's used to exlude bad frames and unify ids from same people, his output will be a csv file cointaining "original id", "new id","warning","name of images".
          
1b make_csv.py: can be found inside Gestore_csv folder it can be used if the the dataset is already labelled, given the path to the dataset, it generate csv file needed in "Txt creator".          
          
2. move_files.py:  can be found inside Labellatore folder, take csv file produced by Main2.py and a dataset folder, it copy frames from their original directory to a new one, frames with same "new id" will be set inside the same folder, frames ignored in the previous step will not be considered.

- B. Txt creator: 

make_txt_from_csv**.py: can be found inside Gestore_csv folder, given the path to csv generated in 1a or 1b, it produces txt files used to divide dataset in train, validation, test and gallery collections.

- C. Preprocessing script

preprocess_cv.py: Before feeding the network a preprocess phase can occurr; given a set of images, this script find the biggest subject inside each frames and than apply a black mask on the background in order to remove noise. For depth images colormap jet is applied. 

- D. Custom RCFusion network

The network need:
- resnet18_ocid_params
- txt collections generated in B
- dataset path structured in subfolders

The network is inside FinalGalleryDataAug folder, it cointains tree variations of the original RCFusion network inside different folders: 
**creare la struttura a cartelle e rinominare i file**
-train_and_eval folder contains the modified network, it can be tested running train_and_eval.py script. 
-train_and_eval_dataaug contain a variation with data augmentation applied, it can be tested running train_and_eval_dataaug.py script.
-train_and_eval_gallery cointain a variation with data augmentation and gallery, it can be tested running train_and_eval_gallery.py script.

Before starting the network some paramethers have to be specified:

- If you are using .rgb frames (such as RealShop dataset), you have to be sure to import image_data_handler_joint_multimodal_png* version
- If you are using .jpg frames (such as TVPR2 dataset), you have to be sure to import image_data_handler_joint_multimodal_jpg* version


**-------------------Data-related params-------------------**
**---------------------------------------------------------**

or  dataset_root_dir = "path/to/RealShopdataset/"

**TVPR2 Path settings:**

main folder of TVPR2 dataset must contain 
-test folder: where are stored test frames grouped in subfolders by person ID, and a folder containing test.txt produced in B
-train folder: where are stored gallery,train,and validation frames grouped in subfolders by person ID, and a folder containing gallery.txt,train.txt,and validation.txt produced in B (gallery if needed).


params_root_dir = "path/to/resnet18_ocid_params"
dataset_root_dir = "path/to/TVPR2dataset"    

dataset_train_dir_rgb = dataset_root_dir + '/train/'
dataset_val_dir_rgb = dataset_root_dir + '/train/'
dataset_test_dir = dataset_root_dir + '/test/'

- specify resnet18 paramethers names
params_dir_rgb = params_root_dir + '/resnet18_ocid_rgb++_params.npy'
params_dir_depth = params_root_dir + '/resnet18_ocid_surfnorm++_params.npy'

pathToSavedModel = "path/to/saved/model"

- Checkpoint dir
checkpoint_dir = "/specify/checkpoint/dir/"
- log dir
simpleLog = "path/to/log/dir/Log_Name.txt"

- specify names of folders containing txt files 
train_file = dataset_train_dir_rgb + 'cartella/contenente/train.txt'
val_file = dataset_val_dir_rgb + 'cartella/contenente/val.txt'
test_file = dataset_test_dir + 'cartella/contenente/test.txt'

gallery_file = dataset_train_dir_rgb + 'cartella/contenente/gallery.txt'


**RealShop Path settings:**

main folder of RealShop dataset must contain 
A series of subfolders grouped by person ID, and a folder containing test.txt gallery.txt, train.txt, and validation.txt produced in B (gallery if needed).

params_root_dir = "path/to/resnet18_ocid_params"
dataset_root_dir = "path/to/RealShopdataset/"    

dataset_train_dir_rgb = dataset_root_dir 
dataset_val_dir_rgb = dataset_root_dir 
dataset_test_dir = dataset_root_dir

- specify resnet18 paramethers names
params_dir_rgb = params_root_dir + '/resnet18_ocid_rgb++_params.npy'
params_dir_depth = params_root_dir + '/resnet18_ocid_surfnorm++_params.npy'

pathToSavedModel = "path/to/saved/model"

- Checkpoint dir
checkpoint_dir = "/specify/checkpoint/dir/"
- log dir
simpleLog = "path/to/log/dir/Log_Name.txt"

- specify names of folders containing txt files 
train_file = dataset_train_dir_rgb + 'cartella/contenente/train.txt'
val_file = dataset_val_dir_rgb + 'cartella/contenente/val.txt'
test_file = dataset_test_dir + 'cartella/contenente/test.txt'

gallery_file = dataset_train_dir_rgb + 'cartella/contenente/gallery.txt'
**---------------------------------------------------------**

- E. Run Custom RCFusion network in Colab

Colab.ipynb: this script contain instructions to run the network in google colab.









