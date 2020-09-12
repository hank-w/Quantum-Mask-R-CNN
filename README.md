# Quantum-Mask-R-CNN
Combing quantum computing using Qiskit with my Mask R-CNN model using PyTorch.


[Introduction](#intro)

![alt text](https://github.com/hank-w/Quantum-Mask-R-CNN/blob/master/images/Renders/detectron.png)

![alt text](https://github.com/hank-w/Quantum-Mask-R-CNN/blob/master/images/Renders/Abbey%20Road%20Beatles%20MASK%20R-CNN.png)


### intro
You may be asking, what is Quantum Mask R-CNN

That's a great question! 

Let's first break down its name, then explain its process and applications!



NN stands for neural network. A neural network is officially called an "Artificial neural networks". They are an artificial intelligence structure model inspired by the biological neural networks, such as how humans think. Neural networks uses a method of nodes and edges, computing a series of algorithms that recognize underlying relationships in a set of data, while adapting to changing or varying input.



CNN stands for Convolutional neural network. A CNN is commonly applied to analyzing visual imagery, such as with MASK R-CNN's use case of instance segmentation. CNNs have a linear flow between an input layer and in output layer, but with the added feature of many hidden layers inbetween that processes information

R-CNN stands for Regional-based  Convolutional neural network. It's a faster form of CNN that uses selective search, it identifies a manageable number of bounding-box object region candidates (“region of interest” or “RoI”). Then runs the CNN input layer on each RoI in parallel, thus reducing the processing time dramatically.



Mask R-CNN, is the inclusion of Masks as a form of additional data. As the bounding boxes are being generated within them, pixels within the boxes are analyzed by a weighted percentage using cross product matrix calculations, as opposed to the dot product calculations used by RCNN for bounding boxes, to return a filled in outline of the object in analysis, known as a mask.

Quantum Mask R-CNN, is my take on the neural network architecture, using Qiskit and quantum computing circuits to significantly speed up the process of both training a model and running the model on images or videos. Through parameter binding and expectation value evaluation, tensorization process can be optimized for each pixel. 

# Setup Instructions 
Overview on how to install
Step 1: create a conda virtual environment with python 3.6
Step 2: install the dependencies
Step 3: Clone the Mask_RCNN repo
Step 4: install pycocotools
Step 5: download the pre-trained weights
Step 6: Test it


Step 1 - Create a conda virtual environment
we will be using Anaconda with python 3.6.

If you don't have Anaconda, follow this tutorial

https://www.youtube.com/watch?v=T8wK5loXkXg

run this command in a CMD window
conda create -n MaskRCNN python=3.6 pip

Step 2 - Install the Dependencies
First install Qiskit from
https://qiskit.org/documentation/install.html
Then place the requirements.txt in your cwdir
https://github.com/markjay4k/Mask-RCNN-series/blob/master/requirements.txt
run these commands

actvitate MaskRCNN
pip install -r requirements.txt
NOTE: we're installing these (tf-gpu requires some pre-reqs)

numpy, scipy, cython, h5py, Pillow, scikit-image, 
tensorflow-gpu==1.5, keras, jupyter

Step 3 - Clone the Mask RCNN Repo
Run this command
git clone https://github.com/matterport/Mask_RCNN.git
Step 4 - Install pycocotools
NOTE: pycocotools requires Visual C++ 2015 Build Tools
download here if needed https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017

clone this repo

git clone https://github.com/philferriere/cocoapi.git
use pip to install pycocotools
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI


Step 5 - Download the Pre-trained Weights
Go here https://github.com/matterport/Mask_RCNN/releases
download the mask_rcnn_coco.h5 file
place the file in the Mask_RCNN directory




Step 6 - Let's Test it!
open up the demo.ipynb and run it

# Quick Setup Instructions
Quick setup will only use the basic features required and can only support images
Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in requirements.txt.

MS COCO Requirements:
To train or test on MS COCO, you'll also need:

pycocotools (installation instructions below)
MS COCO Dataset
Download the 5K minival and the 35K validation-minus-minival subsets. More details in the original Faster R-CNN implementation.
If you use Docker, the code has been verified to work on this Docker container.

Installation
Clone this repository

Install dependencies

pip3 install -r requirements.txt
Run setup from the repository root directory

python3 setup.py install
Download pre-trained COCO weights (mask_rcnn_coco.h5) from the releases page.

(Optional) To train or test on MS COCO install pycocotools from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

Linux: https://github.com/waleedka/coco
Windows: https://github.com/philferriere/cocoapi. You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

# Running my demo
Please Install Anaconda and Jupyter Notebooks:

https://docs.anaconda.com/anaconda/install/
https://jupyter.org/install

Open up anaconda command prompt:


Once inside, activate your virtual environment that you went through my setup instructions with by:
conda activate

If you don't remember the name of it, you can run:
conda env list

Then, cd to your file directory where you cloned my MASK R-CNN repo: 

e.g. cd c:\mask-rcnn



Now launch Jupyter Notebook by typing simply:

Jupyter Notebook



Click on demo.ipynb and you should see this: 




Now you can start running of the cells by clicking the "Run" button
***Note: when a cell is running, instead of displaying a number, it will display [*], please make sure you wait till it goes back to a number, such as [17], 
before you run the next cell, it takes time for first setup, but once you do it once it'll be faster

When running the cells, you may see some errors, but you can ignore them, as they are warning about depreciated dependencies, which are perfectly okay 
in our case because all my depreciated dependencies either depend on the correct legacy/archived versions of each other or are standalone with 
independent function calls, basically its like a fusion between the past and the future, sorta like time dilation lol    

When you run the final step, it will take a random picture from the "images" folder and apply instance segmentation on it, 
only images directly in the directory will be ran, the images in subfolders are just for you to test with, copy and paste whichever one you want to run



Once your at this step, you can keep running the cell over and over again, changing images inbetween! Make sure click the box each time you run 
to see the output, which will be png that you can open in a new tab and save as your desktop background!
