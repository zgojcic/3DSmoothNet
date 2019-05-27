# 3DSmoothNet repository
This repository provides code and data to train and evaluate the 3DSmoothNet, compact local feature descriptor for undstructured point clouds. It represents the official implementation of the paper:

### The Perfect Match: 3D Point Cloud Matching with Smoothed Densities (CVPR 2019).
[PDF](https://arxiv.org/pdf/1811.06879.pdf) | [Group Page](http://www.gseg.igp.ethz.ch/)

*[Zan Gojcic](https://www.ethz.ch/content/specialinterest/baug/institute-igp/geosensors-and-engineering-geodesy/en/people/scientific-assistance/zan-gojcic.html), [Caifa Zhou](https://www.ethz.ch/content/specialinterest/baug/institute-igp/geosensors-and-engineering-geodesy/en/people/scientific-assistance/caifa-zhou.html), [Jan D. Wegner](http://www.prs.igp.ethz.ch/content/specialinterest/baug/institute-igp/photogrammetry-and-remote-sensing/en/group/people/person-detail.html?persid=186562), [Andreas Wieser](https://www.ethz.ch/content/specialinterest/baug/institute-igp/geosensors-and-engineering-geodesy/en/people/group-head/prof-dr--andreas-wieser.html)*

We propose 3DSmoothNet, a full workflow to match
3D point clouds with a siamese deep learning architecture
and fully convolutional layers using a voxelized smoothed
density value (SDV) representation. The latter is computed per interest point and aligned to the local reference frame (LRF) to achieve rotation invariance. Our compact, learned, rotation invariant 3D point cloud descriptor achieves 94.9% average recall on the 3DMatch benchmark data set, outperforming the state-of-the-art by more than 20 percent points with only 32 output dimensions. This very low output dimension allows for near realtime correspondence search with 0.1 ms per feature point on a standard PC. Our approach is sensor- and sceneagnostic because of SDV, LRF and learning highly descriptive features with fully convolutional layers. We show that 3DSmoothNet trained only on RGB-D indoor scenes
of buildings achieves 79.0% average recall on laser scans
of outdoor vegetation, more than double the performance of our closest, learning-based competitors.

![3DSMoothNet](figures/Network.jpg?raw=true)

### Citation

If you find this code useful for your work or use it in your project, please consider citing:

```shell
@inproceedings{gojcic20193DSmoothNet, 
	title={The Perfect Match: 3D Point Cloud Matching with Smoothed Densities}, 
	author={Gojcic, Zan and Zhou, Caifa and Wegner, Jan Dirk and Wieser Andreas}, 
	booktitle={International conference on computer vision and pattern recognition (CVPR)}, 
	year={2019} 
}
```

### Contact
If you have any questions or find any bugs, please let us know: Zan Gojcic, Caifa Zhou {firstname.lastname@geod.baug.ethz.ch}

## Instructions
### Dependencies
The pipeline of 3DSmoothNet consits of two steps: 

1. `main.cpp`: computes the smoothed density value (SDV) voxel grid for a point cloud provided in the .ply format.

2. `main_cnn.py`: Inferes the feature descriptors from the SDV voxel grid using the pretrained model. Can also be used to train 3DSmoothNet from scratch or for fine-tuning

Computation of the SDV voxel-grid impelemented in C++ has dependencies on the *[Point Cloud Library (PCL)](http://www.pointclouds.org/)* and OpenMP. For convinience we include a shell script, which can be used to install PCL as
```shell
./install_pcl.sh
```

The CNN part of the pipeline is based on Python3 (specifically 3.5) and is implemented in Tensorflow. The required libraries can be easily installed by runing
```shell
pip install -r requirements.txt
```
in a new virtual environment.

### Input parametrization

We provide a cmake file that can be used to compile `main.cpp` as:
```shell
cmake -DCMAKE_BUILD_TYPE=Release .
make
```
which will create an executable `3DSmoothNet` that takes the following command line arguments:

```
-f Path to the Input point cloud file in .ply format
-r Half size of the voxel grid in the unit of the point cloud. Defaults to 0.15.
-n Number of voxels in a side of the grid. Whole grid is nxnxn. Defaults to 16.
-h Width of the Gaussia kernel used for smoothing. Defaults to 1.75.
-k Path to the file with the indices of the interest points. Defults to 0 (i.e. all points are considered).
-o Output folder path. Defaults to "./data/sdv_voxel_grid/".
```

### Testing
The testing with the pretrained models `./models/` for 16, 32 and 64 dimensional 3DSmoothNet can be easily done by runing
```
python ./main_cnn.py --run_mode=test
```
which will infer the 3DSmoothNet descriptors for all SDV voxel-grid files (*.csv) located in `./data/test/input_data/`. For more options in runing the inference please see `./core/config.py`

### Training
The provided code can also be used to train 3DSmoothNet from scratch using e.g.:
```
python ./main_cnn.py --run_mode=train --output_dim=32 --batch_size=256
```
to train a 32 dimensional 3DSmoothNet with mini-batch size 256. By defult, the training data saved in `data\train\trainingData3DMatch\` wil be used and the tensorboard log will be saved in `./logs/`. For more training options please see `./core/config.py`. 

### Evaluation
The source-code for the performance evaluation on the 3DMatch data set is available in the `./evaluation/`.

In order to compute the recall, first run the `correspondenceMatching.m` and then the `evaluate3DMatchDataset.m`.

With small changes of the point cloud names and paths, the code can also be used to evaluate the performance on the ETH data set.

## Demo

We prepared a small demo which demonstrates the whole pipeline using two fragments from the 3DMatch dataset. To carry out the demo, please run 
```
python ./demo.py
```
after installing and compiling the necessary source code. It will compute the SDV voxel-grid for the input point clouds, before infering 32 dimensional 3DSmoothNet descriptors using the pretrained model. These descriptors are then used to estimate the rigid-body transformation parameters using RANSAC. Software outputs the results of RANSAC as well as two figures, first showing the inital state and the second the state after the 3DSmoothNet registration

![3DSMoothNet](figures/demo.png?raw=true)

## Data
### Training data
Training data created using the RGB-D data from 3DMatch data set can be downloaded from [here (145GB)](https://share.phys.ethz.ch/~gsg/3DSmoothNet/training_data/trainingData.rar).
It consists of a `*.tfreford` file for each scene, due to the size of the data several scenes are split into more `*.tfreford` files (329 files all together). In order to train the model using this data replace the `sample_training_file.tfrecord` file in `./data/train/trainingData3DMatch/` with the files from this archive. When run in train mode the source code will automatically read all the files from the selected folder.  
 
If you use these data please consider also citing the authors of the data set [3DMatch](http://3dmatch.cs.princeton.edu/).

### Evaluation data sets
##### 3DMatch
The pointclouds and indices of the interest points for the *3DMatch* data set can be downloaded from [here (0.27GB)](https://share.phys.ethz.ch/~gsg/3DSmoothNet/data/3DMatch.rar)

If you use these data please consider also citing the authors of the data set [3DMatch](http://3dmatch.cs.princeton.edu/).

##### 3DSparseMatch
The pointclouds and indices of the interest points for the *3DSparseMatch* data set can be downloaded from [here (0.83GB)](https://share.phys.ethz.ch/~gsg/3DSmoothNet/data/3DSparseMatch.rar)

If you use these data please consider also citing the authors of the original data set [3DMatch](http://3dmatch.cs.princeton.edu/).

##### ETH
The pointclouds and indices of the interest points for the *ETH* data set can be downloaded from [here (0.11GB)](https://share.phys.ethz.ch/~gsg/3DSmoothNet/data/ETH.rar)

If you use these data please consider also citing the authors of the original data set [ETH](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration).


### Pretrained model

The pretrained model of the 3DSmoothNet with 128 dim can be downloaded from [here (0.10GB)](https://share.phys.ethz.ch/~gsg/3DSmoothNet/models/128_dim/3DSmoothNet_model_128_dim.rar)

To use this model please unpack the archive to `./models/128_dim/`. 


## TO DO!!
- Add source code of the descriptors used as baseline


### License
This code is released under the Simplified BSD License (refer to the LICENSE file for details).
