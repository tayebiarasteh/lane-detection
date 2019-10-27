# Semi-supervised road-lane markings detection for autonomous driving

This project was part of the *Labopraktikum Machine Learning in Signal Processing* at [University of Erlangen-Nuremberg (FAU)](https://www.fau.eu/). For more information click [here](http://machinelearning.tf.fau.de/).

Introduction
------
In autonomous driving, among other things, the car needs to steer itself to keep driving in its own lane. To accomplish this, the central problem is to detect the road-lane markings. These are the white solid or dashed lines that are drawn on each side of the lane. 

The standard modern approach to solve this type of problems is to take a large dataset of labeled examples and train a deep neural network model to accomplish the task. This is how car and pedestrian detection algorithms are developed. The difficulty with the road-lane markings is that there is no labeled dataset of them and creating such dataset would cost millions of dollars. 
#### Goal of this project: 
To solve the afformentioned problem, using a dataset of simulated images intermixed with a dataset of real images that contain no road.

Proposed architecture
------
The architecture used in this software is a variant of the [U-Net](https://arxiv.org/pdf/1505.04597.pdf) architecture.

You can see the input video [here](https://github.com/SorooshTA/lane-detection/raw/master/data/input_data/Video/3911-3931.mp4) and the output video [here](https://github.com/SorooshTA/lane-detection/raw/master/data/output_data/outputvideo.mp4).

Software requirements
------

The software is developed in **Python** **2.7** using **Jupyter** **Notebook** development kit. For deep learning, the **PyTorch** framework is used.

All Python modules required for the software can be installed from `reuirements` in two stages:
1. Create an environment and install all modules mentioned in the `spec_file.txt`.
2. Install the remaining dependencies from `requirements.txt`.
