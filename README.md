# road-lane-markings-detection
Im autonomous driving, among other things, the car needs to steer itself to keep driving in it's own lane. To accomplish this, the central problem is to detect the road-lane markings. These are the white solid or dashed lines that are drawn on each side of the lane. 

The standard modern approach to solve this type of problems is to take a large dataset of labeled examples and train a deep neural network model to accomplish the task. This is how car and pedestrian detection algorithms are developed. The difficulty with the road-lane markings is that there is no labeled dataset of them and creating such dataset would cost millions of dollars. This problem is solved here using a dataset of simulated images intermixed with a dataset of real images that contain no road.

For more information: http://machinelearning.tf.fau.de/course_labmlisp.html

Requiremnets
------

The software is developed in **Python** **2.7** using **Jupyter** **Notebook** development kit. For deep learning, the **PyTorch** framework is used.

All Python modules required for the software can be installed in two stages:
1. Create an environment and install all modules mentioned in the `spec_file.txt`.
2. Install the remaining dependencies from `requirements.txt`.
