Visual Interaction Networks
====================

Tensorflow Implementation of Visual Interaction Networks from Google Deepmind.

Implementation is on Tensorflow r1.2.

https://arxiv.org/abs/1706.01433

"Another key part of relational reasoning involves predicting the future in a physical scene. From just a glance, humans can infer not only what objects are where, but also what will happen to them over the upcoming seconds, minutes and even longer in some cases. For example, if you kick a football against a wall, your brain predicts what will happen when the ball hits the wall and how their movements will be affected afterwards (the ball will ricochet at a speed proportional to the kick and - in most cases - the wall will remain where it is)." From an article of Deepmind

https://deepmind.com/blog/neural-approach-relational-reasoning/

![alt tag](https://github.com/jaesik817/visual-interaction-networks_tensorflow/blob/master/figures/vin_fig1.png)

N-objects Gravity Simulations
--------------------------

For changing configure values, please check constants script.

`
cat constracts.py
`

For generating images and data,

`
python physical_engines.py
`

For modeling Visual Interaction Networks

`
python gravity_vin.py
`
### Data
The Data are gathered from my own implemented physics engine, which is same of interaction network git repo.
https://github.com/jaesik817/Interaction-networks_tensorflow

One different thing from IN physics engines is each frame is gathered per "frame_step", because 0.0001 secs frame cannot be recognized in 32 x 32 images.

### Settings
Settings are written in constants.py and gravity_vin.py. 
The number of objects, each simulations, rollout frames and simulation sets are 3, 50, 100 and 10. 
The training max epoches are 1000.
In physical_engines code, every frames are saved as image and coded data, and those things are used in gravity_vin script.
Each image has background ones from CIFAR 10 training data set as the paper.

### Results
The video frame is also 32x32, thus that cannot be easily recognized, however based on a big object, orbit system working is checked in the video.

True :
![alt tag](https://github.com/jaesik817/visual-interaction-networks_tensorflow/blob/master/figures/true.gif)

Modeling :
![alt tag](https://github.com/jaesik817/visual-interaction-networks_tensorflow/blob/master/figures/modeling.gif)
