from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import copy
import tensorflow as tf

import matplotlib.image as mpimg
import numpy as np
import time
from physics_engine import gen, make_video
from vin import VE, DP

FLAGS = None

def variable_summaries(var,idx):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries_'+str(idx)):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def train():
  # Architecture Definition
  height=FLAGS.height;weight=FLAGS.weight;col_dim=FLAGS.col_dim;
  F1=tf.placeholder(tf.float32, [None,height,weight,col_dim], name="F1");
  F2=tf.placeholder(tf.float32, [None,height,weight,col_dim], name="F2");
  F3=tf.placeholder(tf.float32, [None,height,weight,col_dim], name="F3");
  F4=tf.placeholder(tf.float32, [None,height,weight,col_dim], name="F4");
  F5=tf.placeholder(tf.float32, [None,height,weight,col_dim], name="F5");
  F6=tf.placeholder(tf.float32, [None,height,weight,col_dim], name="F6");
  S1,S2,S3,S4=VE(F1,F2,F3,F4,F5,F6,FLAGS);
  out_dp=DP(S1,S2,S3,S4,FLAGS);
  #y=np.array(range(FLAGS.height),dtype=float)/(FLAGS.height-1);
  #x=np.array(range(FLAGS.weight),dtype=float)/(FLAGS.weight-1);
  #nx, ny = (FLAGS.weight, FLAGS.height);
  #x = np.linspace(0, 1, nx);
  #y = np.linspace(0, 1, ny);
  #xv, yv = np.meshgrid(x, y);
  #tf_xv=tf.Variable(xv);
  #tf_yv=tf.Variable(yv);

def main(_):
  FLAGS.log_dir+=str(int(time.time()));
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', type=str, default='/tmp/interaction-network/',
                      help='Summaries log directry')
  parser.add_argument('--Ds', type=int, default=5,
                      help='The State Dimention')
  parser.add_argument('--No', type=int, default=3,
                      help='The Number of Objects')
  parser.add_argument('--Nr', type=int, default=30,
                      help='The Number of Relations')
  parser.add_argument('--Dr', type=int, default=1,
                      help='The Relationship Dimension')
  parser.add_argument('--Dx', type=int, default=1,
                      help='The External Effect Dimension')
  parser.add_argument('--De', type=int, default=50,
                      help='The Effect Dimension')
  parser.add_argument('--Dp', type=int, default=2,
                      help='The Object Modeling Output Dimension')
  parser.add_argument('--Da', type=int, default=1,
                      help='The Abstract Modeling Output Dimension')
  parser.add_argument('--batch_num', type=int, default=100,
                      help='The number of data on each mini batch')
  parser.add_argument('--height', type=int, default=32,
                      help='Height of the image')
  parser.add_argument('--weight', type=int, default=32,
                      help='Weight of the image')
  parser.add_argument('--col_dim', type=int, default=4,
                      help='The color dimensional size')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
