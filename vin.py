from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import copy
import tensorflow as tf
from sklearn.cluster import KMeans

import matplotlib.image as mpimg
import numpy as np
import time
from physics_engine import gen, make_video

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

def m(O,Rr,Rs,Ra):
  return tf.concat([(tf.matmul(O,Rr)-tf.matmul(O,Rs)),Ra],1);
  #return tf.concat([tf.matmul(O,Rr),tf.matmul(O,Rs),Ra],1);

def phi_R(B):
  h_size=150;
  B_trans=tf.transpose(B,[0,2,1]);
  B_trans=tf.reshape(B_trans,[-1,(FLAGS.Ds+FLAGS.Dr)]);
  w1 = tf.Variable(tf.truncated_normal([(FLAGS.Ds+FLAGS.Dr), h_size], stddev=0.1), name="r_w1", dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([h_size]), name="r_b1", dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1);
  w2 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w2", dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([h_size]), name="r_b2", dtype=tf.float32);
  h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
  w3 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w3", dtype=tf.float32);
  b3 = tf.Variable(tf.zeros([h_size]), name="r_b3", dtype=tf.float32);
  h3 = tf.nn.relu(tf.matmul(h2, w3) + b3);
  w4 = tf.Variable(tf.truncated_normal([h_size, h_size], stddev=0.1), name="r_w4", dtype=tf.float32);
  b4 = tf.Variable(tf.zeros([h_size]), name="r_b4", dtype=tf.float32);
  h4 = tf.nn.relu(tf.matmul(h3, w4) + b4);
  w5 = tf.Variable(tf.truncated_normal([h_size, FLAGS.De], stddev=0.1), name="r_w5", dtype=tf.float32);
  b5 = tf.Variable(tf.zeros([FLAGS.De]), name="r_b5", dtype=tf.float32);
  h5 = tf.matmul(h4, w5) + b5;
  h5_trans=tf.reshape(h5,[-1,FLAGS.Nr,FLAGS.De]);
  h5_trans=tf.transpose(h5_trans,[0,2,1]);
  return(h5_trans);

def a(O,Rr,X,E):
  E_bar=tf.matmul(E,tf.transpose(Rr,[0,2,1]));
  O_2=tf.stack(tf.unstack(O,FLAGS.Ds,1)[3:5],1);
  return (tf.concat([O_2,X,E_bar],1));
  #return (tf.concat([O,X,E_bar],1));

def phi_O(C):
  h_size=100;
  C_trans=tf.transpose(C,[0,2,1]);
  C_trans=tf.reshape(C_trans,[-1,(2+FLAGS.Dx+FLAGS.De)]);
  w1 = tf.Variable(tf.truncated_normal([(2+FLAGS.Dx+FLAGS.De), h_size], stddev=0.1), name="o_w1", dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([h_size]), name="o_b1", dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1);
  w2 = tf.Variable(tf.truncated_normal([h_size, FLAGS.Dp], stddev=0.1), name="o_w2", dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([FLAGS.Dp]), name="o_b2", dtype=tf.float32);
  h2 = tf.matmul(h1, w2) + b2;
  h2_trans=tf.reshape(h2,[-1,FLAGS.No,FLAGS.Dp]);
  h2_trans=tf.transpose(h2_trans,[0,2,1]);
  return(h2_trans);

def phi_A(P):
  h_size=25;
  p_bar=tf.reduce_sum(P,2);
  w1 = tf.Variable(tf.truncated_normal([FLAGS.Dp, h_size], stddev=0.1), name="a_w1", dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([h_size]), name="a_b1", dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(p_bar, w1) + b1);
  w2 = tf.Variable(tf.truncated_normal([h_size, FLAGS.Da], stddev=0.1), name="a_w2", dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([FLAGS.Da]), name="a_b2", dtype=tf.float32);
  h2 = tf.matmul(h1, w2) + b2;
  return(h2);

def conv_variable(weight_shape):
  w = weight_shape[0]
  h = weight_shape[1]
  input_channels  = weight_shape[2]
  output_channels = weight_shape[3]
  d = 1.0 / np.sqrt(input_channels * w * h)
  bias_shape = [output_channels]
  weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
  bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
  return weight, bias

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def maxpool2d(x, k=2):
  return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
def IPE(F1F2,F2F3):
  img_pair=tf.concat([F1F2,F2F3],0);
  # First 2 layer conv (kernel size 10 and 4 channels)
  w_1_1,b_1_1=conv_variable([10,10,8,4]);
  w_1_2,b_1_2=conv_variable([10,10,4,4]);
  h_1_1=tf.nn.relu(conv2d(img_pair,w_1_1,1)+b_1_1);
  h_1_2=tf.nn.relu(conv2d(h_1_1,w_1_2,1)+b_1_2);
  # Second 2 layer conv (kernel size 3 and 16 channels)
  w_2_1,b_2_1=conv_variable([3,3,8,16]);
  w_2_2,b_2_2=conv_variable([3,3,16,16]);
  h_2_1=tf.nn.relu(conv2d(img_pair,w_2_1,1)+b_2_1);
  h_2_2=tf.nn.relu(conv2d(h_2_1,w_2_2,1)+b_2_2);
  en_pair=tf.concat([h_1_2,h_2_2],3);
  # Third 2 layer conv (kernel size 3 and 16 channels)
  w_3_1,b_3_1=conv_variable([3,3,20,16]);
  w_3_2,b_3_2=conv_variable([3,3,16,16]);
  h_3_1=tf.nn.relu(conv2d(en_pair,w_3_1,1)+b_3_1);
  h_3_2=tf.nn.relu(conv2d(h_3_1,w_3_2,1)+b_3_2);
  # Inject x and y coordinate channels
  x=tf.placeholder(tf.float32, [None,FLAGS.height,FLAGS.weight,1], name="x-cor");
  y=tf.placeholder(tf.float32, [None,FLAGS.height,FLAGS.weight,1], name="y-cor");
  h_3_2_x_y=tf.concat([h_3_2,x,y],3);
  # Fourth conv and max-pooling layers to unit height and width
  w_4_1,b_4_1=conv_variable([3,3,18,16]);
  h_4_1=tf.nn.relu(conv2d(h_3_2_x_y,w_4_1,1)+b_4_1);
  h_4_1=maxpool2d(h_4_1);
  w_4_2,b_4_2=conv_variable([3,3,16,16]);
  h_4_2=tf.nn.relu(conv2d(h_4_1,w_4_2,1)+b_4_2);
  h_4_2=maxpool2d(h_4_2);
  w_4_3,b_4_3=conv_variable([3,3,16,16]);
  h_4_3=tf.nn.relu(conv2d(h_4_2,w_4_3,1)+b_4_3);
  h_4_3=maxpool2d(h_4_3);
  w_4_4,b_4_4=conv_variable([3,3,16,32]);
  h_4_4=tf.nn.relu(conv2d(h_4_3,w_4_4,1)+b_4_4);
  h_4_4=maxpool2d(h_4_4);
  w_4_5,b_4_5=conv_variable([3,3,32,32]);
  h_4_5=tf.nn.relu(conv2d(h_4_4,w_4_5,1)+b_4_5);
  h_4_5=maxpool2d(h_4_5)
  res_pair=tf.reshape(h_4_5,[-1,32]);
  S1=tf.slice(res_pair,[0,0],[FLAGS.batch_num,-1]);
  S2=tf.slice(res_pair,[FLAGS.batch_num,0],[-1,-1]);
  return S1,S2;
  
  #y=np.array(range(FLAGS.height),dtype=float)/(FLAGS.height-1);
  #x=np.array(range(FLAGS.weight),dtype=float)/(FLAGS.weight-1);
  #nx, ny = (FLAGS.weight, FLAGS.height);
  #x = np.linspace(0, 1, nx);
  #y = np.linspace(0, 1, ny);
  #xv, yv = np.meshgrid(x, y);
  #tf_xv=tf.Variable(xv);
  #tf_yv=tf.Variable(yv);
  

def VE(F1,F2,F3):
  F1F2=tf.concat([F1,F2],3);
  F2F3=tf.concat([F2,F3],3);
  S1,S2=IPE(F1F2,F2F3);
  S1S2=tf.concat([S1,S2],0);
  # shared a linear layer
  w0 = tf.Variable(tf.truncated_normal([32, FLAGS.No*64], stddev=0.1), dtype=tf.float32)
  b0 = tf.Variable(tf.zeros([FLAGS.No*64]), dtype=tf.float32)
  h0 = tf.matmul(S1S2, w0) + b0
  S1=tf.reshape(tf.slice(h0,[0,0],[FLAGS.batch_num,-1]),[-1,FLAGS.No,64]);
  S2=tf.reshape(tf.slice(h0,[FLAGS.batch_num,0],[-1,-1]),[-1,FLAGS.No,64]);
  S=tf.concat([S1,S2],2);
  # shared MLP
  S=tf.reshape(S,[-1,128]);
  w1 = tf.Variable(tf.truncated_normal([128, 64], stddev=0.1), dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([64]), dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(S, w1) + b1);
  w2 = tf.Variable(tf.truncated_normal([64, 64], stddev=0.1), dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([64]), dtype=tf.float32);
  h2 = tf.matmul(h1, w2) + b2;
  h2 = tf.reshape(h2,[-1,FLAGS.No,64]);
  return h2;

 
def train():

  height=FLAGS.height;weight=FLAGS.weight;col_dim=FLAGS.col_dim;
  F1=tf.placeholder(tf.float32, [None,height,weight,col_dim], name="F1");
  F2=tf.placeholder(tf.float32, [None,height,weight,col_dim], name="F2");
  F3=tf.placeholder(tf.float32, [None,height,weight,col_dim], name="F3");
  VE(F1,F2,F3);

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
