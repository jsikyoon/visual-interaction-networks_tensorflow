from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import copy
import tensorflow as tf

import numpy as np
import time

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

def IPE(F1F2,F2F3,F3F4,F4F5,F5F6,x_cor,y_cor,FLAGS):
  img_pair=tf.concat([F1F2,F2F3,F3F4,F4F5,F5F6],0);
  # First 2 layer conv (kernel size 10 and 4 channels)
  w_1_1,b_1_1=conv_variable([10,10,FLAGS.col_dim*2,4]);
  h_1_1=tf.nn.relu(conv2d(img_pair,w_1_1,1)+b_1_1);
  w_1_2,b_1_2=conv_variable([10,10,4,4]);
  h_1_2=tf.nn.relu(conv2d(h_1_1,w_1_2,1)+b_1_2+h_1_1);
  # Second 2 layer conv (kernel size 3 and 16 channels)
  w_2_1,b_2_1=conv_variable([3,3,FLAGS.col_dim*2,16]);
  h_2_1=tf.nn.relu(conv2d(img_pair,w_2_1,1)+b_2_1);
  w_2_2,b_2_2=conv_variable([3,3,16,16]);
  h_2_2=tf.nn.relu(conv2d(h_2_1,w_2_2,1)+b_2_2+h_2_1);
  en_pair=tf.concat([h_1_2,h_2_2],3);
  # Third 2 layer conv (kernel size 3 and 16 channels)
  w_3_1,b_3_1=conv_variable([3,3,20,16]);
  h_3_1=tf.nn.relu(conv2d(en_pair,w_3_1,1)+b_3_1);
  w_3_2,b_3_2=conv_variable([3,3,16,16]);
  h_3_2=tf.nn.relu(conv2d(h_3_1,w_3_2,1)+b_3_2+h_3_1);
  # Inject x and y coordinate channels
  h_3_2_x_y=tf.concat([h_3_2,x_cor,y_cor],3);
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
  pair1=tf.slice(res_pair,[0,0],[FLAGS.batch_num,-1]);
  pair2=tf.slice(res_pair,[FLAGS.batch_num,0],[FLAGS.batch_num,-1]);
  pair3=tf.slice(res_pair,[FLAGS.batch_num*2,0],[FLAGS.batch_num,-1]);
  pair4=tf.slice(res_pair,[FLAGS.batch_num*3,0],[FLAGS.batch_num,-1]);
  pair5=tf.slice(res_pair,[FLAGS.batch_num*4,0],[FLAGS.batch_num,-1]);
  return pair1,pair2,pair3,pair4,pair5;

def VE(F1,F2,F3,F4,F5,F6,x_cor,y_cor,FLAGS):
  F1F2=tf.concat([F1,F2],3);
  F2F3=tf.concat([F2,F3],3);
  F3F4=tf.concat([F3,F4],3);
  F4F5=tf.concat([F4,F5],3);
  F5F6=tf.concat([F5,F6],3);
  pair1,pair2,pair3,pair4,pair5=IPE(F1F2,F2F3,F3F4,F4F5,F5F6,x_cor,y_cor,FLAGS);
  concated_pair=tf.concat([pair1,pair2,pair3,pair4,pair5],0);
  # shared a linear layer
  w0 = tf.Variable(tf.truncated_normal([32, FLAGS.No*64], stddev=0.1), dtype=tf.float32)
  b0 = tf.Variable(tf.zeros([FLAGS.No*64]), dtype=tf.float32)
  h0 = tf.matmul(concated_pair, w0) + b0
  enpair1=tf.reshape(tf.slice(h0,[0,0],[FLAGS.batch_num,-1]),[-1,FLAGS.No,64]);
  enpair2=tf.reshape(tf.slice(h0,[FLAGS.batch_num,0],[FLAGS.batch_num,-1]),[-1,FLAGS.No,64]);
  enpair3=tf.reshape(tf.slice(h0,[FLAGS.batch_num*2,0],[FLAGS.batch_num,-1]),[-1,FLAGS.No,64]);
  enpair4=tf.reshape(tf.slice(h0,[FLAGS.batch_num*3,0],[FLAGS.batch_num,-1]),[-1,FLAGS.No,64]);
  enpair5=tf.reshape(tf.slice(h0,[FLAGS.batch_num*4,0],[FLAGS.batch_num,-1]),[-1,FLAGS.No,64]);
  three1=tf.concat([enpair1,enpair2],2);
  three2=tf.concat([enpair2,enpair3],2);
  three3=tf.concat([enpair3,enpair4],2);
  three4=tf.concat([enpair4,enpair5],2);
  # shared MLP
  three=tf.concat([three1,three2,three3,three4],0);
  three=tf.reshape(three,[-1,128]);
  w1 = tf.Variable(tf.truncated_normal([128, 64], stddev=0.1), dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([64]), dtype=tf.float32);
  h1 = tf.nn.relu(tf.matmul(three, w1) + b1);
  w2 = tf.Variable(tf.truncated_normal([64, 64], stddev=0.1), dtype=tf.float32);
  b2 = tf.Variable(tf.zeros([64]), dtype=tf.float32);
  h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
  w3 = tf.Variable(tf.truncated_normal([64, FLAGS.Ds], stddev=0.1), dtype=tf.float32);
  b3 = tf.Variable(tf.zeros([FLAGS.Ds]), dtype=tf.float32);
  h3 = tf.matmul(h2, w3) + b3;
  h3 = tf.reshape(h3,[-1,FLAGS.No,FLAGS.Ds]);
  S1=tf.slice(h3,[0,0,0],[FLAGS.batch_num,-1,-1]);
  S2=tf.slice(h3,[FLAGS.batch_num,0,0],[FLAGS.batch_num,-1,-1]);
  S3=tf.slice(h3,[FLAGS.batch_num*2,0,0],[FLAGS.batch_num,-1,-1]);
  S4=tf.slice(h3,[FLAGS.batch_num*3,0,0],[FLAGS.batch_num,-1,-1]);
  return S1,S2,S3,S4;

def core1(S,FLAGS):
  M=tf.unstack(S,FLAGS.No,1);
  # Self-Dynamics MLP
  SD_in=tf.reshape(S,[-1,FLAGS.Ds]);
  with tf.variable_scope('self-dynamics1'):
    w1 = tf.get_variable('w1',shape=[FLAGS.Ds, 64]);
    b1 = tf.get_variable('b1',shape=[64]);
    h1 = tf.nn.relu(tf.matmul(SD_in, w1) + b1);
    w2 = tf.get_variable('w2',shape=[64, 64]);
    b2 = tf.get_variable('b2',shape=[64]);
    h2 = tf.matmul(h1, w2) + b2;
  M_self = tf.reshape(h2,[-1,FLAGS.No,64]);
  # Relation MLP
  rel_num=int((FLAGS.No)*(FLAGS.No+1)/2);
  rel_in=np.zeros(rel_num,dtype=object);
  for i in range(rel_num):
    row_idx=int(i/FLAGS.No);
    col_idx=int(i%FLAGS.No);
    rel_in[i]=tf.concat([M[row_idx],M[col_idx]],1);
  rel_in=tf.concat(list(rel_in),0);
  with tf.variable_scope('Relation1'):
    w1 = tf.get_variable('w1',shape=[FLAGS.Ds*2, 64]);
    b1 = tf.get_variable('b1',shape=[64]);
    h1 = tf.nn.relu(tf.matmul(rel_in, w1) + b1);
    w2 = tf.get_variable('w2',shape=[64, 64]);
    b2 = tf.get_variable('b2',shape=[64]);
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
    w3 = tf.get_variable('w3',shape=[64, 64]);
    b3 = tf.get_variable('b3',shape=[64]);
    h3 = tf.matmul(h2, w3) + b3;
  M_rel=np.zeros(rel_num,dtype=object);
  for i in range(rel_num):
    row_idx=int(i/FLAGS.No);
    col_idx=int(i%FLAGS.No);
    M_rel[i]=tf.slice(h3,[(FLAGS.batch_num)*i,0],[(FLAGS.batch_num),-1]);
  M_rel2=np.zeros(FLAGS.No,dtype=object);
  for i in range(FLAGS.No):
    for j in range(FLAGS.No-1):
      M_rel2[i]+=M_rel[i*(FLAGS.No-1)+j];
  M_rel2=tf.stack(list(M_rel2),1);
  # M_update
  M_update=M_self+M_rel2;
  # Affector MLP
  aff_in=tf.reshape(M_update,[-1,64]);
  with tf.variable_scope('Affector1'):
    w1 = tf.get_variable('w1',shape=[64, 64]);
    b1 = tf.get_variable('b1',shape=[64]);
    h1 = tf.nn.relu(tf.matmul(aff_in, w1) + b1);
    w2 = tf.get_variable('w2',shape=[64, 64]);
    b2 = tf.get_variable('b2',shape=[64]);
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
    w3 = tf.get_variable('w3',shape=[64, 64]);
    b3 = tf.get_variable('b3',shape=[64]);
    h3 = tf.matmul(h2, w3) + b3;
  M_affect = tf.reshape(h3,[-1,FLAGS.No,64]);
  # Output MLP
  M_i_M_affect = tf.concat([S,M_update],2);
  out_in=tf.reshape(M_i_M_affect,[-1,FLAGS.Ds+64]);
  with tf.variable_scope('Output1'):
    w1 = tf.get_variable('w1',shape=[FLAGS.Ds+64, 64]);
    b1 = tf.get_variable('b1',shape=[64]);
    h1 = tf.nn.relu(tf.matmul(out_in, w1) + b1);
    w2 = tf.get_variable('w2',shape=[64, 64]);
    b2 = tf.get_variable('b2',shape=[64]);
    h2 = tf.matmul(h1, w2) + b2;
  h2_out = tf.reshape(h2,[-1,FLAGS.No,64]);
  return h2_out;

def core2(S,FLAGS):
  M=tf.unstack(S,FLAGS.No,1);
  # Self-Dynamics MLP
  SD_in=tf.reshape(S,[-1,FLAGS.Ds]);
  with tf.variable_scope('self-dynamics2'):
    w1 = tf.get_variable('w1',shape=[FLAGS.Ds, 64]);
    b1 = tf.get_variable('b1',shape=[64]);
    h1 = tf.nn.relu(tf.matmul(SD_in, w1) + b1);
    w2 = tf.get_variable('w2',shape=[64, 64]);
    b2 = tf.get_variable('b2',shape=[64]);
    h2 = tf.matmul(h1, w2) + b2;
  M_self = tf.reshape(h2,[-1,FLAGS.No,64]);
  # Relation MLP
  rel_num=int((FLAGS.No)*(FLAGS.No+1)/2);
  rel_in=np.zeros(rel_num,dtype=object);
  for i in range(rel_num):
    row_idx=int(i/FLAGS.No);
    col_idx=int(i%FLAGS.No);
    rel_in[i]=tf.concat([M[row_idx],M[col_idx]],1);
  rel_in=tf.concat(list(rel_in),0);
  with tf.variable_scope('Relation2'):
    w1 = tf.get_variable('w1',shape=[FLAGS.Ds*2, 64]);
    b1 = tf.get_variable('b1',shape=[64]);
    h1 = tf.nn.relu(tf.matmul(rel_in, w1) + b1);
    w2 = tf.get_variable('w2',shape=[64, 64]);
    b2 = tf.get_variable('b2',shape=[64]);
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
    w3 = tf.get_variable('w3',shape=[64, 64]);
    b3 = tf.get_variable('b3',shape=[64]);
    h3 = tf.matmul(h2, w3) + b3;
  M_rel=np.zeros(rel_num,dtype=object);
  for i in range(rel_num):
    row_idx=int(i/FLAGS.No);
    col_idx=int(i%FLAGS.No);
    M_rel[i]=tf.slice(h3,[(FLAGS.batch_num)*i,0],[(FLAGS.batch_num),-1]);
  M_rel2=np.zeros(FLAGS.No,dtype=object);
  for i in range(FLAGS.No):
    for j in range(FLAGS.No-1):
      M_rel2[i]+=M_rel[i*(FLAGS.No-1)+j];
  M_rel2=tf.stack(list(M_rel2),1);
  # M_update
  M_update=M_self+M_rel2;
  # Affector MLP
  aff_in=tf.reshape(M_update,[-1,64]);
  with tf.variable_scope('Affector2'):
    w1 = tf.get_variable('w1',shape=[64, 64]);
    b1 = tf.get_variable('b1',shape=[64]);
    h1 = tf.nn.relu(tf.matmul(aff_in, w1) + b1);
    w2 = tf.get_variable('w2',shape=[64, 64]);
    b2 = tf.get_variable('b2',shape=[64]);
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
    w3 = tf.get_variable('w3',shape=[64, 64]);
    b3 = tf.get_variable('b3',shape=[64]);
    h3 = tf.matmul(h2, w3) + b3;
  M_affect = tf.reshape(h3,[-1,FLAGS.No,64]);
  # Output MLP
  M_i_M_affect = tf.concat([S,M_update],2);
  out_in=tf.reshape(M_i_M_affect,[-1,FLAGS.Ds+64]);
  with tf.variable_scope('Output2'):
    w1 = tf.get_variable('w1',shape=[FLAGS.Ds+64, 64]);
    b1 = tf.get_variable('b1',shape=[64]);
    h1 = tf.nn.relu(tf.matmul(out_in, w1) + b1);
    w2 = tf.get_variable('w2',shape=[64, 64]);
    b2 = tf.get_variable('b2',shape=[64]);
    h2 = tf.matmul(h1, w2) + b2;
  h2_out = tf.reshape(h2,[-1,FLAGS.No,64]);
  return h2_out;

def core4(S,FLAGS):
  M=tf.unstack(S,FLAGS.No,1);
  # Self-Dynamics MLP
  SD_in=tf.reshape(S,[-1,FLAGS.Ds]);
  with tf.variable_scope('self-dynamics4'):
    w1 = tf.get_variable('w1',shape=[FLAGS.Ds, 64]);
    b1 = tf.get_variable('b1',shape=[64]);
    h1 = tf.nn.relu(tf.matmul(SD_in, w1) + b1);
    w2 = tf.get_variable('w2',shape=[64, 64]);
    b2 = tf.get_variable('b2',shape=[64]);
    h2 = tf.matmul(h1, w2) + b2;
  M_self = tf.reshape(h2,[-1,FLAGS.No,64]);
  # Relation MLP
  rel_num=int((FLAGS.No)*(FLAGS.No+1)/2);
  rel_in=np.zeros(rel_num,dtype=object);
  for i in range(rel_num):
    row_idx=int(i/FLAGS.No);
    col_idx=int(i%FLAGS.No);
    rel_in[i]=tf.concat([M[row_idx],M[col_idx]],1);
  rel_in=tf.concat(list(rel_in),0);
  with tf.variable_scope('Relation4'):
    w1 = tf.get_variable('w1',shape=[FLAGS.Ds*2, 64]);
    b1 = tf.get_variable('b1',shape=[64]);
    h1 = tf.nn.relu(tf.matmul(rel_in, w1) + b1);
    w2 = tf.get_variable('w2',shape=[64, 64]);
    b2 = tf.get_variable('b2',shape=[64]);
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
    w3 = tf.get_variable('w3',shape=[64, 64]);
    b3 = tf.get_variable('b3',shape=[64]);
    h3 = tf.matmul(h2, w3) + b3;
  M_rel=np.zeros(rel_num,dtype=object);
  for i in range(rel_num):
    row_idx=int(i/FLAGS.No);
    col_idx=int(i%FLAGS.No);
    M_rel[i]=tf.slice(h3,[(FLAGS.batch_num)*i,0],[(FLAGS.batch_num),-1]);
  M_rel2=np.zeros(FLAGS.No,dtype=object);
  for i in range(FLAGS.No):
    for j in range(FLAGS.No-1):
      M_rel2[i]+=M_rel[i*(FLAGS.No-1)+j];
  M_rel2=tf.stack(list(M_rel2),1);
  # M_update
  M_update=M_self+M_rel2;
  # Affector MLP
  aff_in=tf.reshape(M_update,[-1,64]);
  with tf.variable_scope('Affector4'):
    w1 = tf.get_variable('w1',shape=[64, 64]);
    b1 = tf.get_variable('b1',shape=[64]);
    h1 = tf.nn.relu(tf.matmul(aff_in, w1) + b1);
    w2 = tf.get_variable('w2',shape=[64, 64]);
    b2 = tf.get_variable('b2',shape=[64]);
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2);
    w3 = tf.get_variable('w3',shape=[64, 64]);
    b3 = tf.get_variable('b3',shape=[64]);
    h3 = tf.matmul(h2, w3) + b3;
  M_affect = tf.reshape(h3,[-1,FLAGS.No,64]);
  # Output MLP
  M_i_M_affect = tf.concat([S,M_update],2);
  out_in=tf.reshape(M_i_M_affect,[-1,FLAGS.Ds+64]);
  with tf.variable_scope('Output4'):
    w1 = tf.get_variable('w1',shape=[FLAGS.Ds+64, 64]);
    b1 = tf.get_variable('b1',shape=[64]);
    h1 = tf.nn.relu(tf.matmul(out_in, w1) + b1);
    w2 = tf.get_variable('w2',shape=[64, 64]);
    b2 = tf.get_variable('b2',shape=[64]);
    h2 = tf.matmul(h1, w2) + b2;
  h2_out = tf.reshape(h2,[-1,FLAGS.No,64]);
  return h2_out;

def DP(S1,S2,S3,S4,FLAGS):
  Sc1=core4(S1,FLAGS);
  Sc3=core2(S3,FLAGS);
  Sc4=core1(S4,FLAGS);
  # Aggregator MLP
  S=tf.concat([Sc1,Sc3,Sc4],2);
  S=tf.reshape(S,[-1,192]);
  with tf.variable_scope("DP"):
    w1 = tf.get_variable('w1',shape=[192, 32]);
    b1 = tf.get_variable('b1',shape=[32]);
    h1 = tf.nn.relu(tf.matmul(S, w1) + b1);
    w2 = tf.get_variable('w2',shape=[32, FLAGS.Ds]);
    b2 = tf.get_variable('b2',shape=[FLAGS.Ds]);
    h2 = tf.matmul(h1, w2) + b2;
  h2=tf.reshape(h2,[-1,FLAGS.No,FLAGS.Ds]);
  out_dp=tf.slice(h2,[0,0,0],[FLAGS.batch_num,-1,-1]);
  return out_dp;

def SD(output_dp,FLAGS):
  # State Decoder
  input_sd=tf.reshape(output_dp,[-1,FLAGS.Ds]);
  w1 = tf.Variable(tf.truncated_normal([FLAGS.Ds, 4], stddev=0.1), dtype=tf.float32);
  b1 = tf.Variable(tf.zeros([4]), dtype=tf.float32);
  h1 = tf.matmul(input_sd, w1) + b1;
  h1=tf.reshape(h1,[-1,FLAGS.No,4]);
  return h1;
