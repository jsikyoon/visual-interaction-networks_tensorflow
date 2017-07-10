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
from vin import VE, DP, SD
from physics_engine import make_video
from constants import No,img_folder,data_folder

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
  label=tf.placeholder(tf.float32, [None,FLAGS.No,4], name="label");
  S=tf.placeholder(tf.float32, [None,4,FLAGS.No,5], name="S");
  S1,S2,S3,S4=tf.unstack(S,4,1);

  out_dp=DP(S1,S2,S3,S4,FLAGS);
  out_sd=SD(out_dp,FLAGS);
  
  # loss and optimizer
  mse=tf.reduce_mean(tf.reduce_mean(tf.square(out_sd-label),[1,2]));
  optimizer = tf.train.AdamOptimizer(0.001);
  trainer=optimizer.minimize(mse);

  # tensorboard
  params_list=tf.global_variables();
  for i in range(len(params_list)):
    variable_summaries(params_list[i],i);
  tf.summary.scalar('mse',mse);
  merged=tf.summary.merge_all();
  writer=tf.summary.FileWriter(FLAGS.log_dir);

  sess=tf.InteractiveSession();
  tf.global_variables_initializer().run();

  # Get Training Data 
  total_data=np.zeros((FLAGS.set_num,1000,FLAGS.No*5),dtype=float);
  for i in range(FLAGS.set_num):
    f=open(data_folder+"train/"+str(i)+".csv","r");
    total_data[i]=[line[:-1].split(",") for line in f.readlines()];
  
  # reshape data
  input_data=np.zeros((FLAGS.set_num*(1000-7+1),4,FLAGS.No,5),dtype=float);
  output_label=np.zeros((FLAGS.set_num*(1000-7+1),FLAGS.No,4),dtype=float);
  for i in range(FLAGS.set_num):
    for j in range(1000-7+1):
      input_data[i*(1000-7+1)+j]=np.reshape(total_data[i,j+2:j+6],[4,FLAGS.No,5]);
      output_label[i*(1000-7+1)+j]=np.reshape(total_data[i,j+6],[FLAGS.No,5])[:,1:5];
  
  # Normalization
  weights_list=np.sort(np.reshape(input_data[:,:,:,0],[1,-1])[0]);
  weights_median=weights_list[int(len(weights_list)*0.5)];
  weights_min=weights_list[int(len(weights_list)*0.05)];
  weights_max=weights_list[int(len(weights_list)*0.95)];
  position_list=np.sort(np.reshape(input_data[:,:,:,1:3],[1,-1])[0]);
  position_median=position_list[int(len(position_list)*0.5)];
  position_min=position_list[int(len(position_list)*0.05)];
  position_max=position_list[int(len(position_list)*0.95)];
  velocity_list=np.sort(np.reshape(input_data[:,:,:,3:5],[1,-1])[0]);
  velocity_median=velocity_list[int(len(velocity_list)*0.5)];
  velocity_min=velocity_list[int(len(velocity_list)*0.05)];
  velocity_max=velocity_list[int(len(velocity_list)*0.95)];

  input_data[:,:,:,0]=(input_data[:,:,:,0]-weights_median)*(2/(weights_max-weights_min));
  input_data[:,:,:,1:3]=(input_data[:,:,:,1:3]-position_median)*(2/(position_max-position_min));
  input_data[:,:,:,3:5]=(input_data[:,:,:,3:5]-velocity_median)*(2/(velocity_max-velocity_min));

  # shuffle
  tr_data_num=int(len(input_data)*0.8);
  val_data_num=int(len(input_data)*0.1);
  total_idx=range(len(input_data));np.random.shuffle(total_idx);
  mixed_data=input_data[total_idx];mixed_label=output_label[total_idx];
  tr_data=mixed_data[:tr_data_num];tr_label=mixed_label[:tr_data_num];
  val_data=mixed_data[tr_data_num:(tr_data_num+val_data_num)];val_label=mixed_label[tr_data_num:(tr_data_num+val_data_num)];
  
  # training
  for i in range(FLAGS.max_epoches):
    tr_loss=0;
    for j in range(int(len(tr_data)/FLAGS.batch_num)):
      batch_data=tr_data[j*FLAGS.batch_num:(j+1)*FLAGS.batch_num];
      batch_label=tr_label[j*FLAGS.batch_num:(j+1)*FLAGS.batch_num];
      if(j==0):
        estimated,summary,tr_loss_part,_=sess.run([label,merged,mse,trainer],feed_dict={S:batch_data,label:batch_label});
        writer.add_summary(summary,i);
        #print(estimated);
      else:
        tr_loss_part,_=sess.run([mse,trainer],feed_dict={S:batch_data,label:batch_label});
      tr_loss+=tr_loss_part;
    tr_idx=range(len(tr_data));np.random.shuffle(tr_idx);
    tr_data=tr_data[tr_idx];
    tr_label=tr_label[tr_idx];
    val_loss=0;
    for j in range(int(len(val_data)/FLAGS.batch_num)):
      batch_data=val_data[j*FLAGS.batch_num:(j+1)*FLAGS.batch_num];
      batch_label=val_label[j*FLAGS.batch_num:(j+1)*FLAGS.batch_num];
      val_loss_part,_=sess.run([mse,trainer],feed_dict={S:batch_data,label:batch_label});
      val_loss+=val_loss_part;
    val_idx=range(len(val_data));np.random.shuffle(val_idx);
    val_data=val_data[val_idx];
    val_label=val_label[val_idx];
    print("Epoch "+str(i+1)+" Training MSE: "+str(tr_loss/(int(len(tr_data)/FLAGS.batch_num)))+" Validation MSE: "+str(val_loss/(j+1)));
 
  
  ts_frame_num=300;
  # Get Test Data 
  ts_data=np.zeros((1,1000,FLAGS.No*5),dtype=float);
  for i in range(1):
    f=open(data_folder+"test/"+str(i)+".csv","r");
    ts_data[i]=[line[:-1].split(",") for line in f.readlines()];

  # reshape data
  input_data=np.zeros((1*(1000-7+1),4,FLAGS.No,5),dtype=float);
  output_label=np.zeros((1*(1000-7+1),FLAGS.No,4),dtype=float);
  for i in range(1):
    for j in range(1000-7+1):
      input_data[i*(1000-7+1)+j]=np.reshape(ts_data[i,j+2:j+6],[4,FLAGS.No,5]);
      output_label[i*(1000-7+1)+j]=np.reshape(ts_data[i,j+6],[FLAGS.No,5])[:,1:5];
  
  # Normalization
  input_data[:,:,:,0]=(input_data[:,:,:,0]-weights_median)*(2/(weights_max-weights_min));
  input_data[:,:,:,1:3]=(input_data[:,:,:,1:3]-position_median)*(2/(position_max-position_min));
  input_data[:,:,:,3:5]=(input_data[:,:,:,3:5]-velocity_median)*(2/(velocity_max-velocity_min));
  
  xy_origin=output_label[:,:,0:2];
  xy_estimated=np.zeros((1*(1000-7+1),No,2),dtype=float);
  for i in range(len(input_data)):
    xy_estimated[i]=sess.run(label,feed_dict={S:[input_data[i]],label:[output_label[i]]})[:,:,1:3];
  print("Video Recording");
  make_video(xy_origin[:ts_frame_num],"true"+str(time.time())+".mp4");
  make_video(xy_estimated[:ts_frame_num],"modeling"+str(time.time())+".mp4");
  print("Done");
  

def main(_):
  FLAGS.log_dir+=str(int(time.time()));
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  FLAGS.No=No;
  FLAGS.height=32;
  FLAGS.weight=32;
  FLAGS.col_dim=4;
  train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', type=str, default='/tmp/vin/logs/',
                      help='Summaries log directry')
  parser.add_argument('--set_num', type=int, default=1,
                      help='the number of training sets')
  parser.add_argument('--batch_num', type=int, default=4,
                      help='The number of data on each mini batch')
  parser.add_argument('--max_epoches', type=int, default=1000,
                      help='Maximum limitation of epoches')
  parser.add_argument('--Ds', type=int, default=5,
                      help='The dimension of State')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
