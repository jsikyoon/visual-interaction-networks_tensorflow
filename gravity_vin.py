from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import copy
import tensorflow as tf

import math
import matplotlib.image as mpimg
import numpy as np
import time
from vin import VE, DP, SD
from physics_engine import make_video, make_image2
from constants import No,img_folder,data_folder,frame_num, set_num, roll_num

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

def rollout_DP(prev_out,cur_in):
  # rolling DP to roll_num
  S1,S2,S3,S4=tf.unstack(prev_out,4,0);
  S_pred=DP(S1,S2,S3,S4,FLAGS);
  res=tf.stack([S2,S3,S4,S_pred],0);
  return res;

def train():
  # Architecture Definition
  # Frame
  F=tf.placeholder(tf.float32, [None,6,FLAGS.height,FLAGS.weight,FLAGS.col_dim], name="F");
  F1,F2,F3,F4,F5,F6=tf.unstack(F,6,1);
  # Future Data
  label=tf.placeholder(tf.float32, [None,8,FLAGS.No,4], name="label");
  label_part=tf.unstack(label,8,1);
  # State Code
  S_label=tf.placeholder(tf.float32, [None,4,FLAGS.No,4], name="S_label");
  S_label_part=tf.unstack(S_label,4,1);
  # discount factor
  df=tf.placeholder(tf.float32,[],name="DiscountFactor");
  # x and y coordinate channels
  x_cor=tf.placeholder(tf.float32, [None,FLAGS.height,FLAGS.weight,1], name="x_cor");
  y_cor=tf.placeholder(tf.float32, [None,FLAGS.height,FLAGS.weight,1], name="y_cor");

  # Visual Encoder
  S1,S2,S3,S4=VE(F1,F2,F3,F4,F5,F6,x_cor,y_cor,FLAGS);

  # Rolling Dynamic Predictor
  roll_in=tf.identity(tf.Variable(tf.zeros([roll_num]),dtype=tf.float32));
  roll_out=tf.scan(rollout_DP,roll_in,initializer=tf.stack([S1,S2,S3,S4],0));
  S_pred=tf.unstack(roll_out,4,1)[-1];
  S_pred=tf.reshape(tf.stack(tf.unstack(S_pred,FLAGS.batch_num,1),0),[-1,FLAGS.No,FLAGS.Ds]);
  
  # State Decoder
  S=tf.concat([S1,S2,S3,S4,S_pred],0);
  out_sd=SD(S,FLAGS);
  S_est=np.zeros(4,dtype=object);
  for i in range(4):
    S_est[i]=tf.slice(out_sd,[FLAGS.batch_num*i,0,0],[FLAGS.batch_num,-1,-1]);
  label_pred=tf.reshape(tf.slice(out_sd,[FLAGS.batch_num*4,0,0],[-1,-1,-1]),[FLAGS.batch_num,roll_num,FLAGS.No,4]);
  label_pred8=tf.unstack(label_pred,roll_num,1)[:8];
  
  # loss and optimizer
  mse=df*tf.reduce_mean(tf.reduce_mean(tf.square(label_pred8[0]-label_part[0]),[1,2]));
  for i in range(1,8):
    mse+=(df**(i+1))*tf.reduce_mean(tf.reduce_mean(tf.square(label_pred8[i]-label_part[i]),[1,2]));
  mse=mse/8;
  ve_loss=tf.reduce_mean(tf.reduce_mean(tf.square(S_est[0]-S_label_part[0]),[1,2]));
  for i in range(1,4):
    ve_loss+=tf.reduce_mean(tf.reduce_mean(tf.square(S_est[i]-S_label_part[i]),[1,2]));
  ve_loss=ve_loss/4;
  total_loss=mse+ve_loss;
  optimizer = tf.train.AdamOptimizer(0.0005);
  trainer=optimizer.minimize(total_loss);

  # tensorboard
  """
  params_list=tf.global_variables();
  for i in range(len(params_list)):
    variable_summaries(params_list[i],i);
  """
  tf.summary.scalar('tr_loss',total_loss);
  merged=tf.summary.merge_all();
  writer=tf.summary.FileWriter(FLAGS.log_dir);

  sess=tf.InteractiveSession();
  tf.global_variables_initializer().run();

  # Get Training Image and Data 
  total_img=np.zeros((FLAGS.set_num,int(frame_num),FLAGS.height,FLAGS.weight,FLAGS.col_dim),dtype=float);
  for i in range(FLAGS.set_num):
    for j in range(int(frame_num)):
      total_img[i,j]=mpimg.imread(img_folder+"train/"+str(i)+'_'+str(j)+'.png')[:,:,:FLAGS.col_dim];
  total_data=np.zeros((FLAGS.set_num,int(frame_num),FLAGS.No*5),dtype=float);
  for i in range(FLAGS.set_num):
    f=open(data_folder+"train/"+str(i)+".csv","r");
    total_data[i]=[line[:-1].split(",") for line in f.readlines()];
  total_data=np.reshape(total_data,[FLAGS.set_num,int(frame_num),FLAGS.No,5]); 

  # reshape img and data
  input_img=np.zeros((FLAGS.set_num*(int(frame_num)-14+1),6,FLAGS.height,FLAGS.weight,FLAGS.col_dim),dtype=float);
  output_label=np.zeros((FLAGS.set_num*(int(frame_num)-14+1),8,FLAGS.No,4),dtype=float);
  output_S_label=np.zeros((FLAGS.set_num*(int(frame_num)-14+1),4,FLAGS.No,4),dtype=float);
  for i in range(FLAGS.set_num):
    for j in range(int(frame_num)-14+1):
      input_img[i*(int(frame_num)-14+1)+j]=total_img[i,j:j+6];
      output_label[i*(int(frame_num)-14+1)+j]=np.reshape(total_data[i,j+6:j+14],[8,FLAGS.No,5])[:,:,1:5];
      output_S_label[i*(int(frame_num)-14+1)+j]=np.reshape(total_data[i,j+2:j+6],[4,FLAGS.No,5])[:,:,1:5];

  # shuffle
  tr_data_num=int(len(input_img)*1);
  val_data_num=int(len(input_img)*0);
  total_idx=list(range(len(input_img)));np.random.shuffle(total_idx);
  mixed_img=input_img[total_idx];mixed_label=output_label[total_idx];mixed_S_label=output_S_label[total_idx];
  tr_data=mixed_img[:tr_data_num];tr_label=mixed_label[:tr_data_num];tr_S_label=mixed_S_label[:tr_data_num];
  val_data=mixed_img[tr_data_num:(tr_data_num+val_data_num)];val_label=mixed_label[tr_data_num:(tr_data_num+val_data_num)];val_S_label=mixed_S_label[tr_data_num:(tr_data_num+val_data_num)];

  # x-cor and y-cor setting
  nx, ny = (FLAGS.weight, FLAGS.height);
  x = np.linspace(0, 1, nx);
  y = np.linspace(0, 1, ny);
  xv, yv = np.meshgrid(x, y);
  xv=np.reshape(xv,[FLAGS.height,FLAGS.weight,1]);
  yv=np.reshape(yv,[FLAGS.height,FLAGS.weight,1]);
  xcor=np.zeros((FLAGS.batch_num*5,FLAGS.height,FLAGS.weight,1),dtype=float);
  ycor=np.zeros((FLAGS.batch_num*5,FLAGS.height,FLAGS.weight,1),dtype=float);
  for i in range(FLAGS.batch_num*5):
    xcor[i]=xv; ycor[i]=yv;

  # beta
  beta = int(FLAGS.max_epoches*0.05);
  # training
  for i in range(FLAGS.max_epoches):
    #df_value=1-math.exp(-1*i/beta);
    df_value=1;
    tr_loss=0;
    tr_loss2=0;
    for j in range(int(len(tr_data)/FLAGS.batch_num)):
      batch_data=tr_data[j*FLAGS.batch_num:(j+1)*FLAGS.batch_num];
      batch_label=tr_label[j*FLAGS.batch_num:(j+1)*FLAGS.batch_num];
      batch_S_label=tr_S_label[j*FLAGS.batch_num:(j+1)*FLAGS.batch_num];
      if(j==0):
        estimated,summary,tr_loss_part,tr_loss_part2,_=sess.run([label,merged,mse,ve_loss,trainer],feed_dict={F:batch_data,label:batch_label,S_label:batch_S_label,x_cor:xcor,y_cor:ycor,df:df_value});
        writer.add_summary(summary,i);
      else:
        tr_loss_part,tr_loss_part2,_=sess.run([mse,ve_loss,trainer],feed_dict={F:batch_data,label:batch_label,S_label:batch_S_label,x_cor:xcor,y_cor:ycor,df:df_value});
      tr_loss+=tr_loss_part;
      tr_loss2+=tr_loss_part2;
    tr_idx=list(range(len(tr_data)));np.random.shuffle(tr_idx);
    tr_data=tr_data[tr_idx];
    tr_label=tr_label[tr_idx];
    tr_S_label=tr_S_label[tr_idx];
    print("Epoch "+str(i+1)+" Training mse: "+str(tr_loss/(int(len(tr_data)/FLAGS.batch_num)))+" Training ve loss: "+str(tr_loss2/int(len(tr_data)/FLAGS.batch_num)));
  
  # Get Test Image and Data 
  ts_img=np.zeros((1,int(frame_num),FLAGS.height,FLAGS.weight,FLAGS.col_dim),dtype=float);
  for i in range(1):
    for j in range(int(frame_num)):
      ts_img[i,j]=mpimg.imread(img_folder+"test/"+str(i)+"_"+str(j)+'.png')[:,:,:FLAGS.col_dim];
  ts_data=np.zeros((1,int(frame_num),FLAGS.No*5),dtype=float);
  for i in range(1):
    f=open(data_folder+"test/"+str(i)+".csv","r");
    ts_data[i]=[line[:-1].split(",") for line in f.readlines()];
  
  # reshape img and data
  input_img=np.zeros((1*(int(frame_num)-14+1),6,FLAGS.height,FLAGS.weight,FLAGS.col_dim),dtype=float);
  output_label=np.zeros((1*(int(frame_num)-14+1),8,FLAGS.No,4),dtype=float);
  output_S_label=np.zeros((1*(int(frame_num)-14+1),4,FLAGS.No,4),dtype=float);
  for i in range(1):
    for j in range(int(frame_num)-14+1):
      input_img[i*(int(frame_num)-14+1)+j]=total_img[i,j:j+6];
      output_label[i*(int(frame_num)-14+1)+j]=np.reshape(ts_data[i,j+6:j+14],[8,FLAGS.No,5])[:,:,1:5];
      output_S_label[i*(int(frame_num)-14+1)+j]=np.reshape(ts_data[i,j+2:j+6],[4,FLAGS.No,5])[:,:,1:5];

  xy_origin=output_label[:(int(frame_num)-14+1-4+1),0,:,0:2];
  xy_estimated=np.zeros((roll_num,No,2),dtype=float);
 
  # Rollout 
  posi=sess.run(label_pred,feed_dict={F:input_img[0:4],label:output_label[0:4],S_label:output_S_label[0:4],x_cor:xcor,y_cor:ycor,df:1.0})[0];
  #xy_estimated=posi[:,:,:2];
  velo=posi[:,:,2:4];
  xy_estimated[0]=output_S_label[3][3][:,:2]+velo[0]*0.01;
  for i in range(1,len(posi)):
    xy_estimated[i]=xy_estimated[i-1]+velo[i]*0.01;
  
  # Saving
  print("Image Making");
  make_image2(xy_origin,img_folder+"results/","true");
  make_image2(xy_estimated,img_folder+"results/","modeling");
  print("Done");

def main(_):
  FLAGS.log_dir+=str(int(time.time()));
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  FLAGS.No=No;
  FLAGS.set_num=set_num;
  FLAGS.height=32;
  FLAGS.weight=32;
  FLAGS.col_dim=4;
  train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', type=str, default='/tmp/vin/logs/',
                      help='Summaries log directry')
  parser.add_argument('--batch_num', type=int, default=4,
                      help='The number of data on each mini batch')
  parser.add_argument('--max_epoches', type=int, default=80000,
                      help='Maximum limitation of epoches')
  parser.add_argument('--Ds', type=int, default=64,
                      help='The State Code Dimension')
  parser.add_argument('--fil_num', type=int, default=128,
                      help='The Number of filters')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
