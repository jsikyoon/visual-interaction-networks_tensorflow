from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import numpy as np
import time
from math import sin, cos, radians, pi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import cv2
from constants import No
from constants import img_folder
from constants import data_folder
from constants import frame_num
from constants import frame_step
from constants import set_num
import cifar10

# 7 features on the state [mass,x,y,x_vel,y_vel]
fea_num=5;
# G 
#G = 6.67428e-11;
G=10**5;
# time step
diff_t=0.01;

def init(frame_num,n_body,fea_num,orbit):
  data=np.zeros((frame_num,n_body,fea_num),dtype=float);
  if(orbit):
    data[0][0][0]=100;
    data[0][0][1:5]=0.0;
    for i in range(1,n_body):
      data[0][i][0]=np.random.rand()*8.98+0.02;
      distance=np.random.rand()*50.0+50.0;
      theta=np.random.rand()*360;
      theta_rad = pi/2 - radians(theta);    
      data[0][i][1]=distance*cos(theta_rad);
      data[0][i][2]=distance*sin(theta_rad);
      data[0][i][3]=-1*data[0][i][2]/norm(data[0][i][1:3])*(G*data[0][0][0]/norm(data[0][i][1:3])**2)*distance/1000;
      data[0][i][4]=data[0][i][1]/norm(data[0][i][1:3])*(G*data[0][0][0]/norm(data[0][i][1:3])**2)*distance/1000;
  else:
    for i in range(n_body):
      data[0][i][0]=np.random.rand()*8.98+0.02;
      distance=np.random.rand()*90.0+10.0;
      theta=np.random.rand()*360;
      theta_rad = pi/2 - radians(theta);    
      data[0][i][1]=distance*cos(theta_rad);
      data[0][i][2]=distance*sin(theta_rad);
      data[0][i][3]=np.random.rand()*6.0-3.0;
      data[0][i][4]=np.random.rand()*6.0-3.0;
  return data;      

def norm(x):
  return np.sqrt(np.sum(x**2));

def get_f(reciever,sender):
  diff=sender[1:3]-reciever[1:3];
  distance=norm(diff);
  if(distance<10):
    distance=10;
  return G*reciever[0]*sender[0]/(distance**3)*diff;
 
def calc(cur_state,n_body):
  next_state=np.zeros((n_body,fea_num),dtype=float);
  f_mat=np.zeros((n_body,n_body,2),dtype=float);
  f_sum=np.zeros((n_body,2),dtype=float);
  acc=np.zeros((n_body,2),dtype=float);
  for i in range(n_body):
    for j in range(i+1,n_body):
      if(j!=i):
        f=get_f(cur_state[i][:3],cur_state[j][:3]);  
        f_mat[i,j]+=f;
        f_mat[j,i]-=f;  
    f_sum[i]=np.sum(f_mat[i],axis=0); 
    acc[i]=f_sum[i]/cur_state[i][0];
    next_state[i][0]=cur_state[i][0];
    next_state[i][3:5]=cur_state[i][3:5]+acc[i]*diff_t;
    next_state[i][1:3]=cur_state[i][1:3]+next_state[i][3:5]*diff_t;
  return next_state;

def gen(n_body,orbit):
  # initialization on just first state
  data=init(frame_num*frame_step,n_body,fea_num,orbit);
  for i in range(1,frame_num*frame_step):
    data[i]=calc(data[i-1],n_body);
  data=data[0:frame_num*frame_step:frame_step];
  return data;

def make_video(xy,filename):
  FFMpegWriter = manimation.writers['ffmpeg']
  metadata = dict(title='Movie Test', artist='Matplotlib',
                  comment='Movie support!')
  writer = FFMpegWriter(fps=15, metadata=metadata)
  mydpi=100;
  #fig = plt.figure(figsize=(128/mydpi,128/mydpi))
  fig = plt.figure(figsize=(32/mydpi,32/mydpi))
  plt.xlim(-200, 200)
  plt.ylim(-200, 200)
  fig_num=len(xy);
  #color=['ro','bo','go','ko','yo','mo','co'];
  color=['r','b','g','k','y','m','c'];
  with writer.saving(fig, filename, len(xy)):
    for i in range(len(xy)):
      for j in range(len(xy[0])):
        #plt.plot(xy[i,j,1],xy[i,j,0],color[j%len(color)]);
        plt.scatter(xy[i,j,1],xy[i,j,0],c=color[j%len(color)],s=0.5);
      writer.grab_frame();

def make_image(xy,img_folder,prefix,bg_img):
  if not os.path.exists(img_folder):
    os.makedirs(img_folder);
  fig_num=len(xy);
  mydpi=100;
  for i in range(fig_num):
    fig = plt.figure(figsize=(32/mydpi,32/mydpi))
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    plt.axis('off');
    plt.imshow(bg_img,extent=[-200,200,-200,200]);
    color=['r','b','g','k','y','m','c'];
    for j in range(len(xy[0])):
      plt.scatter(xy[i,j,1],xy[i,j,0],c=color[j%len(color)],s=2);
    fig.savefig(img_folder+prefix+"_"+str(i)+".png",dpi=mydpi);

def make_image2(xy,img_folder,prefix):
  if not os.path.exists(img_folder):
    os.makedirs(img_folder);
  fig_num=len(xy);
  mydpi=100;
  for i in range(fig_num):
    #fig = plt.figure(figsize=(32/mydpi,32/mydpi))
    fig = plt.figure(figsize=(128/mydpi,128/mydpi))
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    plt.axis('off');
    color=['r','b','g','k','y','m','c'];
    for j in range(len(xy[0])):
      #plt.scatter(xy[i,j,1],xy[i,j,0],c=color[j%len(color)],s=0.5);
      plt.scatter(xy[i,j,1],xy[i,j,0],c=color[j%len(color)],s=5);
    fig.savefig(img_folder+prefix+"_"+str(i)+".png",dpi=mydpi);

def make_file(data,data_folder,prefix):
  if not os.path.exists(data_folder):
    os.makedirs(data_folder);
  data=np.array(np.reshape(data,[-1,No*5]),dtype=str);
  f=open(data_folder+prefix+".csv","w");
  for i in range(len(data)):
    f.writelines(",".join(data[i])+"\n");

def gen_make(n_body,orbit,img_folder,data_folder,prefix):  
  data=gen(n_body,orbit);
  xy=data[:,:,1:3];
  make_image(xy,img_folder,str(prefix));
  make_file(data,data_folder,str(prefix));
  
if __name__=='__main__':
  # Get CIFAR 10 dataset
  cifar10.maybe_download_and_extract();
  cifar_data_dir="/tmp/cifar10_data/cifar-10-batches-bin/"
  tr_label_cifar10=np.zeros((50000,1),dtype=float);
  for i in range(1,6):
    file_name=os.path.join(cifar_data_dir,"data_batch_"+str(i)+".bin");
    f = open(file_name,"rb");
    data=np.reshape(bytearray(f.read()),[10000,3073]);
    if(i==1):
      tr_data_cifar10=data[:,1:]/255.0;
    else:
      tr_data_cifar10=np.append(tr_data_cifar10,data[:,1:]/255.0,axis=0);
    for j in range(len(data)):
      tr_label_cifar10[(i-1)*10000+j]=data[j,0];
  rand_idx=list(range(50000));np.random.shuffle(rand_idx);
  # Making Training Data
  for i in range(set_num):
    bg_img=np.reshape(tr_data_cifar10[rand_idx[i]],[32,32,3]);
    #bg_img=np.ones((32,32,3));
    data=gen(No,True);
    xy=data[:,:,1:3];
    make_image(xy,img_folder+"train/",str(i),bg_img);
    make_file(data,data_folder+"train/",str(i));
  # Making Test Data
  bg_img=np.reshape(tr_data_cifar10[rand_idx[i]],[32,32,3]);
  data=gen(No,True);
  #bg_img=np.ones((32,32,3));
  xy=data[:,:,1:3];
  make_image(xy,img_folder+"test/",str(0),bg_img);
  make_file(data,data_folder+"test/",str(0));
  #make_video(xy,"test.mp4");
