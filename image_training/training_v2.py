#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a re-implementation of training code of our paper:
# X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley. �Removing Rain from Single Images via a Deep Detail Network�, CVPR, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)

import os
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from GuidedFilter import guided_filter

tf.compat.v1.disable_eager_execution()



##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################

tf.compat.v1.reset_default_graph()

##################### Network parameters ###################################
num_feature = 16             # number of feature maps
num_channels = 3             # number of input's channels 
patch_size = 64              # patch size 
KernelSize = 3               # kernel size 
learning_rate = 0.1          # learning rate
iterations = int(1.1*1e5)    # iterations
batch_size = 20              # batch size
save_model_path = "./model/" # saved model's path
model_name = 'model-epoch'   # saved model's name
############################################################################


input_path = "./TrainData/input/"    # the path of rainy images
gt_path = "./TrainData/label/"       # the path of ground truth


input_files = os.listdir(input_path)
gt_files = os.listdir(gt_path) 
 
# randomly select image patches
def _parse_function(filename, label):  
     
  image_string = tf.io.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)  
  rainy = tf.cast(image_decoded, tf.float32)/255.0
  
  
  image_string = tf.io.read_file(label)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)  
  label = tf.cast(image_decoded, tf.float32)/255.0

  t = time.time()
  rainy = tf.image.random_crop(rainy, [patch_size, patch_size ,3],seed = t)   # randomly select patch
  label = tf.image.random_crop(label, [patch_size, patch_size ,3],seed = t)
  return rainy, label 




# network structure
def inference(images, is_training):
    regularizer = tf.keras.regularizers.l2(l = 0.5 * (1e-10))
    initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")

    base = guided_filter(images, images, 15, 1, nhwc=True) # using guided filter for obtaining base layer
    detail = images - base   # detail layer

   #  layer 1
    with tf.compat.v1.variable_scope('layer_1'):
         output = tf.compat.v1.layers.conv2d(detail, num_feature, KernelSize, padding = 'same', kernel_initializer = initializer, 
                                   kernel_regularizer = regularizer, name='conv_1')
         output = tf.compat.v1.layers.batch_normalization(output, training=is_training, name='bn_1')
         output_shortcut = tf.nn.relu(output, name='relu_1')
  
   #  layers 2 to 29
    for i in range(14):
        with tf.compat.v1.variable_scope('layer_%d'%(i*2+2)):	
             output = tf.compat.v1.layers.conv2d(output_shortcut, num_feature, KernelSize, padding='same', kernel_initializer = initializer, 
                                       kernel_regularizer = regularizer, name=('conv_%d'%(i*2+2)))
             output = tf.compat.v1.layers.batch_normalization(output, training=is_training, name=('bn_%d'%(i*2+2)))	
             output = tf.nn.relu(output, name=('relu_%d'%(i*2+2)))


        with tf.compat.v1.variable_scope('layer_%d'%(i*2+3)): 
             output = tf.compat.v1.layers.conv2d(output, num_feature, KernelSize, padding='same', kernel_initializer = initializer,
                                       kernel_regularizer = regularizer, name=('conv_%d'%(i*2+3)))
             output = tf.compat.v1.layers.batch_normalization(output, training=is_training, name=('bn_%d'%(i*2+3)))
             output = tf.nn.relu(output, name=('relu_%d'%(i*2+3)))

        output_shortcut = tf.add(output_shortcut, output)   # shortcut

   # layer 30
    with tf.compat.v1.variable_scope('layer_30'):
         output = tf.compat.v1.layers.conv2d(output_shortcut, num_channels, KernelSize, padding='same',   kernel_initializer = initializer, 
                                   kernel_regularizer = regularizer, name='conv_30')
         neg_residual = tf.compat.v1.layers.batch_normalization(output, training=is_training, name='bn_30')

    final_out = tf.add(images, neg_residual)

    return final_out

  


if __name__ == '__main__':   
   RainName = os.listdir(input_path)
   for i in range(len(RainName)):
      RainName[i] = input_path + RainName[i]
      
   LabelName = os.listdir(gt_path)    
   for i in range(len(LabelName)):
       LabelName[i] = gt_path + LabelName[i] 
    
   filename_tensor = tf.convert_to_tensor(value=RainName, dtype=tf.string)  
   labels_tensor = tf.convert_to_tensor(value=LabelName, dtype=tf.string)   
   dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, labels_tensor))
   dataset = dataset.map(_parse_function)    
   dataset = dataset.prefetch(buffer_size=batch_size * 10)
   dataset = dataset.batch(batch_size).repeat()  
   iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
   
   rainy, labels = iterator.get_next()     
   
   
   outputs = inference(rainy, is_training = True)
   loss = tf.reduce_mean(input_tensor=tf.square(labels - outputs))    # MSE loss

   
   lr_ = learning_rate
   lr = tf.compat.v1.placeholder(tf.float32 ,shape = [])  

   update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
   with tf.control_dependencies(update_ops):
        train_op =  tf.compat.v1.train.MomentumOptimizer(lr, 0.9).minimize(loss) 

   
   all_vars = tf.compat.v1.trainable_variables()   
   g_list = tf.compat.v1.global_variables()
   bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
   bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
   all_vars += bn_moving_vars
   print("Total parameters' number: %d" %(np.sum([np.prod(v.get_shape().as_list()) for v in all_vars])))  
   saver = tf.compat.v1.train.Saver(var_list=all_vars, max_to_keep=5)
   
   
   config = tf.compat.v1.ConfigProto()
   config.gpu_options.per_process_gpu_memory_fraction = 0.8 # GPU setting
   config.gpu_options.allow_growth = True
   init =  tf.group(tf.compat.v1.global_variables_initializer(), 
                         tf.compat.v1.local_variables_initializer())  
    
   with tf.compat.v1.Session(config=config) as sess:      
      with tf.device('/gpu:0'): 
            sess.run(init)
            tf.compat.v1.get_default_graph().finalize()
            
            if tf.train.get_checkpoint_state('./model/'):   # load previous trained models
               ckpt = tf.train.latest_checkpoint('./model/')
               saver.restore(sess, ckpt)
               ckpt_num = re.findall(r'(\w*[0-9]+)\w*',ckpt)
               start_point = int(ckpt_num[0]) + 1   
               print("successfully load previous model")
       
            else:   # re-training if no previous trained models
               start_point = 0    
               print("re-training")
    
    
            check_data, check_label = sess.run([rainy, labels])
            print("Check patch pair:")  
            plt.subplot(1,2,1)     
            plt.imshow(check_data[0,:,:,:])
            plt.title('input')         
            plt.subplot(1,2,2)    
            plt.imshow(check_label[0,:,:,:])
            plt.title('ground truth')        
            plt.show()
    
    
            start = time.time()  
            
            for j in range(start_point,iterations):   #  iterations
                if j+1 > int(5e4):
                    lr_ = learning_rate*0.1
                if j+1 > int(1e5):
                    lr_ = learning_rate*0.01             
                    
    
                _,Training_Loss = sess.run([train_op,loss], feed_dict={lr: lr_}) # training
          
                if np.mod(j+1,100) == 0 and j != 0: # save the model every 100 iterations
                   end = time.time()              
                   print ('%d / %d iterations, learning rate = %.3f, Training Loss = %.4f, runtime = %.1f s'
                          % (j+1, iterations, lr_, Training_Loss, (end - start)))                  
                   save_path_full = os.path.join(save_model_path, model_name) # save model
                   saver.save(sess, save_path_full, global_step = j+1, write_meta_graph=False)
                   start = time.time()
                   
            print('Training is finished.')
   sess.close()  