#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a re-implementation of testing code of this paper:
# X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley. �Removing Rain from Single Images via a Deep Detail Network�, CVPR, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)



import os
import skimage.io
import numpy as np
import tensorflow as tf
import training_v2 as Network
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()


os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # select GPU device


tf.compat.v1.reset_default_graph()

model_path = './model/'
pre_trained_model_path = './model/trained/model'


img_path = './TestData/input/' # the path of testing images
results_path = './TestData/results/' # the path of de-rained images



def _parse_function(filename):   
  image_string = tf.io.read_file(filename)  
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)  
  rainy = tf.cast(image_decoded, tf.float32)/255.0 
  return rainy 


if __name__ == '__main__':
   imgName = os.listdir(img_path)
   num_img = len(imgName)
   
   whole_path = []
   for i in range(num_img):
      whole_path.append(img_path + imgName[i])
      
    
   filename_tensor = tf.convert_to_tensor(value=whole_path, dtype=tf.string)     
   dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
   dataset = dataset.map(_parse_function)    
   dataset = dataset.prefetch(buffer_size=10)
   dataset = dataset.batch(batch_size=1).repeat()  
   iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
   
   rain = iterator.get_next() 


   output = Network.inference(rain, is_training = False)
   output = tf.clip_by_value(output, 0., 1.)
   output = output[0,:,:,:]

   config = tf.compat.v1.ConfigProto()
   config.gpu_options.allow_growth=True   
   saver = tf.compat.v1.train.Saver()


   with tf.compat.v1.Session(config=config) as sess: 
      with tf.device('/gpu:0'): 
          if tf.train.get_checkpoint_state(model_path):
              ckpt = tf.train.latest_checkpoint(model_path)  # try your own model
              saver.restore(sess, ckpt)
              print ("Loading model")
          else:
             saver.restore(sess, pre_trained_model_path) # try a pre-trained model
             print ("Loading pre-trained model")

          for i in range(num_img):     
             derained, ori = sess.run([output, rain])              
             derained = np.uint8(derained* 255.)
             index = imgName[i].rfind('.')
             name = imgName[i][:index]
             skimage.io.imsave(results_path + name +'.png', derained)         
             print('%d / %d images processed' % (i+1,num_img))
              
      print('All done')
   sess.close()   
   
   plt.subplot(1,2,1)     
   plt.imshow(ori[0,:,:,:])          
   plt.title('rainy')
   plt.subplot(1,2,2)    
   plt.imshow(derained)
   plt.title('derained')
   plt.show()     