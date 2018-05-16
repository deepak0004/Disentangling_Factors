import tensorflow as tf 
import numpy as np 
import pickle, os, sys
import math
from scipy import misc 
from model import Model 
from dataset import Dataset 
import cv2

def restoring(saver, sess):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

def swapper(src_img, att_img, model_dir, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = '' 
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    sess = tf.Session()

    saver = tf.train.Saver()
    restoring(saver, sess)
 
    out2, out1 = sess.run([model.Ae, model.Bx], feed_dict={model.Ax: att_img, model.Be: src_img})
    misc.imsave('output1.jpg', out1[0])
    misc.imsave('output2.jpg', out2[0])
 
def interpp(src_img, att_img, inter_num, model_dir, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    sess = tf.Session()

    saver = tf.train.Saver()
    restoring(saver, sess)
    
    out = src_img[0]
    for i in range(1, inter_num + 1):
        lambda_i = i / float(inter_num)
        model.out_i = model.joiner('G_joiner', model.B, model.x * lambda_i) 
        out_i = sess.run(model.out_i, feed_dict={model.Ax: att_img, model.Be: src_img})
        out = np.concatenate((out, out_i[0]), axis=1)
    misc.imsave('interpp1.jpg', out)

MathewChanged = Model(is_train=False, nhwc=[1,64,64,3])
inputt = 'datasets/celebA/align_5p/000439.jpg'
targett = 'datasets/celebA/align_5p/000523.jpg'
model_dir = './train_log2/model'
'''
src_img = np.expand_dims(misc.imresize(misc.imread(inputt), (MathewChanged.height, MathewChanged.width)), axis=0)
att_img = np.expand_dims(misc.imresize(misc.imread(targett), (MathewChanged.height, MathewChanged.width)), axis=0)
swapper(src_img, att_img, model_dir, MathewChanged)
'''
src_img = np.expand_dims(misc.imresize(misc.imread(inputt), (MathewChanged.height, MathewChanged.width)), axis=0)
att_img = np.expand_dims(misc.imresize(misc.imread(targett), (MathewChanged.height, MathewChanged.width)), axis=0)
interpp(src_img, att_img, 5,  model_dir, MathewChanged)  
