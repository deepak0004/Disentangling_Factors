import tensorflow as tf
import numpy as np

class ModelBase:

	def getConvBlock(self, input, nOutput, k, d, name="XYZ"):
		with tf.name_scope(name) as scope:
			x = tf.layers.conv2d(input, nOutput, [k, k], strides=(d, d), padding='SAME', data_format='channels_first')
			x = tf.layers.batch_normalization(x)
			x = tf.nn.relu(x)
		return x

	def getConv(self, input, nOutput, k, d, name="XYZ"):
		x = tf.layers.conv2d(input, nOutput, [k, k], strides=(d, d), padding='SAME', data_format='channels_first')
		return x

	def getDeConvBlock(self, input, nOutput, k, d, name="XYZ"):
		with tf.name_scope(name) as scope:
			x = tf.layers.conv2d_transpose(input, nOutput, [k, k], strides=[d, d], padding='VALID', data_format='channels_first')
			x = tf.layers.batch_normalization(x)
			x = tf.nn.relu(x)
		return x