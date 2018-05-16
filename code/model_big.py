from __future__ import print_function

import tensorflow as tf
import numpy as np
from model_base import ModelBase
from dataset import Dataset
import math

class Model(ModelBase):

	def getEncoder(self, input, nFeatures, zSize, sSize, zSample, sSample, nChannels, hInput, wInput, scope_name="Encoder"):
		with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
			x = input

			if hInput < 32 or wInput < 32:
				padh = 32 - hInput
				padw = 32 - wInput

				paddings = [[0, 0], [0, 0], [int(math.floor(padw/2)), int(math.ceil(padw/2))], [int(math.floor(padh/2)), int(math.ceil(padh/2))]]
				x = tf.pad(input, paddings)
				hInput = 32
				wInput = 32

			x = self.getConvBlock(x, nFeatures, 3, 1, name="Conv1")
			x = self.getConvBlock(x, nFeatures, 3, 1, name="Conv2")

			x = self.getConvBlock(x, nFeatures * 2, 2, 2, name="Conv3")
			x = self.getConvBlock(x, nFeatures * 2, 3, 1, name="Conv4")

			x = self.getConvBlock(x, nFeatures * 4, 2, 2, name="Conv5")
			x = self.getConvBlock(x, nFeatures * 4, 3, 1, name="Conv6")

			x = self.getConvBlock(x, nFeatures * 8, 2, 2, name="Conv7")
			x = self.getConvBlock(x, nFeatures * 8, 3, 1, name="Conv8")

			npix = nFeatures * 8 * hInput/8 * wInput/8
			x = tf.reshape(x, [-1, npix])

			if zSample:
				z = tf.layers.dense(x, zSize * 2)
				z = tf.reshape(z, [-1, 2, zSize])
			else:
				z = tf.layers.dense(x, zSize)

			if sSample:
				s = tf.layers.dense(x, sSize * 2)
				s = tf.reshape(s, [-1, 2, sSize])
			else:
				s = tf.layers.dense(x, sSize)

		return z, s


	def getDecoder(self, z, s, nFeatures, zSize, sSize, zSample, sSample, nChannels, hInput, wInput, scope_name="Decoder"):
		with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
			if zSample:
				x = {}
				x[0], x[1] = tf.split(z, [1,1], 1)
				
				x[0] = tf.reshape(x[0], [-1, zSize])
				x[1] = tf.reshape(x[1], [-1, zSize])

				eps = np.random.normal(0, 1, (int(x[0].get_shape()[0]), int(x[0].get_shape()[1])))
				z = tf.add(tf.multiply(tf.exp(tf.multiply(x[1], 0.5)), eps), x[0])
			
			if sSample:
				x = {}
				x[0], x[1] = tf.split(s, [1,1], 1)
				
				x[0] = tf.reshape(x[0], [-1, sSize])
				x[1] = tf.reshape(x[1], [-1, sSize])

				eps = np.random.normal(0, 1, (int(x[0].get_shape()[0]), int(x[0].get_shape()[1])))
				s = tf.add(tf.multiply(tf.exp(tf.multiply(x[1], 0.5)), eps), x[0])
			
			npix = nFeatures * 8 * 4 * 4

			z = tf.layers.dense(z, npix)
			s = tf.layers.dense(s, npix)

			x = tf.add(z,s)
			x = tf.reshape(x, [-1, nFeatures*8, 4, 4])

			x = self.getConvBlock(x, nFeatures*8, 3, 1, name="Conv1")
			
			x = self.getDeConvBlock(x, nFeatures*4, 2, 2, name="Conv2")
			x = self.getConvBlock(x, nFeatures*4, 3, 1, name="Conv3")

			x = self.getDeConvBlock(x, nFeatures*2, 2, 2, name="Conv4")
			x = self.getConvBlock(x, nFeatures*2, 3, 1, name="Conv5")

			x = self.getDeConvBlock(x, nFeatures, 2, 2, name="Conv6")
			x = self.getConvBlock(x, nFeatures, 3, 1, name="Conv7")
			x = self.getConv(x, nChannels, 3, 1, name="Conv8")

			x = tf.nn.tanh(x)

		return x


	def getDiscriminator(self, inputpix, inputsid, nFeatures, nSid, nChannels, hInput, wInput, scope_name="Discriminator"):
		with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
			x = inputpix
			sid = inputsid

			if hInput < 32 or wInput < 32:
					padh = 32 - hInput
					padw = 32 - wInput

					paddings = [[0, 0], [0, 0], [int(math.floor(padw/2)), int(math.ceil(padw/2))], [int(math.floor(padh/2)), int(math.ceil(padh/2))]]
					x = tf.pad(x, paddings)
					hInput = 32
					wInput = 32

			x = self.getConv(x, nFeatures, 3, 1, name="Conv1")
			x = tf.nn.leaky_relu(x)

			x = self.getConv(x, nFeatures, 2, 2, name="Conv2")
			
			#Lookup Table
			lookup_table1 = tf.get_variable("Lookup1", [nSid, nFeatures*hInput*wInput/8/8], dtype=tf.float32)
			h = tf.nn.embedding_lookup(lookup_table1, sid)
			h = tf.reshape(h, [-1, nFeatures, hInput/8, wInput/8])
			h = tf.image.resize_nearest_neighbor(tf.transpose(h, [0, 2, 3, 1]), (4 * int(h.get_shape()[2]), 4 * int(h.get_shape()[3])))
			h = tf.transpose(h, [0, 3, 1, 2])

			#x = tf.concat([x, h], 0)
			x = x + h
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)

			x = self.getConv(x, nFeatures*2, 2, 2, name="Conv3")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)
			x = self.getConv(x, nFeatures*2, 3, 1, name="Conv4")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)


			#Lookup Table
			lookup_table2 = tf.get_variable("Lookup2", [nSid, 2*nFeatures*hInput*wInput/8/8], dtype=tf.float32)
			h = tf.nn.embedding_lookup(lookup_table2, sid)
			h = tf.reshape(h, [-1, 2*nFeatures, hInput/8, wInput/8])
			h = tf.image.resize_nearest_neighbor(tf.transpose(h, [0, 2, 3, 1]), (2 * int(h.get_shape()[2]), 2 * int(h.get_shape()[3])))
			h = tf.transpose(h, [0, 3, 1, 2])

			x = x + h
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)


			x = self.getConv(x, nFeatures*4, 2, 2, name="Conv5")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)
			x = self.getConv(x, nFeatures*4, 3, 1, name="Conv6")


			#Lookup Table
			lookup_table3 = tf.get_variable("Lookup3", [nSid, 4*nFeatures*hInput*wInput/8/8], dtype=tf.float32)
			h = tf.nn.embedding_lookup(lookup_table3, sid)
			h = tf.reshape(h, [-1, 4*nFeatures, hInput/8, wInput/8])
			
			x = x + h
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)


			x = self.getConv(x, nFeatures*4, 3, 1, name="Conv7")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)

			npix = nFeatures * 4 * hInput/8 * wInput/8
			x = tf.reshape(x, [-1, npix])
			x = tf.nn.dropout(x, keep_prob = 0.5)
			x = tf.layers.dense(x, 1)
			x = tf.nn.sigmoid(x)

		return x


	def getDiscriminatorMod(self, inputpix, inputpix2, nFeatures, nSid, nChannels, hInput, wInput, scope_name="Discriminator"):
		with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
			x = inputpix
			h = inputpix2

			if hInput < 32 or wInput < 32:
					padh = 32 - hInput
					padw = 32 - wInput

					paddings = [[0, 0], [0, 0], [int(math.floor(padw/2)), int(math.ceil(padw/2))], [int(math.floor(padh/2)), int(math.ceil(padh/2))]]
					x = tf.pad(x, paddings)
					hInput = 32
					wInput = 32

			x = self.getConv(x, nFeatures, 3, 1, name="Conv1")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)

			h = self.getConv(h, nFeatures, 3, 1, name="Conv2")
			h = tf.layers.batch_normalization(h)
			h = tf.nn.leaky_relu(h)
			
			x = tf.concat([x, h], 1)
			
			x = self.getConv(x, nFeatures*2, 2, 2, name="Conv3")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)
			x = self.getConv(x, nFeatures*2, 3, 1, name="Conv4")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)


			x = self.getConv(x, nFeatures*4, 2, 2, name="Conv5")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)
			x = self.getConv(x, nFeatures*4, 3, 1, name="Conv7")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)

	
			npix = nFeatures * 4 * hInput/4 * wInput/4
			x = tf.reshape(x, [-1, npix])
			x = tf.nn.dropout(x, keep_prob = 0.5)
			x = tf.layers.dense(x, 1)
			x = tf.nn.sigmoid(x)

		return x



	def getDiscriminatorVGG(self, inputpix, inputpix2, nFeatures, nSid, nChannels, hInput, wInput, scope_name="Discriminator"):
		with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
			x = inputpix
			h = inputpix2

			if hInput < 32 or wInput < 32:
					padh = 32 - hInput
					padw = 32 - wInput

					paddings = [[0, 0], [0, 0], [int(math.floor(padw/2)), int(math.ceil(padw/2))], [int(math.floor(padh/2)), int(math.ceil(padh/2))]]
					x = tf.pad(x, paddings)
					hInput = 32
					wInput = 32

			x = tf.concat([x, h], 1)

			x = self.getConv(x, 64, 3, 1, name="Conv1")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)
			x = self.getConv(x, 64, 3, 2, name="Conv2")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)

			x = self.getConv(x, 128, 3, 1, name="Conv3")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)
			x = self.getConv(x, 128, 3, 2, name="Conv4")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)

			x = self.getConv(x, 256, 3, 1, name="Conv5")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)
			x = self.getConv(x, 256, 3, 2, name="Conv6")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)

			x = self.getConv(x, 512, 3, 1, name="Conv7")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)
			x = self.getConv(x, 512, 3, 2, name="Conv8")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)

			x = self.getConv(x, 512, 3, 1, name="Conv9")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)
			x = self.getConv(x, 512, 3, 2, name="Conv10")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)
	
			npix = 512
			x = tf.reshape(x, [-1, npix])
			x = tf.layers.dense(x, 4096)
			x = tf.nn.dropout(x, keep_prob = 0.5)
			x = tf.layers.dense(x, 1000)
			x = tf.nn.dropout(x, keep_prob = 0.5)
			x = tf.layers.dense(x, 1)
			x = tf.nn.sigmoid(x)

		return x



	def getDiscriminatorSimple(self, inputpix, nFeatures, nSid, nChannels, hInput, wInput, scope_name="Discriminator"):
		with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
			x = inputpix

			if hInput < 32 or wInput < 32:
					padh = 32 - hInput
					padw = 32 - wInput

					paddings = [[0, 0], [0, 0], [int(math.floor(padw/2)), int(math.ceil(padw/2))], [int(math.floor(padh/2)), int(math.ceil(padh/2))]]
					x = tf.pad(x, paddings)
					hInput = 32
					wInput = 32

			x = self.getConv(x, nFeatures, 2, 2, name="Conv1a")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)
			x = self.getConv(x, nFeatures, 3, 1, name="Conv1b")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)

			x = self.getConv(x, nFeatures*2, 2, 2, name="Conv3")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)
			x = self.getConv(x, nFeatures*2, 3, 1, name="Conv4")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)


			x = self.getConv(x, nFeatures*4, 2, 2, name="Conv5")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)
			x = self.getConv(x, nFeatures*4, 3, 1, name="Conv7")
			x = tf.layers.batch_normalization(x)
			x = tf.nn.leaky_relu(x)

			npix = nFeatures * 4 * hInput/8 * wInput/8
			x = tf.reshape(x, [-1, npix])
			x = tf.nn.dropout(x, keep_prob = 0.5)
			x = tf.layers.dense(x, 1)
			x = tf.nn.sigmoid(x)

		return x


	def classifier(self, input, sSize, nClasses, sSample, nHidden, dropout=0.5):
		with tf.variable_scope("Classifier", reuse=tf.AUTO_REUSE) as scope:
			input = tf.layers.dense(input, nHidden)
			input = tf.nn.relu(input)
			input = tf.nn.dropout(input, keep_prob=dropout)

			input = tf.layers.dense(input, nClasses)
			input = tf.nn.softmax(input)

		return input




			
"""
   local x = conv(nFeatures*4, nFeatures*4, 3, 1)(x)
   local x = nn.SpatialBatchNormalization(nFeatures*4):cuda()(x)
   local x = nn.LeakyReLU(0.2)(x)
   local npix = nFeatures*4*hInput*wInput/8/8
   local x = nn.View(npix):setNumInputDims(3)(x)
   local x = nn.Dropout()(x)
   local x = nn.Linear(npix, 1):cuda()(x)
   local x = nn.Sigmoid()(x)
   return nn.gModule({inputpix, inputsid}, {x}):cuda()
end
"""


		


def main():
	x = Model()
	"""
	input = tf.placeholder(shape=[48, 1, 28, 28], dtype=tf.float32)
	z, s = x.getEncoder(input, 16, 512, 64, True, False, 1, 28, 28)

	print(z)
	print(s)
	
	img = x.getDecoder(z, s, 16, 512, 64, True, False, 1, 28, 28)
	data = Dataset()
	inp, sid = data.getBatch(16, data_type="train")
	
	print(inp.shape)
	print(sid.shape)
	sid = np.reshape(sid, (-1))
	inp = np.reshape(inp, (-1, 1, 28, 28))
	print(inp.shape)
	print(sid.shape)
	

	"""
	inputs = tf.placeholder(shape=[32, 1, 28, 28], dtype=tf.float32)
	sids = tf.placeholder(shape=[32], dtype=tf.int32)
	x.getDiscriminatorVGG(inputs, inputs, 16, 10, 1, 32, 32)






if __name__ == '__main__':
	main()


