from __future__ import print_function

import tensorflow as tf
import numpy as np
from model_big import Model
from dataset import Dataset
from scipy.misc import toimage
import math
import pickle

class Train:

	def __init__(self, model, dataset):
		self.dataset = dataset

		self.no_epoch = 100
		self.no_iters = 500
		self.batch_size = 16
		self.weights_dir = "weights/model.ckpt"
		self.test_dir = "weights/model.ckpt-84"
		
		self.reuse_weights = False
		self.retrain_dir = "weights/model.ckpt-73"

		self.kSize = 4
		self.sSize = 64
		self.zSize = 512
		self.nFeatures = 16
		self.classifierNHiddenUnits = 256

		self.lr_dis = 0.01
		self.lr_gen = 0.01

		self.hInput = 28
		self.wInput = 28
		self.nChannels = 1
		self.nSid = 10

		self.zSample = True
		self.sSample = False

		self.labelReal = 0.95
		self.labelFake = 0
		self.recweight = 0.5
		self.swap1weight = 1
		self.swap2weight = 0.01
		self.classweight = 1
		self.klweightZ = 0.1
		self.klweightS = 0.0
		self.samplingweight = 0

		self.getPlaceholders()

		self.z, self.s = model.getEncoder(self.images, self.nFeatures, self.zSize, self.sSize, self.zSample, self.sSample, self.nChannels, self.hInput, self.wInput)

		self.logits = model.classifier(self.s, self.sSize, self.nSid, self.sSample, self.classifierNHiddenUnits)
		self.loss_classifier = self.classweight * tf.losses.sparse_softmax_cross_entropy(labels=self.sids, logits=self.logits)
		

		#Seperated z values
		self.z1 = self.z[:self.batch_size,:,:]
		self.z2 = self.z[self.batch_size:self.batch_size*2,:,:]
		self.z3 = self.z[self.batch_size*2:self.batch_size*3,:,:]

		self.z12 = tf.concat([self.z1, self.z2], axis=0)
		self.z32 = tf.concat([self.z3, self.z2], axis=0)

		self.z_12_32 = tf.concat([self.z, self.z12, self.z32], axis=0)

		self.sample_z_12_32 = self.getSample(self.z_12_32, (self.batch_size*7, self.zSize))
		self.sample_z = self.getSample(self.z, (self.batch_size*3, self.zSize))

		
		self.s1 = self.s[:self.batch_size, :]
		self.s2 = self.s[self.batch_size:self.batch_size*2, :]
		self.s3 = self.s[self.batch_size*2:self.batch_size*3, :]

		self.s21 = tf.concat([self.s2, self.s1], axis=0)
		self.s23 = tf.concat([self.s2, self.s3], axis=0)

		self.s_21_23 = tf.concat([self.s, self.s21, self.s23], axis=0)

		self.y_rec_swap1_swap2 = model.getDecoder(self.sample_z_12_32, self.s_21_23, self.nFeatures, self.zSize, self.sSize, False, self.sSample, self.nChannels, self.hInput, self.wInput)
		
		self.y_rec = self.y_rec_swap1_swap2[:self.batch_size*3,:,:,:]
		self.y_rec_23 = self.y_rec_swap1_swap2[self.batch_size:self.batch_size*3,:,:,:]
		self.y_swap1 = self.y_rec_swap1_swap2[self.batch_size*3:self.batch_size*5,:,:,:]
		self.y_swap2 = self.y_rec_swap1_swap2[self.batch_size*5:self.batch_size*7,:,:,:]



		images = self.padInput(self.images, self.hInput, self.wInput)
		images1 = images[:self.batch_size, :]
		images2 = images[self.batch_size:self.batch_size*2, :]
		images3 = images[self.batch_size*2:self.batch_size*3, :]

		images12 = tf.concat([images1, images2], axis=0)
		images23 = tf.concat([images2, images3], axis=0)

		images_full = tf.concat([images, images12, images23], axis=0)

		#Recreational MSE Loss
		self.loss_rec1 = self.recweight * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.y_rec, images), 1))
		self.loss_rec2 = self.swap1weight * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.y_swap1, images12), 1))
		self.loss_gen = self.loss_rec1
		self.loss_gen += self.loss_rec2

		sids_real = self.sids_real#tf.one_hot(tf.reshape(self.sids_real, [-1]), depth=10)
		sids_fake = self.sids_fake#tf.one_hot(tf.reshape(self.sids_fake, [-1]), depth=10)


		#Adversial Loss
		self.sid23 = self.sids[self.batch_size:self.batch_size*3]
		self.dis_yswap2 = model.getDiscriminator(self.y_swap2, self.sid23, self.nFeatures, self.nSid, self.nChannels, self.hInput + 4, self.wInput + 4)
		#self.loss_gen += self.swap2weight * tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.sids_real, logits=self.dis_yswap2))
		self.loss_gen += self.swap2weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=sids_real, logits=self.dis_yswap2))

		


		self.z_swap2_normal = tf.random_normal(shape=[self.batch_size*2, 512])
		self.y_swap2_normal = model.getDecoder(self.z_swap2_normal, self.s23, self.nFeatures, self.zSize, self.sSize, False, self.sSample, self.nChannels, self.hInput, self.wInput)
		self.dis_y_normal = model.getDiscriminator(self.y_swap2_normal, self.sid23, self.nFeatures, self.nSid, self.nChannels, self.hInput + 4, self.wInput + 4)
		#self.loss_gen += self.samplingweight * tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.sids_real, logits=self.dis_y_normal))
		self.loss_gen += self.samplingweight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=sids_real, logits=self.dis_y_normal))


		#KL Divergence Loss
		z = tf.unstack(self.z, axis=1)
		z_mu = z[0]
		z_sigma = z[1]
		self.loss_z1 = tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + tf.log(tf.square(z_sigma)) - tf.square(z_mu) - tf.exp(tf.log(tf.square(z_sigma))), 1))
		klweightZ = self.klweightZ*6/(self.nChannels*self.hInput*self.wInput)
		self.loss_gen += klweightZ * self.loss_z1
		

		
		#Discriminator Training
		self.dis_y_rec_23 = model.getDiscriminator(self.y_rec_23, self.sid23, self.nFeatures, self.nSid, self.nChannels, self.hInput + 4, self.wInput + 4)
		#self.loss_discriminator = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.sids_real, logits=self.dis_y_rec_23))
		#self.loss_discriminator += tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.sids_fake, logits=self.dis_yswap2))
		self.loss_discriminator = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=sids_real, logits=self.dis_y_rec_23))
		self.loss_discriminator += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=sids_fake, logits=self.dis_yswap2))
		

		
		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if 'Discriminator_Real' in var.name or 'Discriminator_Fake' in var.name or 'Discriminator' in var.name]
		self.g_vars = [var for var in t_vars if 'Decoder' in var.name or 'Encoder' in var.name]
		
		self.opt_classifier = tf.train.AdamOptimizer(learning_rate=self.lr_gen).minimize(self.loss_classifier)
		self.opt_rec = tf.train.AdamOptimizer(learning_rate=self.lr_gen).minimize(self.loss_gen, var_list=self.g_vars)
		self.opt_adv = tf.train.GradientDescentOptimizer(learning_rate=self.lr_dis).minimize(self.loss_discriminator, var_list=self.d_vars)


	def padInput(self, input, hInput, wInput):
		if hInput < 32 or wInput < 32:
			padh = 32 - hInput
			padw = 32 - wInput

			paddings = [[0, 0], [0, 0], [int(math.floor(padw/2)), int(math.ceil(padw/2))], [int(math.floor(padh/2)), int(math.ceil(padh/2))]]
			input = tf.pad(input, paddings)
		return input


	def padInputNumpy(self, input, hInput, wInput):
		if hInput < 32 or wInput < 32:
			padh = 32 - hInput
			padw = 32 - wInput

			paddings = [[0, 0], [0, 0], [int(math.floor(padw/2)), int(math.ceil(padw/2))], [int(math.floor(padh/2)), int(math.ceil(padh/2))]]
			input = np.pad(input, paddings, mode="constant")
		return input


	



	def getSample2(self, z):
		z = tf.unstack(self.z, axis=1)
		mu = z[0]
		sigma = z[1]
		"""
		mu = tf.reshape(self.z[:,0,:], [112, 512])
		sigma = tf.reshape(self.z[:,1,:], [112, 512])
		"""
		z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
		return z


	def getSample(self, z, shape, size=512):
		x = {}
		x[0], x[1] = tf.split(z, [1,1], 1)

		x[0] = tf.reshape(x[0], [-1, size])
		x[1] = tf.reshape(x[1], [-1, size])

		#eps = np.random.normal(0, 1, (112, 512))
		eps = tf.random_normal( shape, mean=0.0, stddev=1.0)
		z = tf.add(tf.multiply(tf.exp(tf.multiply(x[1], 0.5)), eps), x[0])
		return z



	def getPlaceholders(self):
		self.images = tf.placeholder(shape=[None, self.nChannels, self.hInput, self.wInput], dtype=tf.float32)
		self.sids = tf.placeholder(shape=[None, 1], dtype=tf.int32)

		self.sids_real = tf.placeholder(shape=[None, 1], dtype=tf.float32)
		self.sids_fake = tf.placeholder(shape=[None, 1], dtype=tf.float32)


	def startTraining(self):
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True

		initialize = tf.global_variables_initializer()

		with tf.Session(config = self.config) as sess:
			self.saver = tf.train.Saver(max_to_keep=1)

			sess.run(initialize)

			if self.reuse_weights:
				self.saver.restore(sess, self.retrain_dir)

			if not self.reuse_weights:
				print("Training Classifier")
				for iters in xrange(0):
					x, y = self.dataset.getBatch(self.batch_size)
					f_dict = {self.images: x, self.sids: y}
					_, loss = sess.run([self.opt_classifier, self.loss_classifier], feed_dict=f_dict)
					print("Iters: {}, Classification Loss: {}".format(iters, loss))


			for epoch in xrange(self.no_epoch):
				for iters in xrange(self.no_iters):
					x, y = self.dataset.getBatch(self.batch_size)
					#x1, y1, x2, y2 = self.dataset.getBatchDiscriminator(self.batch_size)

					y = np.reshape(y, (-1, 1))
					#y1 = np.reshape(y1, (-1, 1))
					#y2 = np.reshape(y2, (-1, 1))

					f_dict = {self.images: x, self.sids: y, self.sids_real: np.full((self.batch_size*2, 1), self.labelReal), self.sids_fake: np.full((self.batch_size*2, 1), self.labelFake)}
					sess.run(self.opt_rec, feed_dict = f_dict)
					sess.run(self.opt_adv, feed_dict = f_dict)
					sess.run(self.opt_adv, feed_dict = f_dict)
					
					#sess.run([self.opt_rec], feed_dict = f_dict)
					
					#loss1, loss2, loss3, loss4, loss5 = sess.run([self.loss_rec, self.loss_dis1, self.loss_dis2, self.loss_pos, self.loss_neg], feed_dict = f_dict)

					#print("Epoch: {}, Iters: {}, Rec_Loss: {}, Discriminator_Loss1: {}, Discriminator_Loss2: {}, Adv_Pos_Loss: {}, Adv_Neg_Loss: {}".format(epoch, iters, loss1, loss2, loss3, loss4, loss5))

					loss1, loss2 = sess.run([self.loss_gen, self.loss_discriminator], feed_dict = f_dict)
					if loss1 < 0.01:
						self.saver.save(sess, self.weights_dir, global_step = epoch)
						exit()

					print("Epoch: {}, Iters: {}, Rec_Loss: {}, Discriminator_Loss: {}".format(epoch, iters, loss1, loss2))

				x, y = self.dataset.getBatch(self.batch_size, data_type="train")
				y = np.reshape(y, (-1, 1))
				f_dict = {self.images: x, self.sids: y}
				y_swap1, y_swap2, y_rec = sess.run([self.y_swap1, self.y_swap2, self.y_rec], feed_dict = f_dict)

				y1 = y[:self.batch_size, :]
				y2 = y[self.batch_size:self.batch_size*2, :]
				y3 = y[self.batch_size*2:self.batch_size*3, :]

				y12 = np.concatenate((y1, y2), axis=0)
				y23 = np.concatenate((y2, y3), axis=0)
				y = np.concatenate((y, y12, y23), axis=0)

				#self.dump_latent_space(z, s, y)
				x = self.padInputNumpy(x, self.hInput, self.wInput)

				self.draw(x[0,0,:,:], "y_org1.png")
				self.draw(x[self.batch_size*1,0,:,:], "y_org2.png")
				self.draw(x[self.batch_size*2,0,:,:], "y_org3.png")

				self.draw(x[0,0,:,:], "y_rec1.png")
				self.draw(x[self.batch_size*1,0,:,:], "y_rec2.png")
				self.draw(x[self.batch_size*2,0,:,:], "y_rec3.png")

				self.draw(y_swap1[0,0,:,:], "y_swap11.png")
				self.draw(y_swap1[self.batch_size,0,:,:], "y_swap12.png")

				self.draw(y_swap2[0,0,:,:], "y_swap21.png")
				self.draw(y_swap2[self.batch_size,0,:,:], "y_swap22.png")

				self.saver.save(sess, self.weights_dir, global_step = epoch)


	def startTesting(self):
		import os
		#os.environ["CUDA_VISIBLE_DEVICES"] = ""
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True

		initialize = tf.global_variables_initializer()

		with tf.Session(config = self.config) as sess:
			self.saver = tf.train.Saver(max_to_keep=1)
			sess.run(initialize)

			self.saver.restore(sess, self.test_dir)

			x, y = self.dataset.getBatch(self.batch_size, data_type="train")
			
			y = np.reshape(y, (-1, 1))
			
			f_dict = {self.images: x, self.sids: y}
			
			y_swap1, y_swap2, y_rec = sess.run([self.y_swap1, self.y_swap2, self.y_rec], feed_dict = f_dict)

			z, s, yy = sess.run([self.sample_z, self.s, self.sids], feed_dict=f_dict)

			y1 = y[:self.batch_size, :]
			y2 = y[self.batch_size:self.batch_size*2, :]
			y3 = y[self.batch_size*2:self.batch_size*3, :]

			y12 = np.concatenate((y1, y2), axis=0)
			y23 = np.concatenate((y2, y3), axis=0)
			y = np.concatenate((y, y12, y23), axis=0)

			
			#self.dump_latent_space(z, s, yy)
			x = self.padInputNumpy(x, self.hInput, self.wInput)

			abc=[]
			abc.append(x[0:self.batch_size,0,:,:])
			abc.append(x[self.batch_size:self.batch_size*2,0,:,:])
			abc.append(x[self.batch_size*2:self.batch_size*3,0,:,:])

			abc.append(y_rec[0:self.batch_size,0,:,:])
			abc.append(y_rec[self.batch_size:self.batch_size*2,0,:,:])
			abc.append(y_rec[self.batch_size*2:self.batch_size*3,0,:,:])

			abc.append(y_swap1[0:self.batch_size,0,:,:])
			abc.append(y_swap1[self.batch_size:self.batch_size*2,0,:,:])

			abc.append(y_swap2[0:self.batch_size,0,:,:])
			abc.append(y_swap2[self.batch_size:self.batch_size*2,0,:,:])

			self.show_images(abc)


	def show_images(self, images, cols = 16, rows = 5):
		n_images = rows * cols

		fig = plt.figure()
		count = 1

		padh = 4
		padw = 4
		paddings = [[int(math.floor(padw/2)), int(math.ceil(padw/2))], [int(math.floor(padh/2)), int(math.ceil(padh/2))]]

		for n in xrange(10):
			if n == 6 or n == 7 or n==5 or n==4 or n==3:
				continue

			for i in xrange(16):
				a = fig.add_subplot(5, 16, count)
				count += 1
				plt.gray()

				plt.imshow(images[n][i,:, :])
				plt.axis('off')

		fig.savefig("abc.png", pad_inches=-1)


	def dump_latent_space(self, z, s, y):
		pickle.dump(z, open("latent_space_dumps/label_space_z.dump", "w"))
		pickle.dump(s, open("latent_space_dumps/label_space_s.dump", "w"))
		pickle.dump(y, open("latent_space_dumps/labels.dump", "w"))


				
	def draw(self, data, name="test.png"):
		temp = np.reshape(data, (32,32))
		img = toimage(temp)
		img.save('image_dumps_vae/' + name)
		print('image_dumps_vae/' + name)


def main():
	model = Model()
	dataset = Dataset()
	train = Train(model, dataset)
	#train.startTraining()
	train.startTesting()

if __name__ == '__main__':
	main()