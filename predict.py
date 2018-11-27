import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Siamese as siamese
from predict_util import *
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# image size
WIDTH = 96
HEIGHT = 96
CHANNELS = 3

batch_size = 32
# total number of images for valuation
size = 21150

save_frequency = 2
monitoring_rate = 5
# drop out
keep_prob = 1

NUM_EPOCHS = 1000
# number of images in each class
count = [0]*20000

def Predict(imgs, lbls, count, begin_index, index):
	'''
	Valuation

	Arguments:
		imgs: images list for valuation
		lbls: labels list for valuation
		count: for data_random
		begin_index: for data_random
		index: for data_random

	Return:
		None
	'''

	# GPU configuration
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
		# not necessary
		images = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])
		labels = tf.placeholder(tf.int64, shape = [None])
		# build Siamese network
		siamese_model = siamese.Siamese(size = 96)
		# global variables initialization
		sess.run(tf.global_variables_initializer())
		# restore the trained model
		saver = tf.train.Saver()
		saver.restore(sess, model_file)
		print("载入成功!")

		# record distance and accuracy per epoch
		accuracy_list = []
		distance_list = []
		# record margin distance of positive pairs and negative pairs
		distance_margin_list = []

		for epoch in range(NUM_EPOCHS):
			print("Start Epoch %i" % (epoch + 1))
			# data shuffle
			imgs_shuffle, lbls_shuffle = data_random(imgs,index,lbls,count, begin_index, batch_size, size)
			# count right prediction results
			correct_prediction = 0

			batch_index = 0
			# record distances in the embedding space
			neg_distance = 0
			pos_distance = 0
			neg_count = 0
			pos_count = 0
			pos_maxdis = 0.0
			neg_mindis = 1000.0
			# calculate number of iterations
			if(total_batch % 2 == 0):
				iter_time = (total_batch - 2) // 2
			else:
				iter_time = (total_batch - 2) // 2 + 1

			for i in range(iter_time):
				# batch sampling
				minibatch_X1, minibatch_Y1 = get_minibatch(imgs_shuffle, lbls_shuffle, batch_size, 2 * i, total_batch)
				minibatch_X2, minibatch_Y2 = get_minibatch(imgs_shuffle, lbls_shuffle, batch_size, 2 * i + 1, total_batch)
				minibatch_y = (minibatch_Y1 == minibatch_Y2).astype('float')
				r = sess.run([siamese_model.look_like], feed_dict = {siamese_model.x1: minibatch_X1, siamese_model.x2: minibatch_X2, siamese_model.keep_f: drop_out_rate})
				distance = r[0][0]
				same_list = r[0][1].astype('float')
				# correct predictions in one iteration
				countMax = np.sum(np.abs(same_list - minibatch_y) < 0.5)
				# sum up correct predictions
				correct_prediction += countMax

				if((batch_index % monitoring_rate == 0)):
					siamese_model.set_is_training(False)
					r = sess.run([siamese_model.look_like], feed_dict = {siamese_model.x1: minibatch_X1, siamese_model.x2: minibatch_X2, siamese_model.keep_f: drop_out_rate})
					distance = r[0][0]
					same_list = r[0][1].astype('float')
					neg_dis, pos_dis, neg_c, pos_c, min_neg_dis, max_pos_dis = tf_distance(distance, minibatch_y, batch_size)
					# margin of distance for positive paris
					if(max_pos_dis > pos_maxdis):
						pos_maxdis = max_pos_dis
					# margin of distance for negative pairs
					if(min_neg_dis < neg_mindis):
						neg_mindis = min_neg_dis
					neg_distance += neg_dis
					pos_distance += pos_dis
					neg_count += neg_c
					pos_count += pos_c
					print("Epoch %i Batch %i Pos_count %i Neg_count %i" %(epoch + 1,batch_index, pos_count, neg_count))
					print("Epoch %i Batch %i Pos_avg_distance %f Neg_avg_distance %f" %(epoch + 1,batch_index, pos_distance/(pos_count+1e-10), neg_distance/(neg_count+1e-10)))

				batch_index += 1

			# print information per epoch
			print("Val Accuracy: %f. Pos_avg_distance: %f. Neg_avg_distance: %f" % (correct_prediction / (iter_time * batch_size), pos_distance / (pos_count+1e-10), neg_distance / (neg_count+1e-10)))
			print("Pos_count: %i. Neg_count: %i" % (pos_count, neg_count))
			accuracy_list.append(correct_prediction / (iter_time * batch_size))
			distance_list.append(pos_distance / (pos_count+1e-10))
			distance_list.append(neg_distance / (neg_count+1e-10))
			distance_margin_list.append(pos_maxdis)
			distance_margin_list.append(neg_mindis)

			if((epoch + 1) % save_frequency == 0):
				fp1 = open('./predict_accuracy.txt', 'w')
				fp2 = open('./predict_distance.txt', 'w')
				fp3 = open('./predict_margin_distance.txt', 'w')
				for i in range(len(accuracy_list)):
					fp1.write("Predict accuracy: " + str(accuracy_list[i]) + '\n')
					fp2.write("Pos: " + str(distance_list[2 * i]) + "  Neg: " + str(distance_list[2*i+1]) + '\n')
					fp3.write("Pos_magin: " + str(distance_margin_list[2 * i]) + "  Neg_margin: " + str(distance_margin_list[2*i+1]) + '\n')
				fp1.close()
				fp2.close()
				fp3.close()


if __name__ == '__main__':
	imgs, labels, count, begin_index, index = get_data(size, 'val')
	model_file = './model/model'
	if size % batch_size == 0:
		total_batch = math.floor(size/batch_size)
	else:
		total_batch = math.floor(size/batch_size) + 1
	Predict(imgs, labels, count, begin_index, index)