import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import Siamese as siamese
from skimage import transform
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# image size
WIDTH = 96
HEIGHT = 96
CHANNELS = 3
# number of paris(two faces per pair)
size = 6000
batch_size = 32
# drop out
keep_prob = 1.0


def get_data(size):
	'''
	Read in 'size' pairs of images and labels in the test set

	Arguments:
		size: number of paris
	
	Return:
		img1_list: image list of the first person
		img2_list: image list of the second person
	'''

	img_dir = '/home/guhanxue/jupyter/root/demo/tensorflow-triplet-loss/data/face/'
	f = open(img_dir + "test_lst.csv","r")
	img1_list = np.zeros((size,96, 96, 3), dtype = 'float')
	img2_list = np.zeros((size,96, 96, 3), dtype = 'float')
	i = 0
	for k in range(size):
		line = f.readline().strip('\n')
		face1, face2 = line.split(' ')
		face1_path = img_dir + 'data/' + face1
		face2_path = img_dir + 'data/' + face2
		img1 = plt.imread(face1_path).astype('uint8')
		img2 = plt.imread(face2_path).astype('uint8')
		# scale images to fit the network input
		img1_list[i] = transform.rescale(img1, [0.857,1])
		img2_list[i] = transform.rescale(img2, [0.857,1])
		i += 1

	f.close()

	return img1_list, img2_list


def Test(img1_list, img2_list, size):
	'''
	Test model performance on the test set

	Arguments:
		img1_list: image list of the first person
		img2_list: image list of the second person
		size: number of image pairs

	Return:
		labels: list of preditions
		distances: distances in the embedding space of each pair
	'''

	# GPU configuration
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	
	labels = []
	distances = []
	if(size % batch_size == 0):
		total_batch = math.floor(size / batch_size + 1e-5)
	else:
		total_batch = math.floor(size / batch_size) + 1

	with tf.Session(config=config) as sess:
		# build Siamese network
		siamese_model = siamese.Siamese(size = 96)
		# initialize global variables
		sess.run(tf.global_variables_initializer())
		# restore the model
		saver = tf.train.Saver()
		if os.path.isfile(model_file + '.meta'):
			saver.restore(sess, model_file)
			print('模型重载成功')
		# Positive pairs count in prediction
		countPos = 0
		for i in range(total_batch):
			minibatch_face1, minibatch_face2 = get_minibatch(img1_list, img2_list, batch_size, i, total_batch)
			siamese_model.set_is_training(False)
			r = sess.run([siamese_model.look_like], feed_dict = {siamese_model.x1: minibatch_face1, siamese_model.x2: minibatch_face2, siamese_model.keep_f: keep_prob})
			distance = r[0][0]
			same_list = r[0][1]
			countPos += np.sum(same_list)
			print(same_list)
			for j in range(len(same_list)):
				distances.append(distance[j])
				if(same_list[j] < 0.5):
					labels.append(0)
				else:
					labels.append(1)

		# print count of positive pairs, negative pairs and pos rate
		print("Pos: %i. Neg: %i. Pos rate: %f" % (countPos, size - countPos, countPos / size))

	return labels, distances


def get_minibatch(img1_list, img2_list, batch_size, now_batch, total_batch):
	'''
	Get minibatch of the test set

	Arguments:
		img1_list: images list of the first person
		img2_list: images list of the second person
		batch_size: size fo per batch
		now_batch: index of the current batch
		total_batch: index upper bound of batch

	Return:
		image1_batch: image batch for the first person
		image2_batch: image batch for the second person
	'''

	if(now_batch < total_batch - 1):
		image1_batch = img1_list[now_batch*batch_size:(now_batch+1)*batch_size]
		image2_batch = img2_list[now_batch*batch_size:(now_batch+1)*batch_size]
	else:
		image1_batch = img1_list[now_batch*batch_size:]
		image2_batch = img2_list[now_batch*batch_size:]

	return image1_batch, image2_batch


if __name__ == '__main__':
	img1_list, img2_list = get_data(size)
	# path of the trained model
	model_file = './model/model'
	labels, distances = Test(img1_list, img2_list, size)

	with open("./result.csv", "w") as f:
		for i in range(len(labels)):
			f.write(str(labels[i]) + '\n')

	with open("./distances.csv", "w") as f:
		for i in range(len(labels)):
			f.write(str(distances[i]) + '\n')