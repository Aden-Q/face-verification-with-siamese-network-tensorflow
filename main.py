from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import input

import tensorflow as tf
from functools import reduce
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import Siamese as siamese
from util import *

#import system things
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

batch_size = 32

# image size
WIDTH = 96
HEIGHT = 96
CHANNELS = 3
#"mini batch size"
MINI_BATCH_SIZE = batch_size
# initial learning rate
learning_rate_orig = 1e-04
# tensorboard settings
tensorboard_on = True
TensorBoard_refresh = 50
monitoring_rate = 5
NUM_EPOCHS = 1000
save_frequency = 1
size = 433041
# number of pictures in each class
count = [0]*20000
# drop out
keep_prob = 1.0

def Train(imgs, lbls, count, begin_index, index):
    '''
    Train on siamese network

    Arguments:
        imgs: images list readed before training
        lbls: labels list readed before training
        count: list rep number of pictures of each class
        begin_index: for data_random
        index: for data_random
    
    Return:
        cost_list: list of cost per epoch
    '''
    
    # num of total training image
    num_train_image = size
    # GPU configurations
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True 
    
    with tf.Session(config=config) as sess:
        # not necessary
        images = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])
        # build resnet model, image size 96*96*3
        siamese_model = siamese.Siamese(size = 96)
        siamese_model.network(images, keep_prob)
        # number of batches per epoch
        num_minibatches = int(num_train_image / MINI_BATCH_SIZE)

        # cost function
        learning_rate = learning_rate_orig
        with tf.name_scope("cost"):
            cost = tf.reduce_sum(siamese_model.loss)
        with tf.name_scope("train"):
            # learning rate decay
            global_steps = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(learning_rate_orig, global_steps, num_minibatches * 40, 0.1, staircase = True)
            # train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
            train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            # train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)

        sess.run(tf.global_variables_initializer())
        # model loader or saver
        saver = tf.train.Saver()

        # pre-training or not
        new = True
        if os.path.isfile(model_file + '.meta'):
            input_var = input("我们发现模型，是否需要预训练 [yes/no]?")
            if input_var == 'yes':
                new = False
        if not new:
            saver.restore(sess, model_file)
            print('模型重载成功')

        # store total cost, accuracy and average distance of pos/neg pair in each epoch
        cost_list = []
        accuracy_list = []
        distance_list = []

        for epoch in range(NUM_EPOCHS):
            print("Start Epoch %i" % (epoch + 1))
            # data shuffle
            imgs_shuffle, lbls_shuffle = data_random(imgs,index,lbls,count, begin_index, batch_size, size)
            # record cost per batch
            minibatch_cost = 0.0
            # record cost per 10 batches
            ten_batch_cost = 0.0
            # count the number of batch
            batch_index = 0  
            # for accuracy prediction
            correct_prediction = 0
            # pos/neg pair distance and counts in the embedding space
            neg_distance = 0
            pos_distance = 0
            neg_count = 0
            pos_count = 0

            # number of iterations
            if(total_batch % 2 == 0):
                iter_time = (total_batch - 2) // 2
            else:
                iter_time = (total_batch - 2) // 2 + 1

            for i in range(iter_time):
                # feed two batches into the network at the same time
                minibatch_X1, minibatch_Y1 = get_minibatch(imgs_shuffle, lbls_shuffle, batch_size, 2 * i, total_batch)
                minibatch_X2, minibatch_Y2 = get_minibatch(imgs_shuffle, lbls_shuffle, batch_size, 2 * i + 1, total_batch)
                minibatch_y = (minibatch_Y1 == minibatch_Y2).astype('float')

                # change learning rate
                sess.run(global_steps.assign(epoch * num_minibatches + batch_index))

                # record examples to monitoring the training process
                if((batch_index % monitoring_rate == 0)):
                    siamese_model.set_is_training(False)
                    r = sess.run([siamese_model.look_like], feed_dict = {siamese_model.x1: minibatch_X1, siamese_model.x2: minibatch_X2, siamese_model.keep_f: drop_out_rate})
                    distance = r[0][0]
                    same_list = r[0][1].astype('float')
                    # count correct predictions
                    countMax = np.sum(np.abs(same_list - minibatch_y) < 0.5)
                    print("Epoch %i Batch %i Before Optimization Count %i" %(epoch + 1,batch_index, countMax))
 
                # Training and calculating cost
                siamese_model.set_is_training(True)
                temp_cost,_ = sess.run([cost, train], feed_dict={siamese_model.x1: minibatch_X1, siamese_model.x2: minibatch_X2, siamese_model.y_: minibatch_y.reshape(minibatch_y.shape[0]), siamese_model.keep_f: drop_out_rate})
                minibatch_cost += np.sum(temp_cost)
                ten_batch_cost += np.sum(temp_cost)

                # record examples to monitoring the training process
                if((batch_index % monitoring_rate == 0)):
                    siamese_model.set_is_training(False)
                    r = sess.run([siamese_model.look_like], feed_dict = {siamese_model.x1: minibatch_X1, siamese_model.x2: minibatch_X2, siamese_model.keep_f: drop_out_rate})
                    distance = r[0][0]
                    same_list = r[0][1].astype('float')
                    countMax = np.sum(np.abs(same_list - minibatch_y) < 0.5)
                    # calculate distances in the embedding space
                    neg_dis, pos_dis, neg_c, pos_c, min_neg_dis, max_pos_dis = tf_distance(distance, minibatch_y, batch_size)
                    neg_distance += neg_dis
                    pos_distance += pos_dis
                    neg_count += neg_c
                    pos_count += pos_c
                    print("Epoch %i Batch %i After Optimization Count %i" %(epoch + 1,batch_index, countMax))
                    print("Epoch %i Batch %i Batch Cost %f Learning_rate %f" %(epoch + 1,batch_index, np.sum(temp_cost), sess.run(learning_rate) * 1e10))
                    print("Epoch %i Batch %i Pos_count %i Neg_count %i" %(epoch + 1,batch_index, pos_count, neg_count))
                    print("Epoch %i Batch %i Pos_avg_distance %f Neg_avg_distance %f" %(epoch + 1,batch_index, pos_distance/(pos_count+1e-10), neg_distance/(neg_count+1e-10)))

                if(batch_index % 10 == 0 and batch_index != 0):
                    print("10 batches cost: %f" % ten_batch_cost)
                    ten_batch_cost = 0.0

                correct_prediction += countMax

                batch_index += 1

            cost_list.append(minibatch_cost)
            accuracy_list.append(correct_prediction / (iter_time * batch_size))
            distance_list.append(pos_distance / (pos_count+1e-10))
            distance_list.append(neg_distance / (neg_count+1e-10))

            # information for per epoch
            print("Train Accuracy: %f. Pos_avg_distance: %f. Neg_avg_distance: %f" % (correct_prediction / (iter_time * batch_size), pos_distance / (pos_count+1e-10), neg_distance / (neg_count+1e-10)))
            print("Pos_count: %i. Neg_count: %i" % (pos_count, neg_count))    

            # save model
            if((epoch + 1) % save_frequency == 0):
                saver.save(sess, model_file)
                print('保存成功')
                fp1 = open('./cost_list.txt', 'w')
                fp2 = open('./accuracy_list.txt', 'w')
                fp3 = open('./distance.txt', 'w')
                for i in range(len(cost_list)):
                    fp1.write(str(cost_list[i]) + '\n')
                    fp2.write(str(accuracy_list[i]) + '\n')
                    fp3.write("Pos: " + str(distance_list[2 * i]) + "  Neg: " + str(distance_list[2*i+1]) + '\n')
                fp1.close()
                fp2.close()
                fp3.close()

            # print total cost of this epoch
            print("End Epoch %i" % (epoch + 1))
            print("Total cost of Epoch %f" % minibatch_cost)
    return cost_list


if __name__ == '__main__':
    # read in images and labels
    imgs, labels, count, begin_index, index = get_data(size, 'train')
    num_train_img = imgs.shape[0]
    model_file = './model/model'
    if(size % batch_size) == 0:
        total_batch = math.floor(size / batch_size + 1e-5)
    else:
        total_batch = math.floor(size / batch_size) + 1
    cost_list = Train(imgs, labels, count, begin_index, index)