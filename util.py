import numpy as np
from PIL import Image
from skimage import transform,data,img_as_ubyte
from random import shuffle
import matplotlib.pyplot as plt
import math

def get_data(size, data_type):
    '''
    Read in images and labels on the training set

    Arguments:
        size: size of images
        data_type: train or val

    Return:
        imgs: images readed in
        labels: labels readed in
        count: for data_random
        begin_index: for data_random
        index: for data_random
    '''

    img_dir = '/home/guhanxue/jupyter/root/demo/tensorflow-triplet-loss/data/face/CASIA-WebFace-Align-96/'
    if(data_type == 'train'):  
        f = open(img_dir + "train.txt","r")
    elif(data_type == 'val'):
        f = open(img_dir + "val.txt","r")
    elif(data_type == 'ghx_train'):
        f = open(img_dir + "ghx_train.txt","r")
    elif(data_type == 'ghx_val'):
        f = open(img_dir + "ghx_val.txt","r")

    #imgs: sized 112*96*3
    imgs = np.zeros((size,112, 96, 3), dtype = 'uint8')
    labels = np.zeros((size,1), dtype = 'int32')
    count = [0]*20000 
    i = 0
   
    for k in range(size):
        line = f.readline().strip('\n')
        p, label = line.split(' ')
        img_path = img_dir + p
        index=int(label)
        #count[i]: the number of pictures labeled in i
        count[index] = count[index] + 1 
        # read images
        img = plt.imread(img_path).astype('uint8')
        labels[i] = label
        imgs[i] = img
        i += 1
    #begin_index: the first index of differnet labels 
    #index: the largest label of the images   
    begin_index = [0]*(index+1)
    for i in range(index+1):
        if (i==0): 
            begin_index[i]=0  
        else:
            begin_index [i] = begin_index[i-1]+count[i-1]
    f.close()

    return imgs, labels, count, begin_index, index

def data_random(imgs,index,labels,count,begin_index,batch_size,size):
    '''
    Data shuffle for training and valuation

    Arguments:
        imgs: 112*96*3
        index: the largest label
        labels: labels
        count:the number of different labels
        begin_index: first index of different labels
        batch_size: number of images per batch
        size: number of images in total

    Return:
        train_imags: images list after shuffling
        train_table: labels list after shuffling
    '''
    #number_of_batchpair: we need two batch each time
    number_of_batchpair = math.floor(size/batch_size/2)
    #half of the batch is the positive samples, and the rest is the negative samples
    #every_batch_same: the number of positive samples in each batch
    every_batch_same = math.floor(batch_size/2)
    #every_batch_different: the number of negative samples in each batch
    every_batch_different = batch_size - every_batch_same
    train_index = 0
    train_imags = np.zeros((size,112, 96, 3), dtype = 'uint8')
    train_table = np.zeros((size,1), dtype = 'int32')
    import random
    for i in range (number_of_batchpair):
        #choose the postive pairs
        for j in range(every_batch_same):
            #type_same: random choose the label of positive pairs
            type_same = random.randint(0,index)
            #if this type don't have any pictures, choose another one
            while (count[type_same]==0):
                type_same = random.randint(0,index)
            #choose two different images from the same type
            choose1 = random.randint(0,count[type_same]-1)
            choose2 = random.randint(0,count[type_same]-1)
            while (choose2==choose1):
                choose2 = random.randint(0,count[type_same])
            #find the index of the images in orginal imags, labels
            index1 = begin_index[type_same] + choose1
            index2 = begin_index[type_same] + choose2
            #store the samples
            train_imags[train_index,:,:,:] = imgs[index1,:,:,:]
            train_table[train_index] = labels[index1]
            train_imags[batch_size+train_index,:,:,:] = imgs[index2,:,:,:]
            train_table[batch_size+train_index] = labels[index2]
            train_index = train_index+1;
        #choose different pairs
        for j in range(every_batch_different):
            #choose two types
            type_different1 = random.randint(0,index)
            type_different2 = random.randint(0,index)
            #choose another type
            while (count[type_different1]==0):
                type_different1 = random.randint(0,index)
            while (count[type_different2]==0):
                type_different2 = random.randint(0,index) 
            while (type_different1==type_different2):
                type_different2 = random.randint(0,index)
            #choose two pictures
            choose1 = random.randint(0,count[type_different1]-1)
            choose2 = random.randint(0,count[type_different2]-1)
            #calculate the index
            index1 = begin_index[type_different1] + choose1
            index2 = begin_index[type_different2] + choose2
            #assign the value
            train_imags[train_index,:,:,:] = imgs[index1,:,:,:]
            train_table[train_index] = labels [index1]
            train_imags[batch_size+train_index,:,:,:] = imgs[index2,:,:,:]
            train_table[batch_size+train_index] = labels[index2]
            train_index = train_index+1;
        #find the next two batch
        train_index = train_index + batch_size;

    return train_imags, train_table


def data_shuffle(imgs, labels):
    '''
    Random shuffle

    Arguments:
        imgs: images list
        labels: labels list

    Return:
        imgs: images list after shuffling
        labels: labels list after shuffling
    '''

    index = [i for i in range(len(imgs))]
    shuffle(index)

    imgs = imgs[index, :, :, :]
    labels = labels[index, :]
    return imgs, labels


def get_minibatch(image, label, batch_size, now_batch, total_batch):
    '''
    Get minibath and rescale the images to 96*96*3

    Arguments:
        image: images list
        labels: labels list
        batch_size: number of images per batch
        now_batch: index of the current batch
        total_batch: index upper bound for batches

    Return:
        imgs_scaled: minibatch of images after scaling
        label_batch: minibatch of labels
    '''

    if now_batch < total_batch-1:
        image_batch = image[now_batch*batch_size:(now_batch+1)*batch_size]
        label_batch = label[now_batch*batch_size:(now_batch+1)*batch_size]
    else:
        image_batch = image[now_batch*batch_size:]
        label_batch = label[now_batch*batch_size:]
    imgs_scaled = np.zeros((image_batch.shape[0],96,96,3))
    for i in range (image_batch.shape[0]):
        imgs_scaled[i] = transform.rescale(image_batch[i], [0.857,1])
    return imgs_scaled, label_batch.reshape(label_batch.shape[0])

def tf_distance(distance, labels, batch_size):
    '''
    Calculate the distance of positive paris and negative paris in the embedding space

    Arguments:
        distance: distance list of all paris
        labels: labels of paris(positive or negative)
        batch_size: number of images per batch

    Return:
        neg: total distance of negative pair in a batch
        pos: total distance of positive pair in a batch
        neg_c: number of negative pairs
        pos_c: number of positive pairs
        min_neg_dis: minimum negative distance in a batch
        max_pos_dis: maximum positive distance in a batch
    '''

    neg = 0.0
    neg_c = 0
    # 正样本距离
    pos = 0.0
    pos_c = 0
    min_neg_dis = 1000.0
    max_pos_dis = 0.0
    for i in range(batch_size):
        if(labels[i] < 0.5):
            neg += distance[i]
            neg_c += 1
            if(distance[i] < min_neg_dis):
                min_neg_dis = distance[i]
        else:
            pos += distance[i]
            pos_c += 1
            if(distance[i] > max_pos_dis):
                max_pos_dis = distance[i]
    return neg, pos, neg_c, pos_c, min_neg_dis, max_pos_dis

