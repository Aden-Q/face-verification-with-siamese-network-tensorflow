import numpy as np
from PIL import Image
from skimage import transform,data,img_as_ubyte
from random import shuffle
import matplotlib.pyplot as plt
import math

def get_data(size, data_type):
    '''
    Parameters:
        size: 指定要获取图像的数量
        data_type: 指定获取验证集还是训练集
    '''
    
    img_dir = '/home/guhanxue/jupyter/root/demo/tensorflow-triplet-loss/data/face/CASIA-WebFace-Align-96/'
    if(data_type == 'train'):
        f = open(img_dir + "train.txt","r")   #设置文件对象
    elif(data_type == 'val'):
        f = open(img_dir + "val.txt","r")
    elif(data_type == 'ghx_train'):
        f = open(img_dir + "ghx_train.txt","r")
    elif(data_type == 'ghx_val'):
        f = open(img_dir + "ghx_val.txt","r")
    imgs = np.zeros((size,112, 96, 3), dtype = 'uint8')         #!!!!!注意必须是这种格式才行
    labels = np.zeros((size,1), dtype = 'int32')
    count = [0]*20000 #统计每个类别有多少张图片
    i = 0
    for k in range(size):
        line = f.readline().strip('\n')
        p, label = line.split(' ')
        img_path = img_dir + p
        index=int(label)
        count[index] = count[index]+1 #需要
        # 打开图像并转化为数字矩阵(112x96x3)
        img = plt.imread(img_path).astype('uint8')
        labels[i] = label
        imgs[i] = img
        i += 1

    begin_index = [0]*(index+1)
    for i in range(index+1):
        if (i==0): 
            begin_index[i]=0
        else:
            begin_index [i] = begin_index[i-1]+count[i-1]
    f.close()

    return imgs, labels, count, begin_index, index

def data_random(imgs,index,labels,count,begin_index,batch_size,size):
    epoch = math.floor(size/batch_size/2)
    every_batch_same = math.floor(batch_size/2)
    every_batch_different = batch_size - every_batch_same
    train_index = 0
    train_imags = np.zeros((size,112, 96, 3), dtype = 'uint8')
    train_table = np.zeros((size,1), dtype = 'int32')
    import random
    for i in range (epoch):
        for j in range(every_batch_same):
            type_same = random.randint(0,index)
            while (count[type_same]==0):
                type_same = random.randint(0,index)
            choose1 = random.randint(0,count[type_same]-1)
            choose2 = random.randint(0,count[type_same]-1)
            while (choose2==choose1):
                choose2 = random.randint(0,count[type_same]-1)
            #每2个batch为1组，前16个存相同类别的图片地址
            index1 = begin_index[type_same] + choose1
            index2 = begin_index[type_same] + choose2
            #while (is_use[index1]==1):
            #       type_same = random.randint(0,index)
            #      choose1 = random.randint(0,count[type_same]-1)
            #     index1 = begin_index[type_same] + choose1
            train_imags[train_index,:,:,:] = imgs[index1,:,:,:]
            train_table[train_index] = labels[index1]
            train_imags[batch_size+train_index,:,:,:] = imgs[index2,:,:,:]
            train_table[batch_size+train_index] = labels[index2]
            train_index = train_index+1;
        for j in range(every_batch_different):
            type_different1 = random.randint(0,index)
            type_different2 = random.randint(0,index)
            while (count[type_different1]==0):
                type_different1 = random.randint(0,index)
            while (count[type_different2]==0):
                type_different2 = random.randint(0,index) 
            while (type_different1==type_different2):
                type_different2 = random.randint(0,index)
                #后16个存不同类别图片的地址
            choose1 = random.randint(0,count[type_different1]-1)
            choose2 = random.randint(0,count[type_different2]-1)
            index1 = begin_index[type_different1] + choose1
            index2 = begin_index[type_different2] + choose2
            train_imags[train_index,:,:,:] = imgs[index1,:,:,:]
            train_table[train_index] = labels [index1]
            train_imags[batch_size+train_index,:,:,:] = imgs[index2,:,:,:]
            train_table[batch_size+train_index] = labels[index2]
            train_index = train_index+1;
        train_index = train_index + batch_size;

    return train_imags, train_table

def data_shuffle(imgs, labels):
    index = [i for i in range(len(imgs))]
    shuffle(index)

    imgs = imgs[index, :, :, :]
    labels = labels[index, :]
    return imgs, labels

def get_minibatch(image, label, batch_size, now_batch, total_batch):
    if now_batch < total_batch-1:
        image_batch = image[now_batch*batch_size:(now_batch+1)*batch_size]
        label_batch = label[now_batch*batch_size:(now_batch+1)*batch_size]
    else:
        image_batch = image[now_batch*batch_size:]
        label_batch = label[now_batch*batch_size:]
    a = np.zeros((image_batch.shape[0],96,96,3))
    for i in range (image_batch.shape[0]):
        a[i] = transform.rescale(image_batch[i], [0.857,1])
        #a[i] = img_as_ubyte(transform.rescale(image_batch[i], [0.857,1]))  # 反归一化一次，因为ResNet里面有RGB的归一化操作
    return a, label_batch.reshape(label_batch.shape[0])

def tf_distance(distance, labels, batch_size):
    # 负样本距离
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