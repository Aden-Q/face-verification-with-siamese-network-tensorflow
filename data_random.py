    import math
    import numpy as np
    from random import shuffle
    size = 300000
    batch_size = 32
    count = [0]*20000 #统计每个类别有多少张图片
    imgs = np.zeros((size,112, 96, 3), dtype = 'uint8')
    labels = np.zeros((size,1), dtype = 'int32')
    #读入部分可能还是要改改,应该是你的输入部分+
    with open("/home/guhanxue/jupyter/root/demo/tensorflow-triplet-loss/data/face/CASIA-WebFace-Align-96/train.txt","r") as f:
        for k in range(size):
            line = f.readline().strip('\n')
            p, label = line.split(' ')
            index=int(label)
            addr=p  #存储路径，不需要用数组
            #把存图片放这里，因为还是要算count和begin_index
            #img = plt.imread(img_path).astype('uint8')
            #imgs[i]=img;
            labels[k]=label  #存储类别
            count[index] = count[index]+1 #需要
    #index储存总共有多少类，需要
    epoch = math.floor((size/batch_size)/2)
    begin_index = [0]*(index+1)
    for i in range(index+1):
        if (i==0): 
            begin_index[i]=0
        else:
            begin_index [i] = begin_index[i-1]+count[i-1] 

    #开始生成，每两个连续的batch为一组。
    #前16个先从所有类别中选出一类，再从该类中选出2张图片
    #后16个先从所有类别中选出2个不同的类，再从这两类中选出两张图片
    def data_random(imgs,index,labels,count,begin_index,batch_size,size ):
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
                      choose2 = random.randint(0,count[type_same])
                #每2个batch为1组，前16个存相同类别的图片地址
                index1 = begin_index[type_same] + choose1
                index2 = begin_index[type_same] + choose2
                #while (is_use[index1]==1):
                 #       type_same = random.randint(0,index)
                  #      choose1 = random.randint(0,count[type_same]-1)
                   #     index1 = begin_index[type_same] + choose1
                train_imags[train_index,:,:,:] = imgs [index1,:,:,:]
                train_table[train_index] = labels [index1]
                train_imags[batch_size+train_index,:,:,:] = imgs [index2,:,:,:]
                train_table[batch_size+train_index] = labels [index2]
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
                train_imags [train_index,:,:,:] = imgs [index1,:,:,:]
                train_table [train_index] = labels [index1]
                train_imags [batch_size+train_index,:,:,:] = imgs [index2,:,:,:]
                train_table [batch_size+train_index] = labels [index2]
                train_index = train_index+1;
            train_index = train_index + batch_size;




   # #保存文件,这边最好删掉
   #     doc = open('/home/guhanxue/jupyter/root/demo/train400000.txt','w')
   # for i in range (size):
   #     print(train_imags[i],file=doc,end=' ')
   #     print(train_table[i],file=doc)
   #     print('\n')
   # doc.close()
return [train_imags,train_table]