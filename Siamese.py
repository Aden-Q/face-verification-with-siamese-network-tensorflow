import tensorflow as tf
import numpy as np


class Siamese:

    # Create model
    def __init__(self, Siamese_npy_path=None, trainable=True, size=96):
        '''
        Attributions of Siamese network class
        For siamese or triplet
        '''

        self.x1 = tf.placeholder(tf.float32, [None, size, size, 3])
        self.x2 = tf.placeholder(tf.float32, [None, size, size, 3])
        self.d1 = tf.placeholder(tf.float32, [None, 128])
        self.d2 = tf.placeholder(tf.float32, [None, 128])
        self.keep_f = tf.placeholder(tf.float32)
        self.is_training = True
        self.trainable = trainable

        if Siamese_npy_path is not None:
            self.data_dict = np.load(Siamese_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1, self.keep_f)
            scope.reuse_variables()
            self.o2 = self.network(self.x2, self.keep_f)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()
        self.look_like = self.cal_distance()

    def set_is_training(self, isTrain):
        """
        Set is training bool.
        """

        self.is_training = isTrain

    def network(self, x, keep_f):
        with tf.variable_scope("conv1"):
            # the first conv layer, input 96*96, output 96*96
            conv1 = self.cnn_layer(x, [3, 3, 3, 64], [64])
        with tf.variable_scope("conv2"):
            # the second conv layer, input 96*96, output 48*48
            conv2 = self.cnn_layer(conv1, [3, 3, 64, 64], [64])
            pool1 = self.pool_layer(conv2, 1.0)

        with tf.variable_scope("conv3"):
            # the third conv layer, input 48*48, output 48*48
            conv3 = self.cnn_layer(pool1, [3, 3, 64, 128], [128])
        with tf.variable_scope("conv4"):
            # the forth conv layer, input 48*48, output 24*24
            conv4 = self.cnn_layer(conv3, [3, 3, 128, 128], [128])
            pool2 = self.pool_layer(conv4, 1.0)

        with tf.variable_scope("conv5"):
            # the fifth conv layer, input 24*24, output 24*24
            conv5 = self.cnn_layer(pool2, [3, 3, 128, 256], [256])
        with tf.variable_scope("conv6"):
            # the sixth conv layer, input 24*24, output 24*24
            conv6 = self.cnn_layer(conv5, [3, 3, 256, 256], [256])
        with tf.variable_scope("conv7"):
            # the seventh conv layer, input 24*24, output 12*12
            conv7 = self.cnn_layer(conv6, [3, 3, 256, 256], [256])
            pool3 = self.pool_layer(conv7, 1.0)

        with tf.variable_scope("conv8"):
            # the eigth ocnv layer, input 12*12, output 12*12
            conv8 = self.cnn_layer(pool3, [3, 3, 256, 512], [512])
        with tf.variable_scope("conv9"):
            # the ninth conv layer, input 12*12, output 12*12
            conv9 = self.cnn_layer(conv8, [3, 3, 512, 512], [512])
        with tf.variable_scope("conv10"):
            # the tenth conv layer, input 12*12, output 6*6
            conv10 = self.cnn_layer(conv9, [3, 3, 512, 512], [512])
            pool4 = self.pool_layer(conv10, 1.0)

        with tf.variable_scope("conv11"):
            # the eleventh conv layer, input 6*6, output 6*6
            conv11 = self.cnn_layer(pool4, [3, 3, 512, 512], [512])
        with tf.variable_scope("conv12"):
            # the twelfth conv layer, input 6*6, output 6*6
            conv12 = self.cnn_layer(conv11, [3, 3, 512, 512], [512])
        with tf.variable_scope("conv13"):
            # the thirteenth conv layer, input 6*6, output 3*3
            conv13 = self.cnn_layer(conv12, [3, 3, 512, 512], [512])
            pool5 = self.pool_layer(conv13, 1.0)

        with tf.variable_scope("full_layer1"):
            # fuller connected layer, input 3*3
            self.f1 = self.full_layer(pool5, [3 * 3 * 512, 128], [128], keep_f, True)
        #with tf.variable_scope("full_layer2"):
        #    # fc 2, input 4096
        #    self.f2 = self.full_layer(self.f1, [4096, 1024], [1024], keep_f)
        #with tf.variable_scope("full_layer3"):
            # fc 3, input 1024
        #    self.f3 = self.full_layer(self.f2, [1024, 128], [128], 1.0)
            # Create prob
        #    self.prob = tf.nn.softmax(self.f3, name="prob")
        
        return tf.nn.relu(self.f1)

    @staticmethod
    def cnn_layer(input_image, kernel_shape, bias_shape):
        init = tf.truncated_normal_initializer(stddev=0.04)
        weights = tf.get_variable("cnn_weights", dtype=tf.float32, shape=kernel_shape,
                                  initializer=init)

        biases = tf.get_variable("cnn_biases", dtype=tf.float32,
                                 initializer=tf.constant(0.01, shape=bias_shape, dtype=tf.float32))
        conv = tf.nn.conv2d(input_image, weights,
                            strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(conv + biases)

    @staticmethod
    def pool_layer(input_image, keep):
        pool = tf.nn.max_pool(input_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        drop = tf.nn.dropout(pool, keep)
        return drop

    @staticmethod
    def full_layer(input_image,  kernel_shape, bias_shape, keep, reshape=False):
        init = tf.truncated_normal_initializer(stddev=0.04)
        weights = tf.get_variable("cnn_weights", dtype=tf.float32, shape=kernel_shape,
                                  initializer=init)

        biases = tf.get_variable("cnn_biases", dtype=tf.float32,
                                 initializer=tf.constant(0.01, shape=bias_shape, dtype=tf.float32))
        if reshape:
            input_image = tf.reshape(input_image, [-1, 3*3*512])
        dense = tf.nn.relu(tf.matmul(input_image, weights) + biases)
        drop = tf.nn.dropout(dense, keep)
        return drop

    def loss_with_spring(self):
        # margin const for negtive loss
        margin = 7.0
        # output of the fuller connected layer
        face1_output = self.o1     # shape [None, 128]
        face2_output = self.o2   # shape [None, 128]
        # labels for paris
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")
        # norm 2
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        # positive loss and negative loss
        C = tf.constant(margin, name="C")
        # pos是相同类别对的损失，neg是不同类别对的损失
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")

        return loss

    def loss_with_step(self):
        # another loss function (not used)
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")

        return loss

    def cal_distance(self):
        face1_output = self.o1  # shape [None, 128]
        face2_output = self.o2  # shape [None, 128]
        d_look = tf.reduce_sum(tf.square(face1_output - face2_output), 1, name="d_look")
        # threshold for judging
        same_list = d_look < 15
        distance = tf.reduce_mean(d_look, name="distance")
        return d_look, same_list

    def get_var(self, initial_value, name, idx, var_name):
        """
        load variables from Loaded model or new generated random variables
        initial_value : random initialized value
        name: block_layer name
        index: 0,1 weight or bias
        var_name: name + "_filter"/"_bias"
        """

        if((name, idx) in self.var_dict):
            print("Reuse Parameters...")
            print(self.var_dict[(name, idx)])
            return self.var_dict[(name, idx)]

        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./model/Resnet-save.npy"):
        """
        Save this model into a npy file
        """
        
        assert isinstance(sess, tf.Session)
        
        self.data_dict = None
        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count