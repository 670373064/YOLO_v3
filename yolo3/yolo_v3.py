# -*- coding:utf-8 -*-
#
# Written by leeyoshinari
#
#2018-04-21

import tensorflow as tf
import numpy as np
import yolo3.config as cfg

class yolo_v3(object):
    def __init__(self, is_training = True):
        self.weights_file = cfg.WEIGHTS_FILE
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.box_per_cell = cfg.BOX_PER_CELL
        self.output_size = self.box_per_cell * (self.num_class + 5)
        self.anchor = np.reshape(cfg.ANCHOR, [-1,2])
        self.alpha = cfg.ALPHA

        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name = 'images')
        self.logits = self.yolo_net(self.images)

        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, None, None, None, None])
            self.total_loss = self.loss_layer(self.logits, self.labels)
            #tf.summary.scalar('total_loss', self.total_loss)
        
    def yolo_net(self, input):
        net = self.conv_layer(input, 3, 32, idx = 1)

        net = self.conv_layer(net, 3, 64, stride = 2, idx = 2)
        short_cut = net
        net = self.conv_layer(net, 1, 32, idx = 3)
        net = self.conv_layer(net, 3, 64, idx = 4)
        net = tf.add(short_cut, net)

        net = self.conv_layer(net, 3, 128, stride = 2, idx = 5)
        short_cut = net
        net = self.conv_layer(net, 1, 64, idx = 6)
        net = self.conv_layer(net, 3, 128, idx = 7)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 64, idx = 8)
        net = self.conv_layer(net, 3, 128, idx = 9)
        net = tf.add(short_cut, net)

        net = self.conv_layer(net, 3, 256, stride = 2, idx = 10)
        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 11)
        net = self.conv_layer(net, 3, 256, idx = 12)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 13)
        net = self.conv_layer(net, 3, 256, idx = 14)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 15)
        net = self.conv_layer(net, 3, 256, idx = 16)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 17)
        net = self.conv_layer(net, 3, 256, idx = 18)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 19)
        net = self.conv_layer(net, 3, 256, idx = 20)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 21)
        net = self.conv_layer(net, 3, 256, idx = 22)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 23)
        net = self.conv_layer(net, 3, 256, idx = 24)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 25)
        net = self.conv_layer(net, 3, 256, idx = 26)
        net36 = tf.add(short_cut, net)

        net = self.conv_layer(net36, 3, 512, stride = 2, idx = 27)
        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 28)
        net = self.conv_layer(net, 3, 512, idx = 29)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 30)
        net = self.conv_layer(net, 3, 512, idx = 31)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 32)
        net = self.conv_layer(net, 3, 512, idx = 33)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 34)
        net = self.conv_layer(net, 3, 512, idx = 35)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 36)
        net = self.conv_layer(net, 3, 512, idx = 37)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 38)
        net = self.conv_layer(net, 3, 512, idx = 39)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 40)
        net = self.conv_layer(net, 3, 512, idx = 41)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 42)
        net = self.conv_layer(net, 3, 512, idx = 43)
        net61 = tf.add(short_cut, net)

        net = self.conv_layer(net61, 3, 1024, stride = 2, idx = 44)
        short_cut = net
        net = self.conv_layer(net, 1, 512, idx = 45)
        net = self.conv_layer(net, 3, 1024, idx = 46)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 512, idx = 47)
        net = self.conv_layer(net, 3, 1024, idx = 48)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 512, idx = 49)
        net = self.conv_layer(net, 3, 1024, idx = 50)
        net = tf.add(short_cut, net)

        short_cut = net
        net = self.conv_layer(net, 1, 512, idx = 51)
        net = self.conv_layer(net, 3, 1024, idx = 52)
        net = tf.add(short_cut, net)

        net = self.conv_layer(net, 1, 512, idx = 53)
        net = self.conv_layer(net, 3, 1024, idx = 54)
        net = self.conv_layer(net, 1, 512, idx = 55)
        net = self.conv_layer(net, 3, 1024, idx = 56)
        net57 = self.conv_layer(net, 1, 512, idx = 57)
        net = self.conv_layer(net57, 3, 1024, idx = 58)
        net59 = self.conv_layer(net, 1, self.output_size, batch_norm = False, activation = False, idx = 59)

        net = self.conv_layer(net57, 1, 256, idx = 60)
        net = self.upsampling(net)
        net = tf.concat([net, net61], axis = 3)

        net = self.conv_layer(net, 1, 256, idx = 61)
        net = self.conv_layer(net, 3, 512, idx = 62)
        net = self.conv_layer(net, 1, 256, idx = 63)
        net = self.conv_layer(net, 3, 512, idx = 64)
        net65 = self.conv_layer(net, 1, 256, idx = 65)
        net = self.conv_layer(net65, 3, 512, idx = 66)
        net67 = self.conv_layer(net, 1, self.output_size, batch_norm = False, activation = False, idx = 67)

        net = self.conv_layer(net65, 1, 128, idx = 68)
        net = self.upsampling(net)
        net = tf.concat([net, net36], axis = 3)

        net = self.conv_layer(net, 1, 128, idx = 69)
        net = self.conv_layer(net, 3, 256, idx = 70)
        net = self.conv_layer(net, 1, 128, idx = 71)
        net = self.conv_layer(net, 3, 256, idx = 72)
        net = self.conv_layer(net, 1, 128, idx = 73)
        net = self.conv_layer(net, 3, 256, idx = 74)
        net = self.conv_layer(net, 1, self.output_size, batch_norm = False, activation = False, idx = 75)

        return [net59, net67, net]


    def conv_layer(self, input, filter, size, stride = 1, batch_norm = True, activation = True, idx = 0):
        with tf.variable_scope('convolutional'+str(idx)):
            channel = int(input.shape[3])
            weight = tf.Variable(tf.truncated_normal([filter, filter, channel, size], stddev=0.1), name='weights')
            #biases = tf.Variable(tf.constant(0.1, shape=[size]), name='biases')
            conv = tf.nn.conv2d(input, weight, strides=[1, stride, stride, 1], padding='SAME')

            if batch_norm:
                with tf.variable_scope('BatchNorm'):
                    depth = conv.shape[3]
                    betas = tf.Variable(tf.ones([depth, ], dtype='float32'), name='beta')
                    shift = None#tf.Variable(tf.zeros([depth, ], dtype='float32'), name='shift')
                    mean = tf.Variable(tf.ones([depth, ], dtype='float32'), name='moving_mean')
                    variance = tf.Variable(tf.ones([depth, ], dtype='float32'), name='moving_variance')

                    conv = tf.nn.batch_normalization(conv, mean, variance, shift, betas, 1e-05, name='BatchNorm')

            #conv = tf.add(conv, biases)

        if activation:
            return tf.maximum(self.alpha * conv, conv)
        else:
            return conv


    def upsampling(self, input, size = 2):
        shape = input.shape
        heights = int(shape[1])
        widths = int(shape[2])
        channels = int(shape[3])
        net_transfer = tf.reshape(input, [-1, heights, 1, widths, 1, channels])
        net_transfer = tf.tile(net_transfer, (1, 1, size, 1, size, 1))
        net_transfer = tf.reshape(net_transfer, [-1, heights*size, widths*size, channels])

        return net_transfer


    def loss_layer(self, predict, labels):
        grid_shape = np.shape(predict[0])
        print(grid_shape)
        offset = np.transpose(np.reshape(np.array([np.arange(13)] * 13 * self.box_per_cell), [13, 13, self.box_per_cell]), (1, 2, 0))
        offset = tf.reshape(tf.constant(offset, dtype=tf.float32), [1, 13, 13, self.box_per_cell])
        offset = tf.tile(offset, (self.batch_size, 1, 1, 1))

        predict = tf.reshape(predict[0], [self.batch_size, 13, 13, self.box_per_cell, self.num_class+5])
        box_coordinate = tf.reshape(predict[:, :, :, :, :4], [self.batch_size, 13, 13, self.box_per_cell, 4])
        box_confidence = tf.reshape(predict[:, :, :, :, 4], [self.batch_size, 13, 13, self.box_per_cell])
        box_classes = tf.reshape(predict[:, :, :, :, 5:], [self.batch_size, 13, 13, self.box_per_cell, self.num_class])

        '''boxes1 = tf.stack([(1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 0])) + offset) / 13,
                           (1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 1])) + tf.transpose(offset, (0, 2, 1, 3)))/ 13,
                           tf.exp(box_coordinate[:, :, :, :, 2]) * self.anchor / 13,
                           tf.exp(box_coordinate[:, :, :, :, 3]) * self.anchor / 13])
        box_coor_trans = tf.transpose(boxes1, (1, 2, 3, 4, 0))'''
        box_confidence = 1.0 / (1.0 + tf.exp(-1.0 * box_confidence))
        box_confidence = tf.tile(tf.expand_dims(box_confidence, axis = 4), (1, 1, 1, 1, self.num_class))
        box_classes = 1.0 / (1.0 + tf.exp(-1.0 * box_classes))

        loss = tf.reduce_mean(tf.square(box_confidence - box_classes), axis = 4)
        loss = tf.reduce_mean(tf.reduce_mean(tf.reshape(loss, [self.batch_size, 13*13*self.box_per_cell])))

        return loss