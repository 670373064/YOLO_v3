# -*- coding:utf-8 -*-
#
# Written by leeyoshinari
#
#2018-04-21

import tensorflow as tf
import numpy as np
import yolo3.config as cfg

class yolo_v3(object):
    def __init__(self):
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
        
    def yolo_net(self, input):
        net = self.conv_layer(input, 3, 32, idx = 0)

        net = self.conv_layer(net, 3, 64, stride = 2, idx = 1)
        short_cut = net
        net = self.conv_layer(net, 1, 32, idx = 2)
        net = self.conv_layer(net, 3, 64, idx = 3)
        net = tf.add(short_cut, net, name = '4_add')

        net = self.conv_layer(net, 3, 128, stride = 2, idx = 5)
        short_cut = net
        net = self.conv_layer(net, 1, 64, idx = 6)
        net = self.conv_layer(net, 3, 128, idx = 7)
        net = tf.add(short_cut, net, name = '8_add')

        short_cut = net
        net = self.conv_layer(net, 1, 64, idx = 9)
        net = self.conv_layer(net, 3, 128, idx = 10)
        net = tf.add(short_cut, net, name = '11_add')

        net = self.conv_layer(net, 3, 256, stride = 2, idx = 12)
        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 13)
        net = self.conv_layer(net, 3, 256, idx = 14)
        net = tf.add(short_cut, net, name = '15_add')

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 16)
        net = self.conv_layer(net, 3, 256, idx = 17)
        net = tf.add(short_cut, net, name = '18_add')

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 19)
        net = self.conv_layer(net, 3, 256, idx = 20)
        net = tf.add(short_cut, net, name = '21_add')

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 22)
        net = self.conv_layer(net, 3, 256, idx = 23)
        net = tf.add(short_cut, net, name = '24_add')

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 25)
        net = self.conv_layer(net, 3, 256, idx = 26)
        net = tf.add(short_cut, net, name = '27_add')

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 28)
        net = self.conv_layer(net, 3, 256, idx = 29)
        net = tf.add(short_cut, net, name = '30_add')

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 31)
        net = self.conv_layer(net, 3, 256, idx = 32)
        net = tf.add(short_cut, net, name = '33_add')

        short_cut = net
        net = self.conv_layer(net, 1, 128, idx = 34)
        net = self.conv_layer(net, 3, 256, idx = 35)
        net36 = tf.add(short_cut, net, name = '36_add')

        net = self.conv_layer(net36, 3, 512, stride = 2, idx = 37)
        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 38)
        net = self.conv_layer(net, 3, 512, idx = 39)
        net = tf.add(short_cut, net, name = '40_add')

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 41)
        net = self.conv_layer(net, 3, 512, idx = 42)
        net = tf.add(short_cut, net, name = '43_add')

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 44)
        net = self.conv_layer(net, 3, 512, idx = 45)
        net = tf.add(short_cut, net, name = '46_add')

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 47)
        net = self.conv_layer(net, 3, 512, idx = 48)
        net = tf.add(short_cut, net, name = '49_add')

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 50)
        net = self.conv_layer(net, 3, 512, idx = 51)
        net = tf.add(short_cut, net, name = '52_add')

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 53)
        net = self.conv_layer(net, 3, 512, idx = 54)
        net = tf.add(short_cut, net, name = '55_add')

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 56)
        net = self.conv_layer(net, 3, 512, idx = 57)
        net = tf.add(short_cut, net, name = '58_add')

        short_cut = net
        net = self.conv_layer(net, 1, 256, idx = 59)
        net = self.conv_layer(net, 3, 512, idx = 60)
        net61 = tf.add(short_cut, net, name = '61_add')

        net = self.conv_layer(net61, 3, 1024, stride = 2, idx = 62)
        short_cut = net
        net = self.conv_layer(net, 1, 512, idx = 63)
        net = self.conv_layer(net, 3, 1024, idx = 64)
        net = tf.add(short_cut, net, name = '65_add')

        short_cut = net
        net = self.conv_layer(net, 1, 512, idx = 66)
        net = self.conv_layer(net, 3, 1024, idx = 67)
        net = tf.add(short_cut, net, name = '68_add')

        short_cut = net
        net = self.conv_layer(net, 1, 512, idx = 69)
        net = self.conv_layer(net, 3, 1024, idx = 70)
        net = tf.add(short_cut, net, name = '71_add')

        short_cut = net
        net = self.conv_layer(net, 1, 512, idx = 72)
        net = self.conv_layer(net, 3, 1024, idx = 73)
        net = tf.add(short_cut, net, name = '74_add')

        net = self.conv_layer(net, 1, 512, idx = 75)
        net = self.conv_layer(net, 3, 1024, idx = 76)
        net = self.conv_layer(net, 1, 512, idx = 77)
        net = self.conv_layer(net, 3, 1024, idx = 78)
        net79 = self.conv_layer(net, 1, 512, idx = 79)
        net = self.conv_layer(net79, 3, 1024, idx = 80)
        net81 = self.conv_layer(net, 1, self.output_size, batch_norm = False, idx = 81)

        net = self.conv_layer(net79, 1, 256, idx = 84)
        net = self.upsampling(net, idx = 85)
        net = tf.concat([net, net61], axis = 3)

        net = self.conv_layer(net, 1, 256, idx = 87)
        net = self.conv_layer(net, 3, 512, idx = 88)
        net = self.conv_layer(net, 1, 256, idx = 89)
        net = self.conv_layer(net, 3, 512, idx = 90)
        net91 = self.conv_layer(net, 1, 256, idx = 91)
        net = self.conv_layer(net91, 3, 512, idx = 92)
        net93 = self.conv_layer(net, 1, self.output_size, batch_norm = False, idx = 93)

        net = self.conv_layer(net91, 1, 128, idx = 96)
        net = self.upsampling(net, idx = 97)
        net = tf.concat([net, net36], axis = 3)

        net = self.conv_layer(net, 1, 128, idx = 99)
        net = self.conv_layer(net, 3, 256, idx = 100)
        net = self.conv_layer(net, 1, 128, idx = 101)
        net = self.conv_layer(net, 3, 256, idx = 102)
        net = self.conv_layer(net, 1, 128, idx = 103)
        net = self.conv_layer(net, 3, 256, idx = 104)
        net = self.conv_layer(net, 1, self.output_size, batch_norm = False, idx = 105)

        return [net81, net93, net]


    def conv_layer(self, input, filter, size, stride = 1, batch_norm = True, idx = 0):
        channel = input.shape[3]
        weight = tf.Variable(tf.truncated_normal([filter, filter, channel, size], stddev = 0.1, name = 'weights'))
        biases = tf.Variable(tf.constant(0.1, shape = [size], name = 'biases'))
        conv = tf.nn.conv2d(input, weight, strides = [1, stride, stride, 1], padding = 'SAME', name = str(idx)+'_conv')

        if batch_norm:
            depth = conv.shape[3]
            scale = tf.Variable(tf.ones([depth,], dtype = 'float32'), name = 'scale')
            shift = tf.Variable(tf.zeros([depth,], dtype = 'float32'), name = 'shift')
            mean = tf.Variable(tf.ones([depth,], dtype = 'float32'), name = 'rolling_mean')
            variance = tf.Variable(tf.ones([depth,], dtype = 'float32'), name = 'rolling_variance')

            conv = tf.nn.batch_normalization(conv, mean, variance, shift, scale, 1e-05)
        
        conv = tf.add(conv, biases)

        return tf.maximum(self.alpha * conv, conv)


    def upsampling(self, input, size = 2, idx = 1):
        shape = input.shape
        net_transfer = tf.reshape(input, [shape[0], shape[1], 1, shape[2], 1, shape[3]])
        net_transfer = tf.tile(net_transfer, (1, 1, size, 1, size, 1))
        net_transfer = tf.reshape(net_transfer, [shape[0], shape[1]*size, shape[2]*size, shape[3]])

        return net_transfer


    def loss_layer(self, predict, labels):
        grid_shape = predict.shape
        offset = np.reshape(np.array([np.arange(grid_shape[1])] * grid_shape[2] * self.box_per_cell), [grid_shape[1], grid_shape[1], self.box_per_cell])
        offset = tf.reshape(tf.constant(tf.transpose(offset, (1, 2, 0)), dtype=tf.float32), [1, grid_shape[1], grid_shape[2], self.box_per_cell])
        offset = tf.tile(offset, (self.batch_size, 1, 1, 1))

        predict = tf.reshape(predict, [self.batch_size, grid_shape[1], grid_shape[2], self.box_per_cell, self.num_class+5])
        box_coordinate = tf.reshape(predict[:, :, :, :, :4], [self.batch_size, grid_shape[1], grid_shape[2], self.box_per_cell, 4])
        box_confidence = tf.reshape(predict[:, :, :, :, 4], [self.batch_size, grid_shape[1], grid_shape[2], self.box_per_cell])
        box_classes = tf.reshape(predict[:, :, :, :, 5:], [self.batch_size, grid_shape[1], grid_shape[2], self.box_per_cell, self.num_class])

        boxes1 = tf.stack([(1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 0])) + offset) / grid_shape[1],
                           (1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 1])) + tf.transpose(offset, (0, 2, 1, 3)))/ grid_shape[2],
                           tf.exp(box_coordinate[:, :, :, :, 2]) * anchor / grid_shape[1],
                           tf.exp(box_coordinate[:, :, :, :, 3]) * anchor / grid_shape[2]])
        box_coor_trans = tf.transpose(boxes1, (1, 2, 3, 4, 0))
        box_confidence = 1.0 / (1.0 + tf.exp(-1.0 * box_confidence))
        box_classes = 1.0 / (1.0 + tf.exp(-1.0 * box_classes))