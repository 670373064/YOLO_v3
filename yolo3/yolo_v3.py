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
        self.cell_size = cfg.CELL_SIZE
        self.box_per_cell = cfg.BOX_PER_CELL
        self.output_size = self.box_per_cell * (self.num_class + 5)
        self.anchor = np.reshape(cfg.ANCHOR, [-1,2])
        self.alpha = cfg.ALPHA

        self.class_scale = 1.0
        self.object_scale = 5.0
        self.noobject_scale = 1.0
        self.coordinate_scale = 1.0

        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name = 'images')
        self.logits = self.yolo_net(self.images)

        if is_training:
            self.label1 = tf.placeholder(tf.float32, [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class+5])
            self.label2 = tf.placeholder(tf.float32, [self.batch_size, self.cell_size*2, self.cell_size*2, self.box_per_cell, self.num_class + 5])
            self.label3 = tf.placeholder(tf.float32, [self.batch_size, self.cell_size*4, self.cell_size*4, self.box_per_cell, self.num_class + 5])
            self.total_loss = self.calc_loss(self.logits, [self.label1, self.label2, self.label3])
            tf.summary.scalar('total_loss', self.total_loss)
        
    def yolo_net(self, input):
        net = self.conv_layer(input, 3, 32, idx = 1)  #0

        short_cut = net = self.conv_layer(net, 3, 64, stride = 2, idx = 2)  #1
        #short_cut = net
        net = self.conv_layer(net, 1, 32, idx = 3)  #2
        net = self.conv_layer(net, 3, 64, idx = 4)  #3
        net = tf.add(short_cut, net)  #4

        short_cut = net = self.conv_layer(net, 3, 128, stride = 2, idx = 5)  #5
        net = self.conv_layer(net, 1, 64, idx = 6)  #6
        net = self.conv_layer(net, 3, 128, idx = 7)  #7
        short_cut = net = tf.add(short_cut, net)  #8

        net = self.conv_layer(net, 1, 64, idx = 8)  #9
        net = self.conv_layer(net, 3, 128, idx = 9)  #10
        net = tf.add(short_cut, net)  #11

        short_cut = net = self.conv_layer(net, 3, 256, stride = 2, idx = 10)  #12
        net = self.conv_layer(net, 1, 128, idx = 11)  #13
        net = self.conv_layer(net, 3, 256, idx = 12)  #14
        short_cut = net = tf.add(short_cut, net)  #15

        net = self.conv_layer(net, 1, 128, idx = 13)  #16
        net = self.conv_layer(net, 3, 256, idx = 14)  #16
        short_cut = net = tf.add(short_cut, net)  #18

        net = self.conv_layer(net, 1, 128, idx = 15)  #19
        net = self.conv_layer(net, 3, 256, idx = 16)  #20
        short_cut = net = tf.add(short_cut, net)  #21

        net = self.conv_layer(net, 1, 128, idx = 17)  #22
        net = self.conv_layer(net, 3, 256, idx = 18)  #23
        short_cut = net = tf.add(short_cut, net)  #24

        net = self.conv_layer(net, 1, 128, idx = 19)  #25
        net = self.conv_layer(net, 3, 256, idx = 20)  #26
        short_cut = net = tf.add(short_cut, net)  #27

        net = self.conv_layer(net, 1, 128, idx = 21)  #28
        net = self.conv_layer(net, 3, 256, idx = 22)  #29
        short_cut = net = tf.add(short_cut, net)  #30

        net = self.conv_layer(net, 1, 128, idx = 23)  #31
        net = self.conv_layer(net, 3, 256, idx = 24)  #32
        short_cut = net = tf.add(short_cut, net)  #33

        net = self.conv_layer(net, 1, 128, idx = 25)  #34
        net = self.conv_layer(net, 3, 256, idx = 26)  #35
        net36 = tf.add(short_cut, net)  #36

        short_cut = net = self.conv_layer(net36, 3, 512, stride = 2, idx = 27)  #37
        net = self.conv_layer(net, 1, 256, idx = 28)  #38
        net = self.conv_layer(net, 3, 512, idx = 29)  #39
        short_cut = net = tf.add(short_cut, net)  #40

        net = self.conv_layer(net, 1, 256, idx = 30)  #41
        net = self.conv_layer(net, 3, 512, idx = 31)  #42
        short_cut = net = tf.add(short_cut, net)  #43

        net = self.conv_layer(net, 1, 256, idx = 32)  #44
        net = self.conv_layer(net, 3, 512, idx = 33)  #45
        short_cut = net = tf.add(short_cut, net)  #46

        net = self.conv_layer(net, 1, 256, idx = 34)  #47
        net = self.conv_layer(net, 3, 512, idx = 35)  #48
        short_cut = net = tf.add(short_cut, net)  #49

        net = self.conv_layer(net, 1, 256, idx = 36)  #50
        net = self.conv_layer(net, 3, 512, idx = 37)  #51
        short_cut = net = tf.add(short_cut, net)  #52

        net = self.conv_layer(net, 1, 256, idx = 38)  #53
        net = self.conv_layer(net, 3, 512, idx = 39)  #54
        short_cut = net = tf.add(short_cut, net)  #55

        net = self.conv_layer(net, 1, 256, idx = 40)  #56
        net = self.conv_layer(net, 3, 512, idx = 41)  #57
        short_cut = net = tf.add(short_cut, net)  #58

        net = self.conv_layer(net, 1, 256, idx = 42)  #59
        net = self.conv_layer(net, 3, 512, idx = 43)  #60
        net61 = tf.add(short_cut, net)  #61

        short_cut = net = self.conv_layer(net61, 3, 1024, stride = 2, idx = 44)  #62
        net = self.conv_layer(net, 1, 512, idx = 45)  #63
        net = self.conv_layer(net, 3, 1024, idx = 46)  #64
        short_cut = net = tf.add(short_cut, net)  #65

        net = self.conv_layer(net, 1, 512, idx = 47)  #66
        net = self.conv_layer(net, 3, 1024, idx = 48)  #67
        short_cut = net = tf.add(short_cut, net)  #68

        net = self.conv_layer(net, 1, 512, idx = 49)  #69
        net = self.conv_layer(net, 3, 1024, idx = 50)  #70
        short_cut = net = tf.add(short_cut, net)  #71

        net = self.conv_layer(net, 1, 512, idx = 51)  #72
        net = self.conv_layer(net, 3, 1024, idx = 52)  #73
        net = tf.add(short_cut, net)  #74

        net = self.conv_layer(net, 1, 512, idx = 53)  #75
        net = self.conv_layer(net, 3, 1024, idx = 54)  #76
        net = self.conv_layer(net, 1, 512, idx = 55)  #77
        net = self.conv_layer(net, 3, 1024, idx = 56)  #78
        net57 = self.conv_layer(net, 1, 512, idx = 57)  #79
        net = self.conv_layer(net57, 3, 1024, idx = 58)  #80
        net59 = self.conv_layer(net, 1, self.output_size, batch_norm = False, activation = False, idx = 59)  #81

        net = self.conv_layer(net57, 1, 256, idx = 60)  #84
        net = self.upsampling(net)  #85
        net = tf.concat([net, net61], axis = 3)  #86f

        net = self.conv_layer(net, 1, 256, idx = 61)  #87
        net = self.conv_layer(net, 3, 512, idx = 62)  #88
        net = self.conv_layer(net, 1, 256, idx = 63)  #89
        net = self.conv_layer(net, 3, 512, idx = 64)  #90
        net65 = self.conv_layer(net, 1, 256, idx = 65)  #91
        net = self.conv_layer(net65, 3, 512, idx = 66)  #92
        net67 = self.conv_layer(net, 1, self.output_size, batch_norm = False, activation = False, idx = 67)  #93

        net = self.conv_layer(net65, 1, 128, idx = 68)  #96
        net = self.upsampling(net)  #97
        net = tf.concat([net, net36], axis = 3) #98

        net = self.conv_layer(net, 1, 128, idx = 69)  #99
        net = self.conv_layer(net, 3, 256, idx = 70)  #100
        net = self.conv_layer(net, 1, 128, idx = 71)  #101
        net = self.conv_layer(net, 3, 256, idx = 72)  #102
        net = self.conv_layer(net, 1, 128, idx = 73)  #103
        net = self.conv_layer(net, 3, 256, idx = 74)  #104
        net = self.conv_layer(net, 1, self.output_size, batch_norm = False, activation = False, idx = 75)  #105

        return [net59, net67, net]


    def conv_layer(self, input, filter, size, stride = 1, batch_norm = True, activation = True, idx = 0):
        with tf.variable_scope('convolutional'+str(idx)):
            channel = int(input.shape[3])
            weight = tf.Variable(tf.truncated_normal([filter, filter, channel, size], stddev=0.1), name='weights')
            biases = tf.Variable(tf.constant(0.1, shape=[size]), name='biases')
            conv = tf.nn.conv2d(input, weight, strides=[1, stride, stride, 1], padding='SAME')

            if batch_norm:
                with tf.variable_scope('BatchNorm'):
                    depth = conv.shape[3]
                    betas = tf.Variable(tf.ones([depth, ], dtype='float32'), name='beta')
                    shift = tf.Variable(tf.zeros([depth, ], dtype='float32'), name='shift')
                    mean = tf.Variable(tf.ones([depth, ], dtype='float32'), name='moving_mean')
                    variance = tf.Variable(tf.ones([depth, ], dtype='float32'), name='moving_variance')

                    conv = tf.nn.batch_normalization(conv, mean, variance, shift, betas, 1e-05, name='BatchNorm')

            conv = tf.add(conv, biases)

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


    def loss_layer(self, predict, labels, anchor):
        grid_h, grid_w = predict.shape[1:3]
        grid_h, grid_w = int(grid_h), int(grid_w)
        col = np.reshape(np.tile(np.arange(0, grid_w), grid_w), [grid_h, grid_w, 1])
        row = np.reshape(np.tile(np.reshape(np.arange(0, grid_h), [-1, 1]), grid_h), [1, grid_h, grid_w, 1])
        col = np.tile(col, (self.batch_size, 1, 1, 3))
        row = np.tile(row, (self.batch_size, 1, 1, 3))

        predict = tf.reshape(predict, [self.batch_size, grid_h, grid_w, self.box_per_cell, self.num_class+5])
        box_coordinate = tf.reshape(predict[:, :, :, :, :4], [self.batch_size, grid_h, grid_w, self.box_per_cell, 4])
        box_confidence = tf.reshape(predict[:, :, :, :, 4], [self.batch_size, grid_h, grid_w, self.box_per_cell, 1])
        box_classes = tf.reshape(predict[:, :, :, :, 5:], [self.batch_size, grid_h, grid_w, self.box_per_cell, self.num_class])

        boxes1 = tf.stack([(1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 0])) + col) / grid_w,
                           (1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 1])) + row) / grid_h,
                           tf.exp(box_coordinate[:, :, :, :, 2]) * anchor[:, 0] / self.image_size,
                           tf.exp(box_coordinate[:, :, :, :, 3]) * anchor[:, 1] / self.image_size])
        box_coor_trans = tf.transpose(boxes1, (1, 2, 3, 4, 0)) * self.image_size
        box_confidence = 1.0 / (1.0 + tf.exp(-1.0 * box_confidence))
        #box_confidence = tf.tile(tf.expand_dims(box_confidence, axis = 4), (1, 1, 1, 1, self.num_class))
        box_classes = 1.0 / (1.0 + tf.exp(-1.0 * box_classes))

        response = tf.reshape(labels[:, :, :, :, 0], [self.batch_size, grid_h, grid_w, self.box_per_cell])
        boxes = tf.reshape(labels[:, :, :, :, 1:5], [self.batch_size, grid_h, grid_w, self.box_per_cell, 4])
        classes = tf.reshape(labels[:, :, :, :, 5:], [self.batch_size, grid_h, grid_w, self.box_per_cell, self.num_class])

        iou = self.calc_iou(box_coor_trans, boxes)
        best_box = tf.to_float(tf.equal(iou, tf.reduce_max(iou, [2], True)))
        confs = tf.expand_dims(best_box * response, axis = 4)

        conid = self.noobject_scale * (1.0 - confs) + self.object_scale * confs
        cooid = self.coordinate_scale * tf.concat(4 * [confs], 4)
        proid = self.class_scale * tf.concat(self.num_class * [confs], 4)

        coo_loss = cooid * tf.square(box_coor_trans - boxes)
        con_loss = conid * tf.square(box_confidence - confs)
        pro_loss = proid * tf.square(box_classes - classes)

        loss = tf.concat([coo_loss, con_loss, pro_loss], axis=4)
        loss = tf.reshape(loss, [self.batch_size, grid_h * grid_w * self.box_per_cell * (self.num_class + 5)])
        loss = 0.5 * tf.reduce_mean(tf.reduce_mean(loss, axis=1))

        return loss

    def calc_iou(self, boxes1, boxes2):
        boxes1_square = boxes1[:, :, :, :, 2] * boxes1[:, :, :, :, 3]
        box = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] * 0.5,
                        boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] * 0.5,
                        boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] * 0.5,
                        boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] * 0.5])
        boxes1 = tf.transpose(box, (1, 2, 3, 4, 0))

        boxes2_square = boxes2[:, :, :, :, 2] * boxes2[:, :, :, :, 3]
        box = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] * 0.5,
                        boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] * 0.5,
                        boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] * 0.5,
                        boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] * 0.5])
        boxes2 = tf.transpose(box, (1, 2, 3, 4, 0))

        left_up = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        right_down = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        intersection = tf.maximum(right_down - left_up, 0.0)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        union_square = boxes1_square + boxes2_square - inter_square

        return tf.clip_by_value(1.0 * inter_square / union_square, 0.0, 1.0)

    def calc_loss(self, output, labels):
        total_loss = 0
        for i in range(0, 3):
            loss = self.loss_layer(output[i], labels[i], self.anchor[6-3*i:9-3*i])
            total_loss += loss
        return total_loss/3.0