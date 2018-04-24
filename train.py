# -*- coding:utf-8 -*-
#
# Written by leeyoshinari
#
#

import tensorflow as tf
import numpy as np
import argparse
import datetime
import time
import os

import yolo3.config as cfg
from yolo3.yolo_v3 import yolo_v3

class Train(object):
    def __init__(self, yolo3):
        self.yolov3 = yolo3
        self.num_classes = len(cfg.CLASSES)
        self.init_learn_rate = cfg.LEARNING_RATE
        self.max_step = cfg.MAX_STEP
        self.saver_iter = cfg.SAVE_ITER
        self.summary_iter = cfg.SUMMARY_ITER
        self.output_dir = os.path.join(cfg.DATA_DIR, cfg.WEIGHTS_FILE)
        weights_file = os.path.join(self.output_dir, cfg.WEIGHTS)

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore)
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir)

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.init_learn_rate, self.global_step, cfg.DECAY_STEP, cfg.DECAY_RATE, name='learning_rate')
        self.opertimer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.yolov3.total_loss, global_step = self.global_step)

        self.average_op = tf.train.ExponentialMovingAverage(0.999).apply(tf.trainable_variables())
        with tf.control_dependencies([self.opertimer]):
            self.train_op = tf.group(self.average_op)

        config = tf.ConfigProto(gpu_options = tf.GPUOptions())
        self.sess = tf.Session(config = config)
        self.sess.run(tf.global_variables_initializer())

        print('Restore weights from: ', weights_file)
        self.saver.restore(self.sess, weights_file)
        self.writer.add_graph(self.ses.graph)
        #self.saver = tf.train.Saver(self.variable_to_restore)

    def train(self):
        labels_train = None
        labels_test = None

        group_num = 10
        init_time = time.time()

        for step in range(0, self.max_step + 1):
            images, labels = None, None
            feed_dict = {self.yolov3.images: images, self.yolov3.labels: labels}

            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:
                    summary_, loss, _ = self.sess.run([self.summary_op, self.yolov3.total_loss, self.train_op], feed_dict = feed_dict)

                    sum_loss = 0
                    for i in range(group_num):
                        images_t, labels_t = None, None
                        feed_dict_t = {self.yolov3.images: images_t, self.yolov3.labels: labels_t}
                        loss_t = self.sess.run(self.yolov3.total_loss, feed_dict = feed_dict_t)
                        sum_loss += loss_t
            
                    log_str = ('{} Epoch:{}, step:{}, train_loss:{:.4f},test_loss:{:.4f}, remain:{}').format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), None, step, loss, sum_loss/group_num, self.remain(step, init_time))
                    print(log_str)

                else:
                    summary_, _ = self.sess.run([self.summary_op, self.train_op], feed_dict = feed_dict)
                
                self.writer.add_summary(summary_, step)

            else:
                self.sess.run(self.train_op, feed_dict = feed_dict)

            if step % self.saver_iter == 0:
                self.saver.save(self.sess, self.output_dir + 'yolo_v3.ckpt', global_step = step)


    def remain(self, step, start):
        if step == 0:
            remain_time = 0
        else:
            remain_time = (time.time() - start) * (self.max_step - step) / step

        return str(datetime.timedelta(seconds = int(remain_time)))


def main()：
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'yolo_v3.ckpt', type = str)
    parser.add_argument('--gpu', default = '', type = str)
    args = parser.parse_args()

    if args.weights is not None:
        cfg.WEIGHTS = args.weights
    
    if args.gpu is not None:
        cfg.GPU = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo3 = yolo_v3()

    train = Train(yolo3)

    print('start training ....')
    train.train()
    print('successful training.')

if __name__ == '__main__':
    main()