# -*- coding:utf-8 -*-
#
# Written by leeyoshinari
#
# 2018-04-23

import tensorflow as tf
import numpy as np
import colorsys
import argparse
import cv2
import os

import yolo3.config as cfg
from yolo3.yolo_v3 import yolo_v3

class detector(object):
    def __init__(self, yolov3, weights_file):
        self.yolov3 = yolov3
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.box_per_cell = cfg.BOX_PER_CELL

        self.threshold = cfg.THRESHOLD
        self.anchor = np.shape(cfg.ANCHOR, [-1,2])

        self.offset = np.reshape(np.array([np.arange(grid_shape[1])] * grid_shape[2] * self.box_per_cell), [grid_shape[1], grid_shape[1], self.box_per_cell])
        self.offset = np.transpose(self.offset, (1, 2, 0))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print('Restore weights from: ' + weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, weights_file)

    
    def detect(self, image):
        img_h, img_w, _ = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0 * 2.0 - 1.0
        image = np.reshape(image, [1, self.image_size, self.image_size, 3])
        output = self.sess.run(self.yolov3.logits, feed_dict = {self.yolov3.images: image})

        for i in range(0, 3):
            _boxes, _probs = self.calc_output(output[i], self.anchor[6-3*i: 9-3*i])
            if i == 0:
                boxes, probs = _boxes, _probs
            else:
                boxes = np.concatenate([boxes, _boxes], axis = 0)
                probs = np.concatenate([probs, _probs], axis = 0)

        result = self.calc_object(boxes, probs)

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_w / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_w / self.image_size)

        return result


    def calc_output(self, output, anchor):
        grid_shape = output.shape
        output = np.reshape(output, [grid_shape[1], grid_shape[2], self.box_per_cell, self.num_class+5])
        coordinate = np.reshape(output[:, :, :, :4], [grid_shape[1], grid_shape[2], self.box_per_cell, 4])
        confidence = np.reshape(output[:, :, :, 4], [grid_shape[1], grid_shape[2], self.box_per_cell])
        classes = np.reshape(output[:, :, :, 5:], [ grid_shape[1], grid_shape[2], self.box_per_cell, self.num_class])

        boxes1 = np.stack([(1.0 / (1.0 + np.exp(-1.0 * coordinate[:, :, :, 0])) + self.offset) / grid_shape[1],
                           (1.0 / (1.0 + np.exp(-1.0 * coordinate[:, :, :, 1])) + np.transpose(self.offset, (1, 0, 2)))/ grid_shape[2],
                           np.exp(coordinate[:, :, :, 2]) * anchor[:, 0] / grid_shape[1],
                           np.exp(coordinate[:, :, :, 3]) * anchor[:, 1] / grid_shape[2]])
        coordinate = np.transpose(boxes1, (1, 2, 3, 0)) * self.image_size
        confidence = 1.0 / (1.0 + np.exp(-1.0 * box_confidence))
        confidence = np.tilt(np.expand_dims(confidence, axis = 3), (1, 1, 1, self.num_classes))
        classes = 1.0 / (1.0 + np.exp(-1.0 * box_classes))

        box_scores = confidence * classes

        coordinate = np.reshape(coordinate, [-1, 4])
        box_scores = np.reshape(box_scores, [-1, self.num_classes])

        return coordinate, box_scores


    def calc_object(self, boxes, probs):
        filter_probs = np.array(probs >= self.threshold, dtype = 'bool')
        filter_index = np.nonzero(filter_probs)

        boxes_filter = boxes[filter_index]
        probs_filter = probs[filter_index]
        class_filter = np.argmax(filter_probs, axis = 1)[filter_index]

        sort_num = np.array(np.argsort(probs_filter))[::-1]
        boxes_filter = boxes_filter[sort_num]
        probs_filter = probs_filter[sort_num]
        class_filter = class_filter[sort_num]

        for i in range(len(probs_filter)):
            if probs_filter[i] == 0:
                continue
            for j in range(i+1, len(probs_filter)):
                if self.calc_iou(boxes_filter[i], boxes_filter[j]) > 0.5
                    probs_filter[j] = 0
        
        filter_probs = np.array(probs_filter > 0, dtype = 'bool')
        boxes_filter = boxes_filter[filter_probs]
        probs_filter = probs_filter[filter_probs]
        class_filter = class_filter[filter_probs]

        results = []
        for i in range(len(probs_filter)):
            results.append(self.classes[class_filter[i], boxes_filter[i][0], boxes_filter[i][1], boxes_filter[i][2], boxes_filter[i][3], probs_filter[i]])

        return results


    def calc_iou(self, box1, box2):
        width = min(box1[0] + box1[2] * 0.5, box2[0] + box2[2] * 0.5) - max(box1[0] - box1[2] * 0.5, box2[0] - box2[2] * 0.5)
        height = min(box1[1] + box1[3] * 0.5, box2[1] + box2[3] * 0.5) - max(box1[1] - box1[3] * 0.5, box2[1] - box2[3] * 0.5)

        if width <= 0 or height <= 0:
            intersection = 0
        else:
            intersection = width * height

        return 1.0 * intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)


    def draw(self, image, result):
        img_h, img_w, _ = image.shape
        colors = self.random_colors(len(result))
        for i in range(len(result)):
            xmin = int(max(result[i][1] - 0.5 * result[i][3]), 0)
            ymin = int(max(result[i][2] - 0.5 * result[i][4]), 0)
            xmax = int(min(result[i][1] + 0.5 * result[i][3]), img_w)
            ymax = int(min(result[i][2] + 0.5 * result[i][4]), img_h)
            color = tuple([rgb * 255 for rgb in colors[i]])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(image, result[i][0] + ':%.2f' %result[i][5], (xmin+1, ymin+8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
            print(result[i][0], ':%.2f%%' %(result[i][5] * 100))


    def random_colors(self, num):
        hsv = [(i / num, 1, 1.0) for i in range(num)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        np.random.shuffle(colors)

        return colors


    def image_detect(self, image_name):
        image = cv2.imread(image_name)
        result = self.detect(image)
        self.draw(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'yolo_v3.ckpt', type = str)
    parser.add_argument('--weigths_dir', default = 'output', type = str)
    parser.add_argument('--data_dir', default = 'data', type = str)
    parser.add_argument('--gpu', default = '', type = str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    weights_file = os.path.join(args.data_dir, args.weights_dir, args.weights)

    yolov3 = yolo_v3()
    detect = detector(yolov3, weights_file)

    image_name = ''
    detect.image_detect(image_name)

if __name__ == '__main__':
    main()