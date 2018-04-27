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
        self.anchor = np.reshape(cfg.ANCHOR, [-1,2])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print('Restore weights from: ' + weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, weights_file)

    
    def detect(self, image):
        img_h, img_w, _ = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0# * 2.0 - 1.0
        image = np.reshape(image, [1, self.image_size, self.image_size, 3])
        output = self.sess.run(self.yolov3.logits, feed_dict = {self.yolov3.images: image})

        boxes, probs, class_ind = [], [], []
        for i in range(0, 3):
            _boxes, _probs, _classes = self.calc_output(output[i], self.anchor[6-3*i: 9-3*i])
            boxes.append(_boxes)
            probs.append(_probs)
            class_ind.append(_classes)

        for i in range(len(boxes)):
            boxes1 = boxes[i] * [img_w, img_h, img_w, img_h]
            for j in range(len(boxes1)):
                boxes[i][j][0] -= boxes1[j][2] * 0.5
                boxes[i][j][1] -= boxes1[j][3] * 0.5
                boxes[i][j][0] += boxes1[j][2] * 0.5
                boxes[i][j][1] += boxes1[j][3] * 0.5

        print(np.shape(boxes))
        print(probs)
        print(class_ind)
        result = self.calc_object(boxes, probs, class_ind)

        '''for i in range(len(result)):
            result[i][1] *= (1.0 * img_w)
            result[i][2] *= (1.0 * img_h)
            result[i][3] *= (1.0 * img_w)
            result[i][4] *= (1.0 * img_h)'''

        return result


    def calc_output(self, output, anchor):
        grid_h, grid_w = output.shape[1:3]
        output = np.reshape(output, [grid_h, grid_w, self.box_per_cell, self.num_classes+5])
        coordinate = np.reshape(output[:, :, :, :4], [grid_h, grid_w, self.box_per_cell, 4])
        confidence = np.reshape(output[:, :, :, 4], [grid_h, grid_w, self.box_per_cell])
        classes = np.reshape(output[:, :, :, 5:], [grid_h, grid_w, self.box_per_cell, self.num_classes])
        print(confidence[3, 3, 0])
        print(classes[3, 3, 0, :])

        col = np.reshape(np.tile(np.arange(0, grid_w), grid_w), [grid_h, grid_w, 1])
        row = np.reshape(np.tile(np.reshape(np.arange(0, grid_h), [-1, 1]), grid_h), [grid_h, grid_w, 1])
        col = np.tile(col, (1, 1, 3))
        row = np.tile(row, (1, 1, 3))

        boxes1 = np.stack([(1.0 / (1.0 + np.exp(-1.0 * coordinate[:, :, :, 0])) + col) / grid_w,
                           (1.0 / (1.0 + np.exp(-1.0 * coordinate[:, :, :, 1])) + row) / grid_h,
                           np.exp(coordinate[:, :, :, 2]) * anchor[:, 0] / self.image_size,
                           np.exp(coordinate[:, :, :, 3]) * anchor[:, 1] / self.image_size])
        coordinate = np.transpose(boxes1, (1, 2, 3, 0))
        confidence = 1.0 / (1.0 + np.exp(-1.0 * confidence))
        confidence = np.tile(np.expand_dims(confidence, axis = 3), (1, 1, 1, self.num_classes))
        classes = 1.0 / (1.0 + np.exp(-1.0 * classes))
        #print(confidence[3, 3, 0, :])
        #print(classes[3, 3, 0, :])

        box_scores = confidence * classes
        #print(box_scores[3,3,0,:])

        box_classes = np.argmax(box_scores, axis = -1)
        #print(np.shape(box_classes))
        box_score_max = np.max(box_scores, axis = -1)
        #print(np.shape(box_score_max))
        filter_index = np.where(box_score_max >= self.threshold)
        #print(filter_index)

        boxes_filter = coordinate[filter_index]
        probs_filter = box_score_max[filter_index]
        class_filter = box_classes[filter_index]

        return boxes_filter, probs_filter, class_filter


    def calc_object(self, boxes, probs, class_idx):
        boxes_filter = np.concatenate(boxes)
        probs_filter = np.concatenate(probs)
        class_filter = np.concatenate(class_idx)

        '''boxes = np.concatenate(boxes)
        probs = np.concatenate(probs)
        class_idx = np.concatenate(class_idx)

        _boxes, _probs, _class_idx = [], [], []
        for c in set(class_idx):
            index = np.where(class_idx == c)
            boxes_filter = boxes[index]
            probs_filter = probs[index]
            class_filter = class_idx[index]
            
            sort_num = np.array(np.argsort(probs_filter))[::-1]
            boxes_filter = boxes_filter[sort_num]
            probs_filter = probs_filter[sort_num]
            class_filter = class_filter[sort_num]

            for i in range(len(probs_filter)):
                if probs_filter[i] == 0:
                    continue
                for j in range(i+1, len(probs_filter)):
                    if self.calc_iou(boxes_filter[i], boxes_filter[j]) > 0.5:
                        probs_filter[j] = 0
        '''
        sort_num = np.array(np.argsort(probs_filter))[::-1]
        boxes_filter = boxes_filter[sort_num]
        probs_filter = probs_filter[sort_num]
        class_filter = class_filter[sort_num]

        for i in range(len(probs_filter)):
            if probs_filter[i] == 0:
                continue
            for j in range(i + 1, len(probs_filter)):
                if self.calc_iou(boxes_filter[i], boxes_filter[j]) > 0.5:
                    probs_filter[j] = 0
        filter_probs = np.array(probs_filter > 0, dtype='bool')
        boxes_filter = boxes_filter[filter_probs]
        probs_filter = probs_filter[filter_probs]
        class_filter = class_filter[filter_probs]
        '''_boxes.append(boxes_filter)
            _probs.append(probs_filter)
            _class_idx.append(class_filter)

        boxes_filter = np.concatenate(_boxes)
        probs_filter = np.concatenate(_probs)
        class_filter = np.concatenate(_class_idx)'''
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
    parser.add_argument('--weights', default = 'yolov3.ckpt', type = str)
    parser.add_argument('--weights_dir', default = 'output', type = str)
    parser.add_argument('--data_dir', default = 'data', type = str)
    parser.add_argument('--gpu', default = '0', type = str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    weights_file = os.path.join(args.data_dir, args.weights_dir, args.weights)

    yolov3 = yolo_v3()
    detect = detector(yolov3, weights_file)

    image_name = './test/01.jpg'
    detect.image_detect(image_name)

if __name__ == '__main__':
    main()