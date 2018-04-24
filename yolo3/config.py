# -*- coding:utf-8 -*-

def get_classes_names():
    names = []
    with open('./yolo3/coco_names.txt') as f:
        for name in f.readlines():
            name = name[:-1]
            names.append(name)
    return names

DATA_DIR = 'data'
WEIGHTS_FILE = 'output'
WEIGHTS = 'yolo_V3.ckpt'

'''CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']'''
CLASSES = get_classes_names()

ANCHOR = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

IMAGE_SIZE = 416
BOX_PER_CELL = 3
BATCH_SIZE = 32

LEARNING_RATE = 0.0001

MAX_STEP = 10000
SAVE_ITER = 50
SUMMARY_ITER = 5

ALPHA = 0.1
DECAY_STEP = 30000
DECAY_RATE = 0.1

GPU = ''

THRESHOLD = 0.3
