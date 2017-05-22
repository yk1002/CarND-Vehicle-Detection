#!/usr/bin/env python

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from car_detector import car_detector
import sys
from moviepy.editor import VideoFileClip
from lesson_functions import apply_threshold, draw_labeled_bboxes, add_heat
import collections

class frame_tracker(object):
    def __init__(self, detector):
        self.detector_ = detector
        self.heatmaps_ = collections.deque(maxlen=10) # keeps track of 10 most recent heatmaps
        
    def process_frame(self, img):
        assert img.shape == (720, 1280, 3)

        # 1. get bounding boxes for the current frame
        ystart = 400
        yend = 656
        scale_list = [ 1.0, 1.5, 2.0 ]
        bbox_list = []
        for scale in scale_list:
            boxes = self.detector_.find_cars(img, ystart, yend, scale)
            bbox_list.extend(boxes)

        # 2. create heat map from the bounding boxes
        heatmap = add_heat(np.zeros_like(img[:,:,0]).astype(np.float), bbox_list)

        # 3. append the current heatmap to heatmaps
        self.heatmaps_.append(heatmap)

        # 4. accumlate the most recent heats
        heatmap = sum(self.heatmaps_)

        # 5. apply threshold to remove misidentifications
        heatmap = apply_threshold(heatmap, 5)

        # 6. find final boxes from accumulate heatmap using label function
        labels = label(np.clip(heatmap, 0, 255))
        draw_img = draw_labeled_bboxes(np.copy(img), labels)

        return draw_img

if __name__ == '__main__':
    car_detector_pickle_file_path = 'car_detector.p'
    try:
        with open(car_detector_pickle_file_path, 'rb') as fd:
            detector = pickle.load(fd)
        print('Loadedd detector from {}...'.format(car_detector_pickle_file_path))
    except (OSError, IOError) as e:
        cars = glob.glob('vehicles/**/*.png', recursive=True)
        notcars = glob.glob('non-vehicles/**/*.png', recursive=True)
        print('len(cars)={}'.format(len(cars)))
        print('len(notcars)={}'.format(len(notcars)))

        print('Building detector...')
        detector = car_detector()
        detector.build(cars, notcars)
        print('Done building detector')

        # save detector in file
        with open(car_detector_pickle_file_path, 'wb') as fd:
            pickle.dump(detector, fd)
        print('Saved detector to {}...'.format(car_detector_pickle_file_path))

    input_path = sys.argv[1]
    output_path = input_path.replace('.mp4', '_with_cars_detected.mp4')

    ft = frame_tracker(detector)
    VideoFileClip(input_path).fl_image(ft.process_frame).write_videofile(output_path, audio=False)
