import numpy as np
import cv2 as cv2
import os
import pathlib
import scipy.io
from matplotlib import pyplot as plt

def create_output_directory():
    sub_dir = ['00', '30', '60', '90', '120', '150', '180', '210', '240', '270', '300', '330']
    for i in sub_dir:
        test_path = 'angle_classification/' + i
        pathlib.Path(test_path).mkdir(parents=True, exist_ok=True) 

def get_car_segmentation(input_img_path):

    files = get_files(input_img_path)

    for idx in range(len(files)):
        filename = files[idx]
        file_path = input_img_path + '/' + filename + '.jpg'
        print('processing image: ' + file_path)
        print('file count is: ' + str(idx+1))
        img = cv2.imread(file_path)
        mat_data = scipy.io.loadmat('train_angle/labels/' + filename + '.mat')
        detections = mat_data['annotation'][0][0]

        classes = detections[0][0]
        boxes = detections[3]
        truncated = detections[4][0]
        angles = detections[7]
        occluded = detections[8][0]
        # setting image segment starting from 1
        acc_00, acc_30, acc_60, acc_90, acc_120, acc_150, acc_180, acc_210, acc_240, acc_270, acc_300, acc_330 = 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

        for idx in range(len(classes)):
            if classes[idx][0] == 'Car' and truncated[idx][0] < 0.3 and occluded[idx][0] < 2:
                angle =  angles[idx][0]
                box = boxes[idx]
                xleft, ytop, width, height = int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))
                segment = img[ytop:(ytop+height+1), xleft:(xleft+width+1), :]
                if 00 <= angle < 30:
                    cv2.imwrite('./angle_classification/00/{}_{}_{}.jpg'.format(filename, 00, acc_00), segment)
                    acc_00 += 1
                elif 30 <= angle < 60:
                    cv2.imwrite('./angle_classification/30/{}_{}_{}.jpg'.format(filename, 30, acc_30), segment)
                    acc_30 += 1
                elif 60 <= angle < 90:
                    cv2.imwrite('./angle_classification/60/{}_{}_{}.jpg'.format(filename, 60, acc_60), segment)
                    acc_60 += 1
                elif 90 <= angle < 120:
                    cv2.imwrite('./angle_classification/90/{}_{}_{}.jpg'.format(filename, 90, acc_90), segment)
                    acc_90 += 1
                elif 120 <= angle < 150:
                    cv2.imwrite('./angle_classification/120/{}_{}_{}.jpg'.format(filename, 120, acc_120), segment)
                    acc_120 += 1
                elif 150 <= angle < 180:
                    cv2.imwrite('./angle_classification/150/{}_{}_{}.jpg'.format(filename, 150, acc_150), segment)
                    acc_150 += 1
                elif 180 <= angle < 210:
                    cv2.imwrite('./angle_classification/180/{}_{}_{}.jpg'.format(filename, 180, acc_180), segment)
                    acc_180 += 1
                elif 210 <= angle < 240:
                    cv2.imwrite('./angle_classification/210/{}_{}_{}.jpg'.format(filename, 210, acc_210), segment)
                    acc_210 += 1
                elif 240 <= angle < 270:
                    cv2.imwrite('./angle_classification/240/{}_{}_{}.jpg'.format(filename, 240, acc_240), segment)
                    acc_240 += 1
                elif 270 <= angle < 300:
                    cv2.imwrite('./angle_classification/270/{}_{}_{}.jpg'.format(filename, 270, acc_270), segment)
                    acc_270 += 1
                elif 300 <= angle < 330:
                    cv2.imwrite('./angle_classification/300/{}_{}_{}.jpg'.format(filename, 300, acc_300), segment)
                    acc_300 += 1
                elif 330 <= angle <= 360:
                    cv2.imwrite('./angle_classification/330/{}_{}_{}.jpg'.format(filename, 330, acc_330), segment)
                    acc_330 += 1                

def get_files(path):
    names = os.listdir(path)
    return_list = [name.split('.')[0] for name in names if name.endswith('.jpg')]
    return_list.sort()
    return return_list

if __name__ == '__main__':
    create_output_directory()
    training_directory = './train_angle/image'
    get_car_segmentation(training_directory)
