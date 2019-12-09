from viewPoint_detection import get_resize_info
import cv2
import numpy as np
import math
from skimage.feature import hog 
import joblib
import matplotlib.pyplot as plt
from car_detection import object_detection_api
import pathlib

def test_single_image(img_seg, resize, svm_path):
    # extraxct hog feature from the testing image
    img = cv2.resize(img_seg, (int(resize[1]*1.25), int(resize[0]*1.25)))
    features = hog(img)

    # restore the svm 
    svm = joblib.load(svm_path)

    # predit 
    prediction = svm.predict(features.reshape(1,-1))
    print('The prediction for the new data is: Class ' + str(prediction[0]))
    return prediction

def draw_boxes_and_arrow(boxes, recursive_glob_path, img_path, svm_path):
    resize_info = get_resize_info(recursive_glob_path)
    img_gray = cv2.imread(img_path, 0)
    img = cv2.imread(img_path)
    for box in boxes:
        (left, top), (right, bottom) = box[0], box[1]
        img_seg = img_gray[int(round(top)):int(round(bottom))+1, int(round(left)):int(round(right))+1]
        angle = int(test_single_image(img_seg, resize_info, svm_path))

        center_point_x = (right + left)/2
        center_point_y = (bottom + top)/2
        length  = (bottom - top)/2 + 15
        p2_x =  int(round(center_point_x + length * math.cos((angle+90)* np.pi / 180.0)))
        p2_y =  int(round(center_point_y + length * math.sin((angle+90) * np.pi / 180.0)))

        # Draw Rectangle with the coordinates and put detection class on the left top cornor
        cv2.rectangle(img, box[0], box[1],color=(0, 255, 0), thickness=1)
        cv2.putText(img,'Car', box[0],  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),thickness=2)
        cv2.arrowedLine(img, (int(center_point_x), int(center_point_y)), (p2_x, p2_y), (210, 142, 40), thickness=3)
        cv2.putText(img,str(angle), (int(center_point_x), int(center_point_y)),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),thickness=2)
    cv2.imwrite('./detected_car/results-dets/' + test_img[(test_img.index('/u')+1):], img)
    return img

def create_output_directory():
    pathlib.Path('detected_car/results-dets/').mkdir(parents=True, exist_ok=True) 

if __name__ == '__main__':
    
    create_output_directory()
    test_img = 'train/image_left/um_000011.jpg'
    cars = object_detection_api(test_img)
    draw_boxes_and_arrow(cars, 'angle_classification/*/*.jpg', test_img, 'svm.pkl')