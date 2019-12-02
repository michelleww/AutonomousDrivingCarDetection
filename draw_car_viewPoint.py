from viewPoint_detection import get_resize_info
import cv2
import numpy as np
import math
from skimage.feature import hog 
import joblib
import matplotlib.pyplot as plt

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

def draw_box_and_arrow(corners, recursive_glob_path, img_path, svm_path):
    # resize_info = get_resize_info(recursive_glob_path)
    img_gray = cv2.imread(img_path, 0)
    left, top, right, bottom = float(corners[0]), float(corners[1]), float(corners[2]), float(corners[3])
    img_seg = img_gray[int(round(top)):int(round(bottom))+1, int(round(left)):int(round(right))+1]
    # angle = int(test_single_image(img_seg, resize_info, svm_path))

    angle = int('-120')

    center_point_x = (right + left)/2
    center_point_y = (bottom + top)/2
    length  = (bottom - top)/2 - 2
    p2_x =  int(round(center_point_x + length * math.cos(angle * np.pi / 180.0)))
    p2_y =  int(round(center_point_y + length * math.sin(angle * np.pi / 180.0)))

    img = cv2.imread(img_path)
    cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0,0,255),3)
    cv2.arrowedLine(img, (int(center_point_x), int(center_point_y)), (p2_x, p2_y), (0,0,255), thickness=3)
    plt.imshow(img[:,:,::-1])
    plt.show()
    return img

if __name__ == '__main__':
    corners_1 = [384.19, 184.33, 505.39, 264.59]
    corners_2 = [253.83, 180.17, 371.79, 244.71]
    draw_box_and_arrow(corners_1, 'angle_classification/*/*.jpg', '000110.jpg', 'svm.pkl')
    draw_box_and_arrow(corners_2, 'angle_classification/*/*.jpg', '000161.jpg', 'svm.pkl')

