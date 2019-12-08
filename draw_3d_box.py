from sklearn import linear_model, datasets
from skimage.measure import LineModelND, ransac
import scipy.linalg
import open3d
import random
from numpy.linalg import svd
import math
from car_detection import object_detection_api
from draw_car_viewPoint import create_output_directory
from plane2 import main
import cv2

def draw_3d_boxes(boxes, recursive_glob_path, img_path, svm_path):
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
        p2_x =  int(round(center_point_x + length * math.cos(angle * np.pi / 180.0)))
        p2_y =  int(round(center_point_y + length * math.sin(angle * np.pi / 180.0)))

        # Draw Rectangle with the coordinates and put detection class on the left top cornor
        cv2.rectangle(img, box[0], box[1],color=(0, 255, 0), thickness=1)
        cv2.putText(img,'Car', box[0],  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),thickness=2)
        cv2.arrowedLine(img, (int(center_point_x), int(center_point_y)), (p2_x, p2_y), (210, 142, 40), thickness=3)
        cv2.putText(img,str(angle), (int(center_point_x), int(center_point_y)),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),thickness=2)
    cv2.imwrite('./detected_car/results-dets/' + test_img[(test_img.index('/u')+1):], img)
    return img


if __name__ == "__main__":
    left_im_dir = "train/image_left/um_000088.jpg"
    right_im_dir = "train/image_right/um_000088.jpg"
    calib_dir = "train/calib/um_000088.txt"
    create_output_directory()
    cars = object_detection_api(left_im_dir)
        # left image
    test_left = cv2.imread(left_im_dir)

    # right image
    test_right = cv2.imread(right_im_dir)
    pcd, data, plane_model = main(test_left, test_right, calib_dir)
    
    open3d.visualization.draw_geometries(pcd)

