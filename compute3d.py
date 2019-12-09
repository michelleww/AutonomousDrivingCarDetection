import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot as plt
from disparity import *
from depth import *

def get_3d_locations(img_left, img_right, calib_path):

    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    # get disparity map
    disparity = get_disparity(img_left_gray, img_right_gray)

    # gaussian blur
    disparity = sp.ndimage.gaussian_filter(disparity, sigma=1)

    # get depth map and basic calibaration data
    f,T,px,py,depth = compute_depth(calib_path, disparity)

    height, width = depth.shape
    image_3d = []

    # calculate 3D locations for each pixel, append to a list
    for i in range(height):
        for j in range(width):
            Z = depth[i,j]
            image_3d.append([(i-py)*Z/f, (j-px)*Z/f, Z])

    return image_3d

if __name__ == "__main__":
    im_left_path = "train/image_left/um_000000.jpg"
    im_right_path = "train/image_right/um_000000.jpg"

    im_left = cv2.imread(im_left_path)
    im_left = cv2.cvtColor(im_left, cv2.COLOR_BGR2RGB)

    im_right = cv2.imread(im_right_path)
    im_right = cv2.cvtColor(im_right, cv2.COLOR_BGR2RGB)

    # compute disparity map
    disparity = compute_disparity(im_left, im_right)

    calib_file_dir = 'train/calib/um_000000.txt'
    f,T,px,py,depth = compute_depth(calib_file_dir, disparity)

    get_3d_locations(im_left, im_right, calib_file_dir)
