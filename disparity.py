import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot as plt
import io


def disparity(im_left, im_right):
    stereo = cv2.StereoSGBM_create(minDisparity=0,numDisparities=80,blockSize=11,uniquenessRatio=10,speckleWindowSize=100,speckleRange=2)
    return stereo.compute(im_left, im_right)

if __name__ == "__main__":
    # read two stereo images
    im_left_path = "train/image_left/um_000000.jpg"
    im_left = plt.imread(im_left_path)
    #plt.imshow(im_left)
    #plt.show()
    im_right_path = "train/image_right/um_000000.jpg"
    im_right = plt.imread(im_right_path,0)
    #plt.imshow(im_right)
    #plt.imshow()
    # compute disparity
    d = disparity(im_left, im_right)
    # show the disparity map
    plt.imshow(d)
    plt.show()
    


