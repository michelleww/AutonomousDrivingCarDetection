import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot as plt
from disparity import *
import scipy.ndimage

def compute_depth(calib_file, disparity):
    file = open(calib_file, 'r')
    lines =file.read().split('\n')
    p = {}
    for line in lines:
        if line:
            points = line.split(' ')
            p[points[0].strip(':')] = points[1:]
        
    f = float(p['P1'][0])
    T = -float(p['P1'][3])/f
    px = float(p['P1'][2])
    py = float(p['P1'][6])

    depth = np.divide(f*T,disparity,where=disparity!=0)
    return f,T,px,py,depth

if __name__ == "__main__":
    # choose two stereo images
    im_left_path = "train/image_left/umm_000011.jpg"
    im_right_path = "train/image_right/umm_000011.jpg"
    
    # read images
    im_left = cv2.imread(im_left_path, cv2.IMREAD_GRAYSCALE)
    im_right = cv2.imread(im_right_path, cv2.IMREAD_GRAYSCALE)

    # compute disparity map
    disparity = get_disparity(im_left, im_right)

    # gaussian blur
    disparity = sp.ndimage.gaussian_filter(disparity, sigma=2)


    calib_file_dir = 'train/calib/umm_000011.txt'
    f,T,px,py,depth = compute_depth(calib_file_dir, disparity)

    fig=plt.figure(figsize=(10, 10))
    plt.imshow(depth)
    plt.show()