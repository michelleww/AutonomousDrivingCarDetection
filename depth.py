import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot as plt
from disparity import *
from skimage import io

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
    # print(disparity.as)
    print(f, T, px,py)
    depth = np.divide(f*T,disparity,where=disparity!=0)
    print(depth)
    
    return f,T,px,py,depth

if __name__ == "__main__":
    # choose two stereo images
    im_left_path = "train/image_left/umm_000011.jpg"
    im_right_path = "train/image_right/umm_000011.jpg"
    # read images
    # im_left = io.imread(im_left_path)
    im_left = cv2.imread(im_left_path, 0)
    # im_left = cv2.cvtColor(im_left, cv2.COLOR_BGR2RGB)
    # im_right = io.imread(im_right_path)
    im_right = cv2.imread(im_right_path, 0)
    # im_right = cv2.cvtColor(im_right, cv2.COLOR_BGR2RGB)

    # compute disparity map
    disparity = disparity_2(im_left, im_right)
    # disparity = cv2.cvtColor(im_left, cv2.COLOR_BGR2GRAY)


    calib_file_dir = 'train/calib/umm_000011.txt'
    f,T,px,py,depth = compute_depth(calib_file_dir, disparity)
    # depth = depth.astype(np.int8)

    # print(disparity)
    # print(depth)
    # plot depth map
    fig=plt.figure(figsize=(10, 10))


    # io.imshow(depth,cmap = 'gray')
    # io.show()
    plt.imshow(depth)
    plt.show()