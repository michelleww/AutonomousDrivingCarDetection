import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot as plt
import io



def compute_disparity(im_left, im_right):
    """ cv2.StereoSGBM_create:
        minDisparity: Minimum possible disparity value.
        numDisparities: Maximum disparity minus minimum disparity.  This parameter must be divisible by 16 and >= 0.
        blockSize: Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        uniquenessRatio: Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. (Usually 5-15)
        speckleWindowSize: Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleRange: Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
    """
    stereo = cv2.StereoSGBM_create(minDisparity=0,numDisparities=80,blockSize=11,uniquenessRatio=10,speckleWindowSize=150,speckleRange=2)
    return stereo.compute(im_left, im_right)

def disparity_own(im_left, im_right):
    pass


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
    disparity = compute_disparity(im_left, im_right)
    # show the disparity map
    plt.imshow(disparity)
    plt.show()


