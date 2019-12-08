import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot as plt
import io

# Reference: http://timosam.com/python_opencv_depthimage
def get_disparity(imgL, imgR):
    """ cv2.StereoSGBM_create:
        minDisparity: Minimum possible disparity value.
        numDisparities: Maximum disparity minus minimum disparity.  This parameter must be divisible by 16 and >= 0.
        blockSize: Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        uniquenessRatio: Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. (Usually 5-15)
        speckleWindowSize: Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleRange: Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
    """
    window_size = 3                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    # create a Stereo with SGBM algorithm
    left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=112,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=5,
    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters (image post processing), in order to smooth the disparity map for eliminate noise
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0
    
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    print('computing disparity...')
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    return filteredImg


if __name__ == "__main__":
    # read two stereo images
    im_left_path = "train/image_left/umm_000011.jpg"
    im_left = cv2.imread(im_left_path, 0)

    im_right_path = "train/image_right/umm_000011.jpg"
    im_right = cv2.imread(im_right_path,0)

    # compute disparity
    disparity2 = get_disparity(im_left, im_right)

    # show the disparity map
    plt.imshow(disparity2)
    plt.show()


