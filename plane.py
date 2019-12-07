import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot as plt
from disparity import *
from skimage import io
from disparity import *
from depth import *
from compute3d import *
from road_detection import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model, datasets
from skimage.measure import LineModelND, ransac
import scipy.linalg


def fit_plane(mask_2d):
    pass


def plot_3d_cloud_pt():
    pass


def visualize_fitted_plane(data, order=1):
    """ data: 3D points with shape: (x,3)
        order:  1: linear, 2: quadratic
    """
    # regular grid covering the domain of the data
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    XX = X.flatten()
    YY = Y.flatten()

    #order = 1    # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
        
        # evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]
        
        # or expressed using matrix/vector product
        #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
        
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

    # plot points and fitted surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()



if __name__ == "__main__":
    # dir
    left_im_dir = ".\\train\\image_left\\um_000000.jpg"
    right_im_dir = ".\\train\\image_right\\um_000000.jpg"
    calib_dir = ".\\train\\calib\\um_000001.txt"

    # left
    test_left = cv2.imread(left_im_dir)
    predictions_l, img_seg_l = test_single_data(left_im_dir, calib_dir)
    gt_mask_left = get_segmentation(predictions_l, test_left, img_seg_l)

    # right
    test_right = cv2.imread(right_im_dir)
    predictions_r, img_seg_r = test_single_data(right_im_dir, calib_dir)
    gt_mask_right = get_segmentation(predictions_r, test_right, img_seg_r)

    # from float64 to float32: to get depth of 8 image
    gt_mask_left = (gt_mask_left*255).round().astype(np.uint8)
    gt_mask_right = (gt_mask_right*255).round().astype(np.uint8)
    # print("Shape: " + str(gt_mask_left.shape, gt_mask_right.shape))
    # print("Shape: " + str(gt_mask_left, gt_mask_right.shape))
    # compute disparity
    disparity = compute_disparity(gt_mask_left, gt_mask_right)
    # plt.imshow(disparity)
    # plt.show()
    # compute depth
    f, T, px, py, depth = compute_depth(calib_dir, disparity)
    depth = (depth*255).round().astype(np.uint8)
    print(depth)
    print(depth.shape)
    # compute 3D location
    #location_3d_ = compute_3d(depth, px, py, f)
    # print(location_3d_)

    # float64
    # X = ((location_3d_[:,:,0])*255).round().astype(np.uint8).flatten()
    # Y = ((location_3d_[:,:,1])*255).round().astype(np.uint8).flatten()
    # Z = ((location_3d_[:,:,2])*255).round().astype(np.uint8).flatten()
    # data = np.c_[X,Y,Z]
    # print(data.shape)

    visualize_fitted_plane(depth)

    



    # # RANSAC
    # # ransac = linear_model.RANSACRegressor()
    # # ransac.fit()
    # model_robust, inliers = ransac(temp, LineModelND, min_samples=2, residual_threshold=1, max_trials=1000)
    # outliers = inliers == False

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(temp[inliers][:, 0], temp[inliers][:, 1], temp[inliers][:, 2], c='b', marker='o', label='Inlier data')
    # ax.scatter(temp[outliers][:, 0], temp[outliers][:, 1], temp[outliers][:, 2], c='r', marker='o', label='Outlier data')
    # ax.legend(loc='lower left')
    # plt.show()




    # fig=plt.figure(dpi=120)
    # ax=fig.add_subplot(111,projection='3d')
    # plt.title('point cloud')
    # ax.scatter(X,Y,Z,c='b',marker='.',s=2,linewidth=0,alpha=1,cmap='spectral')

    # # ax.set_facecolor((0,0,0))
    # # ax.axis('scaled')          
    # # ax.xaxis.set_visible(False) 
    # # ax.yaxis.set_visible(False) 
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()




    #visualize_segementation(test, gt_mask_1)
    #print(gt_mask_1)

    # fit a plane in 3D to the road pixels by using the depth of the pixels
    #fit_plane(gt_mask_1)


    
