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
import open3d
import random

def prepare_3d_points(gt_mask, map_3ds):
    points = []

    road_points = np.nonzero(gt_mask)
    gt_mask = np.array(gt_mask)
    height, width = gt_mask.shape

    size = len(road_points[0])
    for idx in range(size):
        i = road_points[0][idx]
        j = road_points[1][idx]
        points.append(map_3ds[i*width+j])

    return points


def v2(data, colors):

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(data) 
    pcd.colors = open3d.utility.Vector3dVector(colors) 

    # print("Downsample the point cloud with a voxel of 0.05")
    # downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    # open3d.visualization.draw_geometries([downpcd])

    # print("Recompute the normal of the downsampled point cloud")
    # downpcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
    #     radius=0.1, max_nn=30))
    open3d.visualization.draw_geometries([pcd])


def fit_plane_LSE(points):
    # points: Nx4 homogeneous 3d points
    # return: 1d array of four elements [a, b, c, d] of
    # ax+by+cz+d = 0
    assert points.shape[0] >= 3 # at least 3 points needed
    U, S, Vt = svd(points)
    null_space = Vt[-1, :]
    return null_space

def fit_plane_ransac(road_pixels, sample_size, max_iterations, inlier_thresh=0.03, random_seed=None):
    max_inlier_num = -1
    best_model = None
    random.seed(random_seed)

    for i in range(max_iterations):
        sample = np.array(random.sample(road_pixels, sample_size))

        # A = np.c_[sample[:,0], sample[:,1], np.ones(sample.shape[0])]

        # get plane coefficient
        coefficients = fit_plane_LSE(np.array(sample))
        print(coefficients)
        # coefficients,_,_,_ = scipy.linalg.lstsq(A, sample[:,2]) 

        # get point distances from the estimated plane
        dists = np.abs(sample @ coefficients) / np.sqrt(coefficients[0]**2 + coefficients[1]**2 + coefficients[2]**2)
        
        num_inliers = len(np.where(dists < inlier_thresh)[0])
        if num_inliers > max_inlier_num:
            max_inlier_num = num_inliers
            best_model = coefficients

    return best_model

if __name__ == "__main__":
    # dir
    left_im_dir = "train/image_left/umm_000011.jpg"
    right_im_dir = "train/image_right/umm_000011.jpg"
    calib_dir = "train/calib/umm_000011.txt"

    # left image
    test_left = cv2.imread(left_im_dir)

    # right image
    test_right = cv2.imread(right_im_dir)

    data = get_3d_locations(test_left, test_right,calib_dir)

    # data = (np.array(data)*255).astype(np.uint8)


    # use left image as the test image, only make prediction on the left image
    # gt=[]
    # if os.path.isfile('gt.npy'):
    #     gt = np.load('gt.npy', allow_pickle=True)
    # else:
    #     predictions_l, img_seg_l = test_single_data(left_im_dir, calib_dir)
    #     gt_mask_left = get_segmentation(predictions_l, test_left, img_seg_l)
    
    #     np.save('gt', gt_mask_left)

    # road_3ds = prepare_3d_points(gt, data)

    # sample_size = int(len(road_3ds)*0.65)
    # best_model = fit_plane_ransac(road_3ds, sample_size, 2, random_seed=7)

    # print(best_model)

    # road_3ds = np.array(road_3ds)
    # x = np.arange(np.min(road_3ds[:, 0]), np.max(road_3ds[:, 0]), 0.1)
    # z = np.arange(np.min(road_3ds[:, 2]), np.max(road_3ds[:, 2]), 0.1)

    # xx, zz = np.meshgrid(x, z)

    # yy = (-best_model[0]*xx -best_model[2]*zz -best_model[3])/best_model[1]

    # plane = np.c_[xx, yy, zz]

    # print(plane)

    height, width, depth = test_left.shape
    test_left = cv2.cvtColor(test_left, cv2.COLOR_BGR2RGB)
    test_left = test_left/255
    data_color = test_left.reshape(height*width, 3)

    v2(data, data_color)


