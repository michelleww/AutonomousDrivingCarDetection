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
import math

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


def v2(data, colors, plane):

    pcd_img = open3d.geometry.PointCloud()
    pcd_img.points = open3d.utility.Vector3dVector(data) 
    pcd_img.colors = open3d.utility.Vector3dVector(colors) 

    plane_color = [[1,0,0] for point in plane]
    print(plane)
    pcd_plane = open3d.geometry.PointCloud()
    pcd_plane.points = open3d.utility.Vector3dVector(plane) 
    pcd_plane.colors = open3d.utility.Vector3dVector(plane_color) 

    # print("Downsample the point cloud with a voxel of 0.05")
    # downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    # open3d.visualization.draw_geometries([downpcd])

    # print("Recompute the normal of the downsampled point cloud")
    # downpcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
    #     radius=0.1, max_nn=30))

    return [pcd_img, pcd_plane]


def fitPlaneLTSQ(XYZ):
    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  #X
    G[:, 1] = XYZ[:, 1]  #Y
    Z = XYZ[:, 2]
    (a, b, c),resid,rank,s = np.linalg.lstsq(G, Z)
    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return (c, normal)

def fit_plane_ransac(road_pixels, sample_size, max_iterations, inlier_thresh=0.05, random_seed=100):
    max_inlier_num = -1
    best_model = None
    random.seed(random_seed)

    for i in range(max_iterations):
        sample = np.array(random.sample(road_pixels, sample_size))

        x, y, z = sample[:, 0], sample[:, 1], sample[:, 2]
        A = np.c_[x, y, np.ones(sample.shape[0])]

        # get plane coefficient
        # coefficients = fitPlaneLTSQ(np.array(sample))
        # print(coefficients)
        C,_,_,_ = scipy.linalg.lstsq(A, z) 

        # Coefficients in the form: a*x + b*y + c*z + d = 0.
        a, b, c, d = C[0], C[1], -1., C[2]

        # get point distances from the estimated plane
        dists = math.sqrt((((a*x + b*y + d) - z)**2).sum())
        # dists = np.abs(sample @ coefficients) / np.sqrt(coefficients[0]**2 + coefficients[1]**2 + coefficients[2]**2)
        # length_squared = a**2 + b**2 + c**2
        # dists = ((a * x + b * y + c * z + d) ** 2 / length_squared).sum() 

        
        num_inliers = len(np.where(dists < inlier_thresh)[0])
        if num_inliers > max_inlier_num:
            max_inlier_num = num_inliers
            best_model = [a, b, c, d]

    return best_model

def main(test_left, test_right, calib_dir):
    data = get_3d_locations(test_left, test_right,calib_dir)

    # use left image as the test image, only make prediction on the left image
    gt=[]
    if os.path.isfile('gt.npy'):
        gt = np.load('gt.npy', allow_pickle=True)
    else:
        predictions_l, img_seg_l = test_single_data(left_im_dir, calib_dir)
        gt_mask_left = get_segmentation(predictions_l, test_left, img_seg_l)
    
        np.save('gt', gt_mask_left)

    print("starting prepare road pixels")
    road_3ds = prepare_3d_points(gt, data)

    print('starting ransac')
    sample_size = int(len(road_3ds)*0.65)
    best_model = fit_plane_ransac(road_3ds, sample_size, max_iterations=50)

    print(best_model)



    road_3ds = np.array(road_3ds)
    x = np.arange(np.min(road_3ds[:, 0]), np.max(road_3ds[:, 0]), 0.1)
    z = np.arange(np.min(road_3ds[:, 2]), np.max(road_3ds[:, 2]), 0.1)

    xx, zz = np.meshgrid(x, z)

    xx = xx.flatten()
    zz = zz.flatten()

    yy = (-best_model[0]*xx - best_model[2]*zz - best_model[3])/best_model[1]
    plane = np.c_[xx, yy, zz]

    # print(plane[0])

    height, width, depth = test_left.shape
    test_left = cv2.cvtColor(test_left, cv2.COLOR_BGR2RGB)
    test_left = test_left/255
    data_color = test_left.reshape(height*width, 3)

    return v2(data, data_color, plane), data, best_model

if __name__ == "__main__":
    # dir
    left_im_dir = "train/image_left/umm_000015.jpg"
    right_im_dir = "train/image_right/umm_000015.jpg"
    calib_dir = "train/calib/umm_000015.txt"

    # left image
    test_left = cv2.imread(left_im_dir)

    # right image
    test_right = cv2.imread(right_im_dir)
   
    pcd, data, model = main(test_left, test_right, calib_dir)

    open3d.visualization.draw_geometries(pcd)
    

    


