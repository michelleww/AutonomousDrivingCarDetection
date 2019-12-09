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
from sklearn import linear_model, datasets
from skimage.measure import LineModelND, ransac
import scipy.linalg
import open3d
import random
import math
import matplotlib.pyplot as plt

def get_plane(points):

    p1 = np.array(points[0])
    p2 = np.array(points[1])
    p3 = np.array(points[2])
# These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

# the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

# This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)
    return [a, b, c,d]


def run_ransac(data, max_iterations=65, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    # random.sample cannot deal with "data" being a numpy array
    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, 3)
        m = get_plane(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(np.array(m), data[j],0.01):
                ic += 1

        if ic > best_ic:
            best_ic = ic
            best_model = m

    return best_model

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def is_inlier(m, xyz, threshold):
    return np.abs(m.dot(augment([xyz]).T)) < threshold

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

    pcd_plane = open3d.geometry.PointCloud()
    pcd_plane.points = open3d.utility.Vector3dVector(plane) 
    pcd_plane.colors = open3d.utility.Vector3dVector(plane_color) 

    return [pcd_img, pcd_plane]


def main(left_im_dir, test_left, test_right, calib_dir):
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

    [a,b,c,d] = run_ransac(road_3ds)

    road_3ds = np.array(road_3ds)

    x = np.arange(np.min(road_3ds[:, 0]), np.max(road_3ds[:, 0]), 0.1)
    y = np.arange(np.min(road_3ds[:, 1]), np.max(road_3ds[:, 1]), 0.1)

    X, Y = np.meshgrid(x, y)

    # ax + by +cz = d
    Z = np.divide((d - a * X - b * Y),  c)

    plane = np.c_[X.flatten(), Y.flatten(), Z.flatten()]

    height, width, depth = test_left.shape
    # convert BGR to RGB
    test_left = cv2.cvtColor(test_left, cv2.COLOR_BGR2RGB)
    test_left = test_left/255
    data_color = test_left.reshape(height*width, 3)

    return v2(data, data_color, plane), data, [a,b,c,d]

if __name__ == "__main__":
    # dir
    left_im_dir = "train/image_left/umm_000011.jpg"
    right_im_dir = "train/image_right/umm_000011.jpg"
    calib_dir = "train/calib/umm_000011.txt"

    # left image
    test_left = cv2.imread(left_im_dir)

    # right image
    test_right = cv2.imread(right_im_dir)
   
    pcd, data, model = main(left_im_dir,test_left, test_right, calib_dir)

    open3d.visualization.draw_geometries(pcd)
    

    


