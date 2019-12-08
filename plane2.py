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


# def visualize_3d_point_cloud():
 
# # 绘制open3d坐标系
#     axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
# # 在3D坐标上绘制点：坐标点[x,y,z]对应R，G，B颜色
#     points = np.array([[0.1, 0.1, 0.1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
#     colors = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
 
#     test_pcd = open3d.geometry.PointCloud()  # 定义点云

    # vis = open3d.Visualizer()
    # vis.create_window(window_name="Open3D1")
    # vis.get_render_option().point_size = 3
    # first_loop = True
    # # 先把点云对象添加给Visualizer
    # vis.add_geometry(axis_pcd)
    # vis.add_geometry(test_pcd)
    # while True:
    # # 给点云添加显示的数据
    #     points -= 0.001
    #     test_pcd.points = open3d.utility.Vector3dVector(points)  # 定义点云坐标位置
    #     test_pcd.colors = open3d.Vector3dVector(colors)  # 定义点云的颜色
    # # update_renderer显示当前的数据
    #     vis.update_geometry()
    #     vis.poll_events()
    #     vis.update_renderer()
    #     cv2.waitKey(100)
 
    # 方法2（阻塞显示）：调用draw_geometries直接把需要显示点云数据
    # test_pcd.points = open3d.utility.Vector3dVector(points)  # 定义点云坐标位置
    # test_pcd.colors = open3d.utility.Vector3dVector(colors)  # 定义点云的颜色
    # open3d.visualization.draw_geometries([test_pcd] + [axis_pcd], window_name="Open3D2")

def prepare_3d_points(gt_mask, map_3ds, img):
    points = []
    colors = []
    # plt.imshow(gt_mask)
    # plt.show()
    road_points = np.nonzero(gt_mask)
    gt_mask = np.array(gt_mask)
    height, width = gt_mask.shape

    size = len(road_points[0])
    for idx in range(size):
        i = road_points[0][idx]
        j = road_points[1][idx]
        position = map_3ds[i*width+j]
        points.append([position[0], position[1], position[2]])
        # print(img[i,j])
        colors.append(img[i,j]/255)
    print(len(points))
    return points, colors


def v2(data, colors):

    # print(x)
    print(colors[0])
    print(data[0])
    # colors = [[0.5,0,0] for c in colors]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(data) 
    pcd.colors = open3d.utility.Vector3dVector(colors) 

    print("Downsample the point cloud with a voxel of 0.05")
    # downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    # open3d.visualization.draw_geometries([downpcd])

    print("Recompute the normal of the downsampled point cloud")
    # downpcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
    #     radius=0.1, max_nn=30))
    open3d.visualization.draw_geometries([pcd])

def v1(data):
    x=[k[0] for k in data]
    y=[k[1] for k in data]
    z=[k[2] for k in data]

    fig=plt.figure(dpi=120)
    ax=fig.add_subplot(111,projection='3d')
    plt.title('point cloud')
    ax.scatter(x,y,z,c='b',marker='.',s=2,linewidth=0,alpha=1,cmap='spectral')

    #ax.set_facecolor((0,0,0))
    # ax.axis('scaled')          
    # ax.xaxis.set_visible(False) 
    # ax.yaxis.set_visible(False) 
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def fit_plane_ransac(road_pixels, sample_size, max_iterations, inlier_thresh=0.03, random_seed=None):
    max_inlier_num = -1
    best_model = None
    random.seed(random_seed)

    for i in range(max_iterations):
        sample = np.array(random.sample(road_pixels, sample_size))

        A = np.c_[sample[:,0], sample[:,1], np.ones(sample.shape[0])]

        # get plane coefficient
        coefficients,_,_,_ = scipy.linalg.lstsq(A, sample[:,2]) 
        print(coefficients)

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


    # # compute disparity
    # disparity = compute_disparity(test_left, test_right)
    # # plt.imshow(disparity)
    # # plt.show()
    # # compute depth
    # f, T, px, py, depth = compute_depth(calib_dir, disparity)
    # depth = (depth*255).round().astype(np.uint8)
    # print(depth)
    # print(depth.shape)
    
    # # compute 3d locations
    # image_3d = compute_3d_2(depth, px, py, f)

    # X = image_3d[:,:,0].flatten()
    # Y = image_3d[:,:,1].flatten()
    # Z = image_3d[:,:,2].flatten()
    # data = np.c_[X,Y,Z]
    # print(data)

    data = get_3d_locations(test_left, test_right,calib_dir)

    data = (np.array(data)*255).astype(np.uint8)
    


    # print(data)

    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(data)  # 定义点云坐标位置

    # open3d.visualization.draw_geometries([pcd])

    # open3d.visualization.draw_geometries([pcd])



    # # use left image as the test image, only make prediction on the left image
    # gt=[]
    # if os.path.isfile('gt.npy'):
    #     gt = np.load('gt.npy', allow_pickle=True)
    # else:
    #     predictions_l, img_seg_l = test_single_data(left_im_dir, calib_dir)
    #     gt_mask_left = get_segmentation(predictions_l, test_left, img_seg_l)
    
    #     np.save('gt', gt_mask_left)
    #road_3ds, colors = prepare_3d_points(gt, data, test_left)
    # sample_size = int(len(road_3ds)*0.65)
    # fit_plane_ransac(road_3ds, sample_size, 2, random_seed=7)
    # print(prepare_3d_points(gt, data))

    # z = [(point[2]=0) for point in road_3ds]
    # new = np.array(road_3ds)
    # new[new]
    # counts = np.bincount(z)
    # print(np.argmax(counts))
    data_color = np.zeros_like(test_left)

    height, width, depth = test_left.shape
    test_left = test_left/255
    data_color = test_left.reshape(height*width, 3)
    

    v2(data, data_color)


