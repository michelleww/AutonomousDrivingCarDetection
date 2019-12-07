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

def prepare_3d_points(gt_mask, map_3ds):
    points = []
    colors = []
    road_points = np.nonzero(gt_mask)
    height, width = gt_mask.shape
    size = len(road_points[0])
    for idx in range(size):
        i = road_points[0][idx]
        j = road_points[1][idx]
        points.append(map_3ds[i*width+j])
    print(len(points))
    return points




if __name__ == "__main__":
    # dir
    left_im_dir = "train/image_left/umm_000011.jpg"
    right_im_dir = "train/image_right/umm_000011.jpg"
    calib_dir = "train/calib/umm_000011.txt"

    # left image
    test_left = cv2.imread(left_im_dir)

    # right image
    test_right = cv2.imread(right_im_dir)

    print(test_left.shape)


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

    data = (np.array(data)*255).round().astype(np.uint8)
    
    
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(data) 

    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    # open3d.visualization.draw_geometries([downpcd])

    print("Recompute the normal of the downsampled point cloud")
    downpcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    open3d.visualization.draw_geometries([downpcd])


    # print(data)

    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(data)  # 定义点云坐标位置

    # open3d.visualization.draw_geometries([pcd])

    # open3d.visualization.draw_geometries([pcd])



    # # use left image as the test image, only make prediction on the left image
    # predictions_l, img_seg_l = test_single_data(left_im_dir, calib_dir)
    # gt_mask_left = get_segmentation(predictions_l, test_left, img_seg_l)

    # road_3ds = prepare_3d_points(gt_mask_left, data)
