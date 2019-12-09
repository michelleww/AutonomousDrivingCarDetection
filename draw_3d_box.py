
import open3d
import math
from car_detection import object_detection_api
from draw_car_viewPoint import create_output_directory
from plane_new import main
import cv2

def draw_3d_boxes(boxes, plane_model, data, img_path):
    # line connections
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    # red colors for every line 
    colors = [[1, 0, 0] for i in range(len(lines))]
    boxes_3d = []

    img_gray = cv2.imread(img_path, 0)
    height, width = img_gray.shape

    for box in boxes:
        points = []
        (left, top), (right, bottom) = box[0], box[1]

        left, top, right, bottom = int(round(left)), int(round(top)), int(round(right)), int(round(bottom))

        left_top_3d = data[top*width+left]
        right_bottom_3d = data[bottom*width+right]

        # get the 3D location for the center point inside the 2D bounding box
        center_x = int(round((right + left)/2))
        center_y = int(round((bottom + top)/2))
        center_3d = data[center_y*width+center_x]
        
        x= center_3d[0]
        y= center_3d[1]
        z = center_3d[2]
        # get the estimated Z in the plane     
        # # ax + by +cz = d
        z_prime = (-plane_model[0]*x - plane_model[1]*y + plane_model[3])/plane_model[2]

        # compute distance 
        z_dis = normalize(abs((z - z_prime)/z))
        x_dis = normalize(abs((center_x - left_top_3d[0])/max(center_x, left_top_3d[0])))
        y_dis = normalize(abs((center_y - right_bottom_3d[1])/max(center_y, right_bottom_3d[1])))
        

   
        # get 8 points that forming the 3D bounding box
        points.extend([[x-x_dis, y-y_dis,z+z_dis], [x+x_dis, y-y_dis,z+z_dis], [x-x_dis, y+y_dis,z+z_dis], [x+x_dis, y+y_dis,z+z_dis]])
        points.extend([[x-x_dis, y-y_dis,z-z_dis], [x+x_dis, y-y_dis,z-z_dis], [x-x_dis, y+y_dis,z-z_dis], [x+x_dis, y+y_dis,z-z_dis]])

        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(points)
        line_set.lines = open3d.utility.Vector2iVector(lines)
        line_set.colors = open3d.utility.Vector3dVector(colors)
        boxes_3d.append(line_set)
    return boxes_3d

def normalize(x):
    while x > 1 or x < 0.35:
        if x>1:
            x = x/2
        else:
            x = 2*x
    return x
    
if __name__ == "__main__":
    left_im_dir = "train/image_left/um_000088.jpg"
    right_im_dir = "train/image_right/um_000088.jpg"
    calib_dir = "train/calib/um_000088.txt"
    create_output_directory()
    cars = object_detection_api(left_im_dir)
    # left image
    test_left = cv2.imread(left_im_dir)

    # right image
    test_right = cv2.imread(right_im_dir)
    pcd, data, plane_model = main(left_im_dir, test_left, test_right, calib_dir)
    line_sets = draw_3d_boxes(cars, plane_model, data, left_im_dir)
    pcd.extend(line_sets)
    open3d.visualization.draw_geometries(pcd)

