# AutonomousDriving
This is our group's final project's code and data.  

## Data

There are full dataset avaiable on the official KITTI website: http://www.cvlibs.net/datasets/kitti/
Or Download the project2 data from https://piazza.com/class/k0bef0n637b26t?cid=90
Move the downloaded data to the AutonomousDriving/ Folder    

  - Place test, train, train_angle folder under AutonomousDriving/   

## Depth, disparity and 3D locations
  - disparity.py for computing filtered disparity
  - depth.py for computinng depth map using calibration information
  - compute3D.py

## Road classifier

road_detection.py 
  - pre-processing and extracting features
  - training and testing

## 3D point cloud:   

plane_new.py 
  - plotting 3D point cloud
  - fitting points into 3D plane
  - visualization for 3D point cload and the estimated ground plane

## Car detection and view point detection:   
  - For car detection only:
    - car_detection.py that returns the bounding box for cars
  - For car detection and view point viosualization:  
    - car_segmentation.py for preprocessing the image segmentations   
    - viewPoint_detection.py for training and testing
    - draw_car_view_point.py for visualizing the result both detected car and view point    

## 3d bounding box:

draw_3d_box.py for plotting the 3D point cloud, estimated ground plane and the 3D bounding box   

## Reference for code and tutorial

- Filtering disparity: http://timosam.com/python_opencv_depthimage
- Superpixel tutorial: https://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/
- Projection matrix details: https://www.mrt.kit.edu/z/publ/download/2013/GeigerAl2013IJRR.pdf
- Haralic feature introduction: https://gogul.dev/software/texture-recognition
- List of color spaces: https://en.wikipedia.org/wiki/List_of_color_spaces_and_their_uses
- Three points plane fitting: http://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
