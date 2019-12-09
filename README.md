# AutonomousDriving
This is our group's final project's code and data.  

## Data

There are full dataset avaiable on the official KITTI website: http://www.cvlibs.net/datasets/kitti/
Or Download the project2 data from https://piazza.com/class/k0bef0n637b26t?cid=90
Move the downloaded data to the AutonomousDriving/ Folder    

  - Place test, train, train_angle folder under AutonomousDriving/   
  
## Road classifier

run road_detection.py   

## 3D point cloud:   

run plane_new.py    

## Car detection and view point detection:   
  - For car detection only:
    - run car_detection.py
  - For car detection and view point viosualization:  
    - run car_segmentation.py for preprocessing the image segmentations   
    - run viewPoint_detection.py for training   
    - run draw_car_view_point.py for visualizing the result both detected car and view point    

## 3d bounding box:

run draw_3d_box.py     

## Reference for code and tutorial

- Filtering disparity: http://timosam.com/python_opencv_depthimage
- Superpixel tutorial: https://www.pyimagesearch.com/2014/07/28/a-slic-superpixel-tutorial-using-python/
- Projection matrix details: https://www.mrt.kit.edu/z/publ/download/2013/GeigerAl2013IJRR.pdf
- Haralic feature introduction: https://gogul.dev/software/texture-recognition
- List of color spaces: https://en.wikipedia.org/wiki/List_of_color_spaces_and_their_uses
- Three points plane fitting: http://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
