from skimage import io
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from skimage.measure import perimeter
import os
import joblib
import mahotas as mt
import matplotlib.colors as mcolors
import webcolors
from compute3d import get_3d_locations
from sklearn.ensemble import RandomForestClassifier
from skimage.color import rgb2lab, lab2rgb
import copy
from skimage import feature
from sklearn.model_selection import train_test_split

def extract_feature(image, image_gt, image_right, calib_path):
    # calculate image gradient using sobel
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=5)

    image_lab = rgb2lab(copy.deepcopy(image)/255)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # compute 3D locations
    locations_3D = get_3d_locations(image, image_right, calib_path)

    # super pixels for approximatly 1200 size
    img_seg = slic(image, n_segments = 1200, compactness=8, convert2lab=True, min_size_factor=0.3)
    # actual total number of labels
    num_labels = np.amax(img_seg)+1

    features_list = []
    labels = []
    for label in range(num_labels):
        features = []
        roi = (img_seg == label)
        roi_idx = np.nonzero(roi)

        seg_center_x = np.mean(roi_idx[0])
        seg_center_y = np.mean(roi_idx[1])

        seg_sobel_x = sobelx[roi]
        seg_sobel_y = sobely[roi]

        magnitude_mean = np.mean(np.sqrt(seg_sobel_x ** 2 + seg_sobel_y ** 2))

        direction_mean = np.mean(np.arctan2(np.absolute(seg_sobel_y), np.absolute(seg_sobel_x)))

        # add computed 2 D features
        features.extend([np.mean(seg_sobel_x), np.mean(seg_sobel_y), magnitude_mean, direction_mean])
        
        # extract color features  and 3D feature
        x, y, z = 0, 0, 0
        flag = True
        rgb_mean = []
        lab_mean = []
        hsv_mean = []
        size = len(roi_idx[0])
        lable_sum = 0
        for channel in range(3):
            sum_rgb = 0
            sum_lab = 0
            sum_hsv = 0
            for idx in range(size):
                i = roi_idx[0][idx]
                j = roi_idx[1][idx]
                sum_rgb += image[i, j, channel]
                sum_lab += image_lab[i,j,channel]
                sum_hsv += image_hsv[i,j,channel]
                if flag:
                    if idx == size - 1:
                        flag = False
                    item_idx = image.shape[1] * i + j
                    x += locations_3D[item_idx][0]
                    y += locations_3D[item_idx][1]
                    z += locations_3D[item_idx][2]
                    if image_gt[i,j,0] > 0:
                        lable_sum+=1
            rgb_mean.append(sum_rgb/size)
            lab_mean.append(sum_lab/size)
            hsv_mean.append(sum_hsv/size)        
        features.extend(rgb_mean)
        features.extend(lab_mean)
        features.extend(hsv_mean)
        features.extend([x/size, y/size, z/size])

        seg_gray = image_gray*roi

        # add texture feature  
        textures = mt.features.haralick(seg_gray)
        features.append(np.mean(textures))

        # adding lable
        labels.append(int(round(lable_sum/size)))

        features_list.append(features)

    return features_list, labels

def extract_test_feature(image, image_right, calib_path):
    # calculate image gradient using sobel
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=5)

    image_lab = rgb2lab(copy.deepcopy(image)/255)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # compute 3D locations
    locations_3D = get_3d_locations(image, image_right, calib_path)

    # super pixels for approximatly 1200 size
    img_seg = slic(image, n_segments = 1200, compactness=8, convert2lab=True, min_size_factor=0.3)
    # actual total number of labels
    num_labels = np.amax(img_seg)+1

    features_list = []
    for label in range(num_labels):
        features = []
        roi = (img_seg == label)
        roi_idx = np.nonzero(roi)

        seg_sobel_x = sobelx[roi]
        seg_sobel_y = sobely[roi]
        magnitude_mean = np.mean(np.sqrt(seg_sobel_x ** 2 + seg_sobel_y ** 2))

        direction_mean = np.mean(np.arctan2(np.absolute(seg_sobel_y), np.absolute(seg_sobel_x)))

        # add computed 2 D features
        features.extend([np.mean(seg_sobel_x), np.mean(seg_sobel_y), magnitude_mean, direction_mean])

        # extract color features  and 3D feature
        x, y, z = 0, 0, 0
        flag = True
        rgb_mean = []
        lab_mean = []
        hsv_mean = []
        size = len(roi_idx[0])

        for channel in range(3):
            sum_rgb = 0
            sum_lab = 0
            sum_hsv = 0
            for idx in range(size):
                i = roi_idx[0][idx]
                j = roi_idx[1][idx]
                sum_rgb += image[i, j, channel]
                sum_lab += image_lab[i,j,channel]
                sum_hsv += image_hsv[i,j,channel]
                if flag:
                    if idx == size - 1:
                        flag = False
                    item_idx = image.shape[1] * i + j
                    x += locations_3D[item_idx][0]
                    y += locations_3D[item_idx][1]
                    z += locations_3D[item_idx][2]
            rgb_mean.append(sum_rgb/size)
            lab_mean.append(sum_lab/size)
            hsv_mean.append(sum_hsv/size)        
        features.extend(rgb_mean)
        features.extend(lab_mean)
        features.extend(hsv_mean)
        features.extend([x/size, y/size, z/size])

        seg_gray = image_gray*roi
        # add texture feature   
        textures = mt.features.haralick(seg_gray)
        features.append(np.mean(textures))

        features_list.append(features)

    return features_list, img_seg

def extract_trainings():

    img_name = [name for name in os.listdir('train/image_left') if name.endswith('.jpg') or name.endswith('.png')]
    # all data features
    data_features = []
    data_labels = []
    
    print('Start Reading training images...')
    # Loop through all images
    for name in img_name:
 
        print('Loading image ' + name)
    
        # Load image and ground truth image
        img_path = 'train/image_left/' + name
        img_gt_name = name.replace('_', '_road_', 1).replace('jpg', 'png')
        img_gt_path = 'train/gt_image_left/' + img_gt_name
        img_right_path = 'train/image_right/' + name
        calib_path= ('train/calib/'+ name).replace('jpg', 'txt')

        img = cv2.imread(img_path)
        img_gt = cv2.imread(img_gt_path)
        img_right = cv2.imread(img_right_path)

        features, labels= extract_feature(img, img_gt, img_right, calib_path)

        # add features for a single image
        data_features.extend(features)
        data_labels.extend(labels)

    # Output Data to file
    np.save('training_data_features_2', data_features)
    np.save('training_data_labels_2', data_labels)

    # Display Message
    print('Data Successfully Saved!')
    return data_features, data_labels

def train(data, labels, model_path):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    print('Start Training process...')
    
    model = RandomForestClassifier(n_estimators=330, criterion='entropy', min_samples_leaf=3, min_samples_split=8)  
    model.fit(X_train, y_train)
    print('The training accuracy is: ' + str(model.score(X_train, y_train)))
    joblib.dump(model, model_path)
    print('Done training!')

    print('The testing accuracy is: ' + str(model.score(X_test, y_test)))
    return model

def test_single_data(image_path, calib_path=None):
    data, labels = [], []

    if os.path.isfile('training_data_features_2.npy') and os.path.isfile('training_data_labels_2.npy'):
        data, labels = load_training_data()
    else: 
        data, labels = extract_trainings()

    image = cv2.imread(image_path)
    image_right = cv2.imread(image_path.replace('image_left', 'image_right'))
    if calib_path == None:
        calib_path = image_path.replace('image_left', 'calib').replace('.jpg', '.txt')
    test_data, img_seg = extract_test_feature(image, image_right, calib_path)

    model = train(data, labels, 'road_detection_RF.pkl')

    predictions = model.predict(test_data)

    return predictions, img_seg

def load_training_data():
    # Load Data
    data = np.load('training_data_features_2.npy', allow_pickle=True)
    labels = np.load('training_data_labels_2.npy', allow_pickle=True)
    
    # Display Message
    print('Training Data Successfully Loaded!')
    return data, labels

def get_segmentation(predictions, img, img_seg):
    num_labels = np.amax(img_seg)+1
    blank = np.zeros((img.shape[0],img.shape[1]))
    for label in range(num_labels):
        pre = predictions[label]
        if pre == 1:
            roi = (img_seg == label)
            roi_idx = np.nonzero(roi)
            for idx in range(len(roi_idx[0])):
                i = roi_idx[0][idx]
                j = roi_idx[1][idx]
                blank[i,j] = 10
    return blank

# see tut7 for reference
def visualize_segementation(imgLeft, gt_mask):
    unique, counts = np.unique(gt_mask, return_counts=True)
    obj_ids = np.unique(gt_mask)
    
    number_object = obj_ids.shape[0]

    # norm ids
    count = 0
    for o_id in obj_ids:
        gt_mask[gt_mask == o_id] = count
        count += 1

    base_COLORS = []

    for key, value in mcolors.CSS4_COLORS.items():
        rgb = webcolors.hex_to_rgb(value)
        base_COLORS.append([rgb.blue, rgb.green, rgb.red])
    base_COLORS = np.array(base_COLORS)

    np.random.seed(99)
    base_COLORS = np.random.permutation(base_COLORS)

    colour_id = np.array([(id) % len(base_COLORS) for id in range(number_object)])

    COLORS = base_COLORS[colour_id]
    COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
    gt_mask = np.int_(gt_mask)
    mask = COLORS[gt_mask]

    output = ((0.4 * imgLeft) + (0.6 * mask)).astype("uint8")
    
    fig=plt.figure(figsize=(10, 10))
    plt.imshow(output[:,:,::-1])
    plt.show()

if __name__ == '__main__':
    test_path = 'train/image_left/um_000025.jpg'
    test = cv2.imread('train/image_left/um_000025.jpg')
    predictions, img_seg = test_single_data(test_path)
    gt_mask_1 = get_segmentation(predictions, test, img_seg)
    visualize_segementation(test, gt_mask_1)
