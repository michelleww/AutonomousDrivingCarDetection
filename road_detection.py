from skimage import io
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from skimage.measure import perimeter
import os
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import joblib
import mahotas as mt
import matplotlib.colors as mcolors
import webcolors

def extract_feature(image, image_gt):
    # calculate image gradient using sobel
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=5)

    # super pixels for approximatly 1000 size
    img_seg = slic(image, n_segments = 1000, compactness=8, convert2lab=True, min_size_factor=0.3)
    # actual total number of labels
    num_labels = np.amax(img_seg)+1

    bins = np.linspace(0,1,num=21)

    features_list = []
    labels = []
    for label in range(num_labels):
        features = []
        roi = (img_seg == label)
        roi_idx = np.nonzero(roi)

        seg_center_x = np.mean(roi_idx[0])
        seg_center_y = np.mean(roi_idx[1])

        seg_size = np.shape(roi_idx)[1]

        seg_perimeter = perimeter(roi)

        seg_sobel_x = sobelx[roi]
        seg_sobel_y = sobely[roi]

        magnitude_mean = np.mean(np.sqrt(seg_sobel_x ** 2 + seg_sobel_y ** 2))

        direction_mean = np.mean(np.arctan2(np.absolute(seg_sobel_y), np.absolute(seg_sobel_x)))

        # add computed 2 D features
        features.extend([seg_size, seg_perimeter, np.mean(seg_sobel_x), np.mean(seg_sobel_y), magnitude_mean, direction_mean])
        # extract color features 
        for channel in range(3):
            sum = 0
            for idx in range(len(roi_idx[0])):
                i = roi_idx[0][idx]
                j = roi_idx[1][idx]
                sum += image[i, j, channel]
            features.append(sum/len(roi_idx[0]))

        # add texture feature     
        seg_rgb = image[roi]
        textures = mt.features.haralick(seg_rgb)
        features.extend(textures.mean(axis=0))

        # adding lable
        if image_gt[int(seg_center_x),int(seg_center_y),2] > 0:
            labels.append(1)
        else:
            labels.append(0)

        features_list.append(features)

    return features_list, labels

def extract_test_feature(image):
    # calculate image gradient using sobel
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image_gray,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image_gray,cv2.CV_64F,0,1,ksize=5)

    # super pixels for approximatly 1000 size
    img_seg = slic(image, n_segments = 1000, compactness=8, convert2lab=True, min_size_factor=0.3)
    # actual total number of labels
    num_labels = np.amax(img_seg)+1

    bins = np.linspace(0,1,num=21)

    features_list = []
    for label in range(num_labels):
        features = []
        roi = (img_seg == label)
        roi_idx = np.nonzero(roi)

        seg_size = np.shape(roi_idx)[1]

        seg_perimeter = perimeter(roi)

        seg_sobel_x = sobelx[roi]
        seg_sobel_y = sobely[roi]
        magnitude_mean = np.mean(np.sqrt(seg_sobel_x ** 2 + seg_sobel_y ** 2))

        direction_mean = np.mean(np.arctan2(np.absolute(seg_sobel_y), np.absolute(seg_sobel_x)))

        # add computed 2 D features
        features.extend([seg_size, seg_perimeter, np.mean(seg_sobel_x), np.mean(seg_sobel_y), magnitude_mean, direction_mean])

        # extract color features 
        for channel in range(3):
            sum = 0
            for idx in range(len(roi_idx[0])):
                i = roi_idx[0][idx]
                j = roi_idx[1][idx]
                sum += image[i, j, channel]

            features.append(sum/len(roi_idx[0]))
        # add texture feature     
        seg_rgb = image[roi]
        textures = mt.features.haralick(seg_rgb)
        features.extend(textures.mean(axis=0))
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

        img = cv2.imread(img_path)
        img_gt = cv2.imread(img_gt_path)

        features, labels= extract_feature(img, img_gt)

        # add features for a single image
        data_features.extend(features)
        data_labels.extend(labels)

    # Output Data to file
    np.save('training_data_features', data_features)
    np.save('training_data_labels', data_labels)

    # Display Message
    print('Data Successfully Saved!')
    return data_features, data_labels

def train(data, labels, svm_path):
    X_train, y_train = data,labels
    print('Start Training process...')
    svm = LinearSVC(random_state=0, dual= False, multi_class='ovr', class_weight = 'balanced', max_iter= 10000, C= 100.0, tol=1e-5)
    svm.fit(X_train, y_train)
    print('The accuracy is: ' + str(svm.score(X_train, y_train)))
    joblib.dump(svm, svm_path)
    print('Done training!')

def test_single_data(image):
    data, labels = [], []

    if os.path.isfile('training_data_features.npy') and os.path.isfile('training_data_labels.npy'):
        data, labels = load_training_data()
    else: 
        data, labels = extract_trainings()

    test_data, img_seg = extract_test_feature(image)
    train(data, labels, 'road_detection_svm_lsvc.pkl')

    svm = joblib.load('road_detection_svm_lsvc.pkl')
 
    # evaluate loaded model on test data
    predictions = svm.predict(test_data)

    return predictions, img_seg

def load_training_data():
    # Load Data
    data = np.load('training_data_features.npy', allow_pickle=True)
    labels = np.load('training_data_labels.npy', allow_pickle=True)
    
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
    test = cv2.imread('test/image_left/umm_000061.jpg')
    predictions, img_seg = test_single_data(test)
    gt_mask_1 = get_segmentation(predictions, test, img_seg)
    visualize_segementation(test, gt_mask_1)
