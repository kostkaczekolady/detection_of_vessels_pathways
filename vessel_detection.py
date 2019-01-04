from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from sklearn import cluster
import ntpath
import csv


# DRIVE_TRAIN_IMAGES = './DRIVE/training/images/'
# DRIVE_TRAIN_MASK = './DRIVE/training/mask/'
DRIVE_TEST_IMAGES = './DRIVE/test/images/'
DRIVE_TEST_MASK = './DRIVE/test/mask/'
DRIVE_TEST_1T_MANUAL = './DRIVE/test/1st_manual/'
DRIVE_TEST_2nd_MANUAL = './DRIVE/test/2nd_manual/'
    
def get_files_list(path):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]

def get_file_path_with_prefix(prefix, path):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f[0:2] == prefix]

def get_file_name_from_path(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def apply_mask(img, img_path):
    image_name = get_file_name_from_path(img_path)
    mask_path = get_file_path_with_prefix(image_name[0:2], DRIVE_TEST_MASK)[0]
    mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if mask[y][x] != 0:
                new_pixel = img[y][x]
            else:
                new_pixel = 0
            img[y][x] = new_pixel

def get_green_channel(img_path):
    img = cv2.imread(img_path)
    _,g,_ = cv2.split(img)
    apply_mask(g, img_path)
    return g

def subtract_images(minuend, subtrahend):
    height, width = minuend.shape
    subtracted_image = np.zeros((height,width), np.uint8)
    for y in range(height):
        for x in range(width):
            new_pixel = minuend[y][x] - subtrahend[y][x]
            if new_pixel < 0 or new_pixel > 255:
                subtracted_image[y][x] = 0
            else:
                subtracted_image[y][x] = new_pixel
    return subtracted_image

def get_max_pix_value(image):
    height, width = image.shape
    max_value = 0
    for y in range(height):
        for x in range(width):
            if image[y][x] > max_value:
                max_value = image[y][x]
    return max_value

def normalize_image(image, max_value=None):
    if not max_value:
        max_value = get_max_pix_value(image)
    height, width = image.shape
    normalized_image = np.zeros((height,width), np.uint8)
    factor = 255/max_value
    for y in range(height):
        for x in range(width):
            normalized_image[y][x] = image[y][x] * factor
    return normalized_image

def combine_images(image1, image2):
    height, width = image1.shape
    combined_image = np.zeros((height,width), np.uint8)
    max_pix_value = 0
    for y in range(height):
        for x in range(width):
            new_pixel = image1[y][x] + image2[y][x]
            if new_pixel > max_pix_value:
                max_pix_value = new_pixel
            combined_image[y][x] = new_pixel
    if max_pix_value > 255:
        combined_image = normalize_image(combined_image, max_pix_value)
    return combined_image

def cluster_image(image):
    height, width = image.shape
    reshaped_image = image.reshape((width*height,1))
    kmeans_cluster = cluster.KMeans(n_clusters=10)
    kmeans_cluster.fit(reshaped_image)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    clustered_img = cluster_centers[cluster_labels].reshape((image.shape))
    clustered_img = np.uint8(clustered_img)
    return clustered_img

def postprocess_image(image):
    median_filtred_img = cv2.medianBlur(image, 3)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(median_filtred_img, cv2.MORPH_OPEN, kernel)
    erosion = cv2.erode(opening,kernel,iterations = 1)
    return erosion

def create_csv_headers(method):
    with open('{}_result.csv'.format(method), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['File', 'Sensitivity', 'Specificity', 'Accuracy', 'TP', 'FP', 'TN', 'FN'])

def save_result_to_csv(results, image_file_name, method):
    with open('{}_result.csv'.format(method), 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([image_file_name, results[0], results[1], results[2], results[3], results[4], results[5], results[6]])

def measure_performance(image, image_path, method):
    image_name = get_file_name_from_path(image_path)
    manual_path = get_file_path_with_prefix(image_name[0:2], DRIVE_TEST_1T_MANUAL)[0]
    manual = cv2.imread(manual_path, flags=cv2.IMREAD_GRAYSCALE)
    height, width = manual.shape
    TP = 0 # True positive
    FP = 0 # False positive
    TN = 0 # True negative
    FN = 0 # False negative
    for y in range(height):
        for x in range(width):
            if image[y][x] == 0 and manual[y][x] == 0:
                TN += 1
            elif image[y][x] >= 1 and manual[y][x] >= 1:
                TP += 1
            elif image[y][x] == 0 and manual[y][x] >= 1:
                FN += 1
            elif image[y][x] >= 1 and manual[y][x] == 0:
                FP += 1
    sensitivity = TP/(TP + FN)
    specificity = TN/(TN + FP)
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    save_result_to_csv((sensitivity, specificity, accuracy, TP, FP, TN, FN), image_name, method)

def main():
    images_path = get_files_list(DRIVE_TEST_IMAGES)
    methods = ('DIMDF', 'DIMNF', 'DIGF', 'DIMDMNF', 'DIMDGF', 'DIMNGF')
    for method in methods:
        create_csv_headers(method)

    for idx, image_path in enumerate(images_path):
        ### PREPROCESSING ##
        # Take green channel from vessel image
        gc = get_green_channel(image_path)

        # Use 3 filtering techniques
        mean = cv2.blur(gc, (11,11))
        median = cv2.medianBlur(gc, 15)
        gaussian = cv2.GaussianBlur(gc, (11,11), 0)

        # Create difference image
        # DIMDF -  difference image based on median filter 
        # DIMNF - difference image based on mean filter 
        # DIGF - difference image based on Gaussian filter
        DIMDF = subtract_images(gc, mean)
        DIMNF = subtract_images(gc, median)
        DIGF = subtract_images(gc, gaussian)

        # DIMDMNF - combination of median filter and mean filter based difference images
        # DIMDGF - combination of median filter and Gaussian filter based difference images
        # DIMNGF - combination of mean filter and Gaussian filter based difference images
        DIMDMNF = combine_images(DIMDF, DIMNF)
        DIMDGF = combine_images(DIMDF, DIGF)
        DIMNGF = combine_images(DIMNF, DIGF)

        ### CLUSTERING ###
        # Cluster all images
        cluster_DIMDF = cluster_image(DIMDF)
        cluster_DIMNF = cluster_image(DIMNF)
        cluster_DIGF = cluster_image(DIGF)
        cluster_DIMDMNF = cluster_image(DIMDMNF)
        cluster_DIMDGF = cluster_image(DIMDGF)
        cluster_DIMNGF = cluster_image(DIMNGF)


        ### POSTPROCESSING ###
        postprocess_DIMDF = postprocess_image(cluster_DIMDF)
        postprocess_DIMNF = postprocess_image(cluster_DIMNF)
        postprocess_DIGF = postprocess_image(cluster_DIGF)
        postprocess_DIMDMNF = postprocess_image(cluster_DIMDMNF)
        postprocess_DIMDGF = postprocess_image(cluster_DIMDGF)
        postprocess_DIMNGF = postprocess_image(cluster_DIMNGF)


        ### PERFORMANCE ###
        measure_performance(postprocess_DIMDF, image_path, 'DIMDF')
        measure_performance(postprocess_DIMNF, image_path, 'DIMNF')
        measure_performance(postprocess_DIGF, image_path, 'DIGF')
        measure_performance(postprocess_DIMDMNF, image_path, 'DIMDMNF')
        measure_performance(postprocess_DIMDGF, image_path, 'DIMDGF')
        measure_performance(postprocess_DIMNGF, image_path, 'DIMNGF')

        print('Progress {}/{}'.format(idx+1, len(images_path)))

if __name__ == "__main__":
    main()
