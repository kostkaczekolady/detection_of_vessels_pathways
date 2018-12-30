from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from sklearn import cluster
import ntpath

# DRIVE_TRAIN_IMAGES = './DRIVE/training/images/'
# DRIVE_TRAIN_MASK = './DRIVE/training/mask/'
DRIVE_TEST_IMAGES = './DRIVE/test/images/'
DRIVE_TEST_MASK = './DRIVE/test/mask/'
    
def get_files_list(path):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]

def get_file_path_with_prefix(prefix, path):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f)) and f[0:2] == prefix]

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def apply_mask(img, img_path):
    image_name = path_leaf(img_path)
    mask_path = get_file_path_with_prefix(image_name[0:2], DRIVE_TEST_MASK)[0]
    mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
    width = mask.shape[1]
    height = mask.shape[0]
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
    width = minuend.shape[1]
    height = minuend.shape[0]
    subtracted_image = np.zeros((height,width), np.uint8)
    for y in range(height):
        for x in range(width):
            new_pixel = minuend[y][x] - subtrahend[y][x]
            if new_pixel < 0 or new_pixel > 255:
                subtracted_image[y][x] = 0
            else:
                subtracted_image[y][x] = new_pixel
    return subtracted_image

def normalize_image(image, max_value):
    width = image.shape[1]
    height = image.shape[0]
    normalized_image = np.zeros((height,width), np.uint8)
    factor = 255/max_value
    for y in range(height):
        for x in range(width):
            normalized_image[y][x] = image[y][x] * factor
    return normalized_image

def combine_images(image1, image2):
    width = image1.shape[1]
    height = image1.shape[0]
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
    width = image.shape[1]
    height = image.shape[0]
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

def main():
    ### PREPROCESSING ##
    # Take green channel from vessel image
    images_path = get_files_list(DRIVE_TEST_IMAGES)
    gc = get_green_channel(images_path[0])

    # Use 3 filtering techniques
    mean = cv2.blur(gc, (11,11))
    median = cv2.medianBlur(gc, 11)
    gaussian = cv2.GaussianBlur(gc, (7,7), 0)

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
    ### POSTPROCESSING ###
    postprocess_DIMDF = postprocess_image(cluster_DIMDF)
    postprocess_DIMNF = postprocess_image(cluster_DIMNF)
    postprocess_DIGF = postprocess_image(cluster_DIGF)
    postprocess_DIMDMNF = postprocess_image(cluster_DIMDMNF)
    postprocess_DIMDGF = postprocess_image(cluster_DIMDGF)
    postprocess_DIMNGF = postprocess_image(cluster_DIMNGF)

    cv2.imshow("image", gc)
    cv2.imshow("mean", mean)
    cv2.imshow("gaussian", gaussian)
    cv2.imshow("median", median)
    cv2.imshow("DIMDF", DIMDF)
    cv2.imshow("DIMNF", DIMNF)
    cv2.imshow("DIGF", DIGF)
    cv2.imshow("DIMDMNF", DIMDMNF)
    cv2.imshow("DIMDGF", DIMDGF)
    cv2.imshow("DIMNGF", DIMNGF)
    cv2.imshow("cluster_DIMDF", cluster_DIMDF)
    cv2.imshow("cluster_DIMNF", cluster_DIMNF)
    cv2.imshow("cluster_DIGF", cluster_DIGF)
    cv2.imshow("cluster_DIMDMNF", cluster_DIMDMNF)
    cv2.imshow("cluster_DIMDGF", cluster_DIMDGF)
    cv2.imshow("cluster_DIMNGF", cluster_DIMNGF)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
