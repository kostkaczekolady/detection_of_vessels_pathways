from os import listdir
from os.path import isfile, join
import cv2
import numpy as np

DRIVE_TRAIN_IMAGES = './DRIVE/training/images/'
DRIVE_TEST_IMAGES = './DRIVE/test/images/'
    
def get_files_list(path):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]

def get_green_channel(img_path):
    img = cv2.imread(img_path)
    _,g,_ = cv2.split(img)
    return g

def subtract_images(minuend, subtrahend):
    width = minuend.shape[1]
    height = minuend.shape[0]
    subtracted_image = np.zeros((height,width,1), np.uint8)
    for y in range(height):
        for x in range(width):
            new_pixel = minuend[y][x] - subtrahend[y][x]
            if new_pixel < 0:
                subtracted_image[y][x] = 0
            else:
                subtracted_image[y][x] = new_pixel
    return subtracted_image

images_path = get_files_list(DRIVE_TRAIN_IMAGES)
gc = get_green_channel(images_path[0])

mean = cv2.blur(gc, (5,5))
median = cv2.medianBlur(gc, 5)
gaussian = cv2.GaussianBlur(gc, (5,5), 0)

DIMDF = subtract_images(mean, gc)
DIMNF = subtract_images(median, gc)
DIGF = subtract_images(gaussian, gc)

cv2.imshow("image", gc);
cv2.imshow("mean", mean);
cv2.imshow("gaussian", gaussian);
cv2.imshow("median", median);
cv2.imshow("DIMDF", DIMDF);
cv2.imshow("DIMNF", DIMNF);
cv2.imshow("DIGF", DIGF);
cv2.waitKey();
cv2.destroyAllWindows()