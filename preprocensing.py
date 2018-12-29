from os import listdir
from os.path import isfile, join
import cv2

DRIVE_TRAIN_IMAGES = './DRIVE/training/images/'
DRIVE_TEST_IMAGES = './DRIVE/test/images/'
    
def get_files_list(path):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]

def get_green_channel(img_path):
    img = cv2.imread(img_path)
    b,g,r = cv2.split(img)
    return g


images_path = get_files_list(DRIVE_TRAIN_IMAGES)
gc = get_green_channel(images_path[0])


mean = cv2.blur(gc,(5,5))
gaussian = cv2.GaussianBlur(gc, (5,5), 0)
median = cv2.medianBlur(gc,5)

cv2.imshow("image", gc);
cv2.imshow("mean", mean);
cv2.imshow("gaussian", gaussian);
cv2.imshow("median", median);
cv2.waitKey();