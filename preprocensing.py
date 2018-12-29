from os import listdir
from os.path import isfile, join

DRIVE_TRAIN_IMAGES = './DRIVE/training/images/'
DRIVE_TEST_IMAGES = './DRIVE/test/images/'
    
def get_files_list(path):
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]


images_path = get_files_list(DRIVE_TRAIN_IMAGES)
