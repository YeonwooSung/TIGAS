import os
import time

TARGET_HOURS = 2
SLEEP_MINUTES = 5

HOUR_TO_MINUTES = 60
MINUTES_TO_SECONDS = 60
TIME_THRESHOLD = HOUR_TO_MINUTES * MINUTES_TO_SECONDS * TARGET_HOURS

SLEEP_TIME = MINUTES_TO_SECONDS * SLEEP_MINUTES

IMG_DIR_PATH = '/home/ys60/images/'

# check if IMG_DIR_PATH exists
if not os.path.isdir(IMG_DIR_PATH):
    os.mkdir(IMG_DIR_PATH)


def delete_old_files():
    files = os.listdir(IMG_DIR_PATH)
    for file in files:
        # get the file's creation time
        ctime = os.path.getctime(os.path.join(IMG_DIR_PATH, file))

        # if the file is older than threshold value, delete it
        if time.time() - ctime > TIME_THRESHOLD:
            os.remove(os.path.join(IMG_DIR_PATH, file))


def run_every_n_seconds(n):
    while True:
        delete_old_files()
        time.sleep(n)

if __name__ == '__main__':
    run_every_n_seconds(SLEEP_TIME)
