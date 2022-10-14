import os
import time
import yaml


with open('cron.yaml', 'r') as f:
    config = yaml.safe_load(f)
    cron_cycle_config = config['cron']['cleanUpNodeAssets']['cycleInterval']
    _SLEEP_INTERVAL = cron_cycle_config['sleep']
    _ACTIVATE_INTERVAL = cron_cycle_config['activate']

TARGET_HOURS = _ACTIVATE_INTERVAL
SLEEP_MINUTES = _SLEEP_INTERVAL

HOUR_TO_MINUTES = 60
MINUTES_TO_SECONDS = 60
TIME_THRESHOLD = HOUR_TO_MINUTES * MINUTES_TO_SECONDS * TARGET_HOURS

SLEEP_TIME = MINUTES_TO_SECONDS * SLEEP_MINUTES

IMG_DIR_PATH = '/home/ys60/TIGAS_Web/assets/result/'

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
