from calendar import monthrange
from datetime import datetime
import os
import pytz
import time
import yaml


EXIT_CONFIG_FAILED = 2
EXIT_PM2_TARGET_NON_NUMERIC = 3

MIN_TO_SEC = 60
HOUR_TO_MIN = 60
DAY_TO_HOUR = 24
WEEK_TO_DAY = 7

def read_yaml_file(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
        return config['cron']['reloadCiCd']


def move_log_files():
    with open('tigas.yaml', 'r') as f:
        config = yaml.safe_load(f)
    log_path = config['tigas']['generate']['tti']['path']['log']

    dir_for_prev_logs = f'{log_path}/prev' if log_path.endswith('/') else f'{log_path}prev'
    dir_for_parse_logs = f'{log_path}/parse' if log_path.endswith('/') else f'{log_path}parse'
    
    today = datetime.astimezone(datetime.now(), pytz.timezone('Asia/Seoul'))
    time_str = today.strftime('%Y%m%d')
    prev_tody_log_dir = f'{dir_for_prev_logs}/{time_str}'

    if not os.path.exists(dir_for_prev_logs):
        os.makedirs(dir_for_prev_logs)
    if not os.path.exists(dir_for_parse_logs):
        os.makedirs(dir_for_parse_logs)
    if not os.path.exists(prev_tody_log_dir):
        os.makedirs(prev_tody_log_dir)

    # copy log files to parse directory
    os.system(f'cp {dir_for_parse_logs}/*.log {prev_tody_log_dir}')
    # move log files to save directory
    os.system(f'mv {prev_tody_log_dir}/*.log {dir_for_prev_logs}')

# ----------------- Reload functions for pm2 -----------------

def filter_pm2_target_list(targets: list):
    filtered_list = list(filter(lambda x: x.isnumeric(), targets))
    if len(filtered_list) != len(targets):
        raise ValueError('Invalid input. All elements must be numeric.')
    return filtered_list

def reload_pm2(targets: list):
    try:
        filtered_list = filter_pm2_target_list(targets)
        list(map(lambda x: os.system(f'pm2 reload {x}'), filtered_list))
    except ValueError as e:
        print(e)
        exit(EXIT_PM2_TARGET_NON_NUMERIC)

def stop_pm2(targets: list):
    try:
        filtered_list = filter_pm2_target_list(targets)
        list(map(lambda x: os.system(f'pm2 stop {x}'), filtered_list))
    except ValueError as e:
        print(e)
        exit(EXIT_PM2_TARGET_NON_NUMERIC)

# ------------------------------------------------------------


def reload_ci_cd():
    config = read_yaml_file('cron.yaml')
    daemon_config = config['daemon']
    used_daemon = daemon_config['use']

    if used_daemon == 'systemd':
        print('systemd')
        raise NotImplementedError('Reload logic for systemd is not implemented yet.')
    elif used_daemon == 'supervisor':
        print('supervisor')
        raise NotImplementedError('Reload logic for supervisor is not implemented yet.')
    elif used_daemon == 'pm2':
        print('pm2')
        pm2_config = daemon_config['pm2']
        pm2_target_services = pm2_config['services']
        
        stop_pm2(pm2_target_services)
        move_log_files()
        reload_pm2(pm2_target_services)


def calculate_daily_interval():
    return MIN_TO_SEC * HOUR_TO_MIN * DAY_TO_HOUR

def calculate_weekly_interval():
    return calculate_daily_interval() * WEEK_TO_DAY

def calculate_monthly_interval():
    today = datetime.astimezone(datetime.now(), pytz.timezone('Asia/Seoul'))
    num_of_months = monthrange(today.year, today.month)[1]
    return calculate_daily_interval() * num_of_months

def configure_cron(cycle:str):
    if cycle == 'daily':
        print('Activate cron job daily')
        interval = calculate_daily_interval()
    elif cycle == 'weekly':
        print('Activate cron job weekly')
        interval = calculate_weekly_interval()
    elif cycle == 'monthly':
        print('Activate cron job monthly')
        interval = calculate_monthly_interval()
    else:
        raise ValueError('Invalid cycle value. Please check cron.yaml file.')
    return interval

def main():
    config = read_yaml_file('cron.yaml')
    cycle = config['cycle']
    try:
        interval = configure_cron(cycle)
    except ValueError as e:
        print('Cycle configuration failed. Please check cron.yaml file.')
        exit(EXIT_CONFIG_FAILED)

    while True:
        time.sleep(interval)
        reload_ci_cd()

if __name__ == '__main__':
    main()

