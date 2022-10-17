import os
import time
import pandas as pd
import yaml


_TTI_KEYWORD = 'generated image for'
INFO_KEYWORD = 'INFO -'


def parse_config(path:str):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return config

def parse_cron_config():
    config = parse_config('cron.yaml')
    return config['cron']['parseLogs']

def parse_tigas_config():
    config = parse_config('tigas.yaml')
    return config['tigas']

def get_log_path():
    config = parse_tigas_config()
    return config['generate']['tti']['path']['log']

def calculate_interval():
    config = parse_cron_config()
    interval = config['interval']
    return interval * 3600


def parse_logs(logs_path:str):
    config = parse_cron_config()
    targetLogs = config['targetLog']

    def parse_log(log_path:str):
        df = pd.DataFrame(columns=['uuid', 'prompt'])
        with open(log_path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                
                try:
                    if _TTI_KEYWORD in line:
                        target_str = line.split(INFO_KEYWORD)[1].strip()
                        splitted = target_str.split(_TTI_KEYWORD)
                        uuid = splitted[0].strip().replace('user=', '')
                        prompt = splitted[1].strip().replace('"', '').strip()
                        df = df.append({'uuid': uuid, 'prompt': prompt}, ignore_index=True)
                except:
                    pass
        #TODO

    for log in targetLogs:
        log_path = f'{logs_path}/{log}'
        # check if log file exists
        if os.path.exists(log_path):
            parse_log(log_path)


def cleanup_parsed_logs(logs_dir_path:str):
    os.remove(f'rm -rf {logs_dir_path}/*')


def main():
    log_path = get_log_path()
    logs_dir = f'{log_path}/parse'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    interval = calculate_interval()
    while True:
        parse_logs(logs_dir)
        cleanup_parsed_logs(logs_dir)
        time.sleep(interval)


if __name__ == '__main__':
    main()
