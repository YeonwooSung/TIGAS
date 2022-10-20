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

    df_config = config['dataframe']
    df_dir_path = df_config['path']
    df_file_name = df_config['name']
    df_file_format = df_config['format']
    if not os.path.exists(df_dir_path):
        os.makedirs(df_dir_path)
    df_path = f'{df_dir_path}{df_file_name}.{df_file_format}'

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
        return df

    for log in targetLogs:
        log_path = f'{logs_path}/{log}'
        # check if log file exists
        if os.path.exists(log_path):
            df = parse_log(log_path)

            # save to csv with append mode if file exists. otherwise, create new file            
            if os.path.exists(df_path):
                df.to_csv(df_path, mode='a', header=False, index=False)
            else:
                df.to_csv(df_path, index=False)


def cleanup_parsed_logs(logs_dir_path:str):
    try:
        if os.path.exists(logs_dir_path):
            os.remove(f'rm -rf {logs_dir_path}/*')
    except FileNotFoundError:
        print('Nothing to cleanup - file not found')


def main():
    log_path = get_log_path()
    logs_dir = f'{log_path}parse'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    interval = calculate_interval()
    while True:
        try:
            parse_logs(logs_dir)
            cleanup_parsed_logs(logs_dir)
        except Exception as e:
            print(e)
        time.sleep(interval)


if __name__ == '__main__':
    main()
