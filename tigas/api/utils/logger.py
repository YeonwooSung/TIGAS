import datetime
import logging
import pytz


class Formatter(logging.Formatter):
    """override logging.Formatter to use an aware datetime object"""
    def converter(self, timestamp):
        dt = datetime.datetime.fromtimestamp(timestamp)
        tzinfo = pytz.timezone('Asia/Seoul')
        return tzinfo.localize(dt)
        
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec='milliseconds')
            except TypeError:
                s = dt.isoformat()
        return s


class StableLogger:
    def __init__(self, log_file_path:str, name: str='tigas'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        self.formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%a, %d %b %Y %H:%M:%S.%f %z")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)


    def log(self, message: str, level: str='info'):
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'debug':
            self.logger.debug(message)
        else:
            raise ValueError('Invalid log level')
