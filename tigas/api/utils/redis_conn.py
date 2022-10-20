import redis
import json


SERVICE_LIST = ['tti', 'i2i']


class RedisClient(object):
    def __init__(self, host='localhost', port=6379, db=0):
        self._client = redis.StrictRedis(host=host, port=port, db=db)

    def get(self, key):
        return self._client.get(key)
    
    def get_all_keys(self):
        return self.keys('*')
    
    def get_all_values(self):
        return [self.get(key) for key in self.get_all_keys()]

    def set(self, key, value):
        self._client.set(key, value)

    def delete(self, key):
        self._client.delete(key)

    def exists(self, key):
        return self._client.exists(key)

    def keys(self, pattern):
        return self._client.keys(pattern)

    def flushall(self):
        self._client.flushall()

    def flushdb(self):
        self._client.flushdb()
    
    def close(self):
        self._client.close()

    def __repr__(self):
        return f'<RedisClient {self._client}>'

    def __str__(self):
        return self.__repr__()

    def __del__(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._client.close()



def register_uuid_with_prompt(prompt:str, uuid:str, service:str='tti'):
    if service not in SERVICE_LIST:
        raise ValueError(f'Invalid service: {service}')
    client = RedisClient()
    data = {
        'prompt': prompt,
        'service': service,
        'status': 0,
    }
    data_str = json.dumps(data)
    client.set(uuid, data_str)
    client.close()


def get_service_info_by_uuid(uuid:str):
    client = RedisClient()
    data_str = client.get(uuid)
    client.close()
    data = json.loads(data_str)
    if data is not None and 'service' in data:
        return data['service']
    return None

def get_status_info_by_uuid(uuid:str):
    client = RedisClient()
    data_str = client.get(uuid)
    client.close()
    data = json.loads(data_str)
    if data is not None and 'status' in data:
        return data['status']
    return None


def update_user_status(uuid:str, status:int):
    client = RedisClient()
    data_str = client.get(uuid)
    if data_str is None:
        client.close()
        return
    data = json.loads(data_str)
    data['status'] = status
    data_str = json.dumps(data)
    client.set(uuid, data_str)
    client.close()

def update_user_status_ok(uuid:str):
    update_user_status(uuid, 1)

def update_user_status_error(uuid:str):
    update_user_status(uuid, -1)


def delete_all():
    client = RedisClient()
    client.flushdb()
    client.close()


if __name__ == '__main__':
    import uuid
    prompt = 'This is a test prompt'
    test_uuid = str(uuid.uuid4())
    test_uuid2 = str(uuid.uuid4())
    register_uuid_with_prompt(prompt, test_uuid)
    register_uuid_with_prompt(prompt, test_uuid2, service='i2i')
    service1 = get_service_info_by_uuid(test_uuid)
    service2 = get_service_info_by_uuid(test_uuid2)

    print(f'{test_uuid} is waiting for {service1} service')
    print(f'{test_uuid2} is waiting for {service2} service')

    test_cli = RedisClient()
    ret = test_cli.get('test_none')
    # ret is None, since the key does not exist
    print(ret)

    print('All keys:', test_cli.get_all_keys())
