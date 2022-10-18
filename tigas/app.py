import threading
import uvicorn
import yaml

from server import Server
from api import signin_router
from diffusion import inference_loop


def parse_config():
    with open('tigas.yaml') as f:
        parsed_yaml = yaml.safe_load(f)
        tigas_conf = parsed_yaml['tigas']
        port = int(tigas_conf['port'] if 'port' in tigas_conf else '5000')
        host = tigas_conf['host'] if 'host' in tigas_conf else '0.0.0.0'
        name = tigas_conf['name'] if 'name' in tigas_conf else 'tigas'
        return port, host, name



if __name__ == '__main__':
    port, host, name = parse_config()
    server = Server(name=name)
    try:
        server.add_endpoint('/signin', 'signin', signin_router)
    except:
        print('failed to add endpoints')

    # make a new thread with inference_loop function
    inference_thread = threading.Thread(target=inference_loop)
    inference_thread.start()

    uvicorn.run(server.app, host=host, port=port)
