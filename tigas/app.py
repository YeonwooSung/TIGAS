import threading
import uvicorn
import yaml

from server import Server
from api import signin_router
from api.utils import delete_all
from diffusion import inference_loop


def parse_config():
    with open('tigas.yaml') as f:
        parsed_yaml = yaml.safe_load(f)
        tigas_conf = parsed_yaml['tigas']
        port = int(tigas_conf['port'] if 'port' in tigas_conf else '5000')
        host = tigas_conf['host'] if 'host' in tigas_conf else '0.0.0.0'
        name = tigas_conf['name'] if 'name' in tigas_conf else 'tigas'
        reload = tigas_conf['reload'] if 'reload' in tigas_conf else False
        debug = tigas_conf['debug'] if 'debug' in tigas_conf else False
        return port, host, name, reload, debug



if __name__ == '__main__':
    print('Clean up all cached uuids in redis...')
    delete_all()

    port, host, name, reload, debug = parse_config()
    server = Server(name=name)
    try:
        server.add_endpoint('/signin', 'signin', signin_router)
    except:
        print('failed to add endpoints')

    # make a new thread with inference_loop function
    inference_thread = threading.Thread(target=inference_loop)
    inference_thread.start()

    uvicorn.run(
        server.app, 
        host=host, 
        port=port, 
        reload=reload, 
        debug=debug,
    )
