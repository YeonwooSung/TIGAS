import uvicorn
from server import Server
from api import signin_router


if __name__ == '__main__':
    server = Server()
    try:
        server.add_endpoint('/signin', 'signin', signin_router)
    except:
        print('failed to add endpoints')
    uvicorn.run(server.app, host="0.0.0.0", port=5000)
