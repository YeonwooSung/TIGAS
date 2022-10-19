# Environment settings

## Model

TODO

## Node.js

You should use node.js version 12.x or higher to run the latest version of the axios.

```bash
sudo apt update
sudo apt -y install curl dirmngr apt-transport-https lsb-release ca-certificates
curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
```

If you want to install node.js version 14.x, you can use the following command:

```bash
sudo apt update
sudo apt -y install curl dirmngr apt-transport-https lsb-release ca-certificates
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
```

### PM2

Before installing pm2, you should make sure that the node version is 10.x or higher.
However, the latest version of the axios requires 12.x or higher, so no need to worry about it.

```bash
sudo npm install pm2 -g
```

If you are already super user, then you can replace the sudo command.

## Redis

To activate the redis server, run the following command:

```bash
sudo systemctl enable redis-server --now
```
