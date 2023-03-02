# sads_online
#### implementation of the online Shop-floor Anomaly Detection System

## Getting started 
```bash

git clone https://github.com/souayb/sads_online.git 
cd sads_online 
```
### Using only `docker`
```bash
# build the docker image replace the $(your_tag) with your own tag
docker build -t your_tag . 
#running the container and mapping the internal port ( 8500 ) to 8000 ( you can change to your own port ), feel free to change the 
# detach mode 
docker run -p 8000:8500 your_tag -d 
```
### Using docker-compose 
#### Using the development mode
this allow to make changes directly by mounting your local volume to container in `docker-compose.yml`
```bash
#running the container in detach mode 
docker-compose up --build -d 
```

#### Production instance
Comment the this section in the `docker-compose.yml` file 
```bash
    volumes:
      - .:/app
```

      






