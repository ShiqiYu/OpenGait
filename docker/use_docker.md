# Run OpenGait with Docker
This guide will show you how to run OpenGait using Docker. Docker is a platform for developing, shipping, and running applications in containers. Containers allow a developer to package up an application with all parts it needs, such as libraries and other dependencies, and ship it all out as one package.

### Note:
This a work in progress solution. Please note that inside the docker container the default user is root. This is not recommended for security reasons.

## Prerequisites
- Docker installed on your machine. You can download Docker from [here](https://docs.docker.com/desktop/setup/install/linux/). 
- Docker should not require sudo permissions to run. You can follow the instructions [here](https://docs.docker.com/engine/install/linux-postinstall/) to run Docker without sudo permissions.
- NVIDIA Driver installed on your machine. You can download the driver from [here](https://www.nvidia.com/Download/index.aspx).
- CUDA Toolkit installed on your machine. You can download the toolkit from [here](https://developer.nvidia.com/cuda-downloads).
- NVIDIA Container Toolkit installed on your machine. You can install the toolkit by following the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- Clone the OpenGait repository.


## Running OpenGait with Docker
### Step 0: Add execute permissions to the scripts
```bash
sudo chmod +x docker/build_docker.sh docker/run_docker.sh
```
If you need to enable cuda for OpenCV - unlikely - then follow the instructions in the Dockerfile to enable it.
### Step 1: Build the Docker image
1. Open a terminal and navigate to the OpenGait repository.
2. Run the following command to build the Docker image:
```bash
./docker/build_docker.sh
```
This builds the docker image based on the cu116.Dockerfile in the repository.
### Step 2: Run the Docker container
1. Open the run script.
```bash
docker/run_docker.sh
```
2. Modify the script to set the correct paths for the OpenGait repository and the dataset you want to use. These will be attached as volumes to the Docker container.
3. Run the script to start the Docker container.
```bash
./docker/run_docker.sh
```
This will start the Docker container and open a shell inside the container.
