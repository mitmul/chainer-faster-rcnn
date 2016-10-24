#!/bin/bash

command_exists () {
    type "$1" &> /dev/null ;
}

if ! [ -x "$(command -v docker)" ]; then
    echo "docker command is not found"
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates
    sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
    echo "deb https://apt.dockerproject.org/repo ubuntu-trusty main" | sudo tee /etc/apt/sources.list.d/docker.list
    sudo apt-get install -y linux-image-extra-$(uname -r) linux-image-extra-virtual
    sudo apt-get update
    sudo apt-get install -y docker-engine
    sudo service docker start
    sudo groupadd docker
    sudo usermod -aG docker $USER
fi

if ! [ -x "$(command -v nvidia-docker)" ]; then
    echo "nvidia-docker command is not found"
    # Install nvidia-docker and nvidia-docker-plugin
    wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.0-rc.3/nvidia-docker_1.0.0.rc.3-1_amd64.deb
    sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

    # Test nvidia-smi
    nvidia-docker run --rm nvidia/cuda:7.5 nvidia-smi
fi

if [ ! -d caffe ]; then
    git clone https://github.com/BVLC/caffe.git
    cd caffe/docker; make cpu_standalone
fi

if [ ! -f ../data/VGG_ILSVRC_16_layers_deploy.prototxt ]; then
    cd ../data
    wget https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt
    cd ../docker
fi

if [ ! -f ../data/VGG_ILSVRC_16_layers.caffemodel ]; then
    cd ../data
    wget http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
    cd ../docker
fi
