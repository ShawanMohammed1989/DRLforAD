#!/bin/sh

echo "Building the docker image from the original ray-ml image."

# Check if the we want to build with GPU support
read -p "Do you want to build with GPU support? (i.e.: Do you have an NVidia graphics accelerator?)" -n 1 -r
GPU=""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    GPU="-gpu"
fi

# Build the docker file
echo "Building our docker image from the original ray-ml:latest docker image."
docker build --build-arg GPU="$GPU" -t pub-gitlab.iss.rwth-aachen.de:5050/niederfahrenhorst/rllib-for-students-at-the-ice:latest$GPU -f Dockerfile .

echo "Note: If you want to push a docker to the registry, you will have to do this manually."
echo "Done"

