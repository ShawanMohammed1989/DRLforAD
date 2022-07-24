#!/bin/sh
set -e
set -o pipefail

osx=false
tag="latest"
gpu_option=""
cpus_option=""
display=$DISPLAY

# functions
usage() {
    echo '---------------- help -------------------'
    echo '-h, --help                Show this help.'
    echo '-o, --osx		            Set display to "host.docker.internal:0" and your IP to the X access control list of the container'
    echo '-d, --display DISPLAY     Speficy a display for X11.          (-e DISPLAY=$DISPLAY)'
    echo '-g, --gpu                 Hand over GPUs to Container.        (--gpus=all -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all)'
    echo '-c, --cpus decimal        Limit number of CPUs.               (--cpus decimal)'
    echo 'Note: Different options can be combined.'
}



while [ "${1+defined}" ]; do # Simple and safe loop over arguments: https://wiki.bash-hackers.org/scripting/posparams
key="$1"
shift
case $key in
    -o|--osx)
    display=host.docker.internal:0
    osx=true
    ;;
    -h|--help)
    usage >&2;
    exit
    ;;
    -g| --gpus)
    tag="latest-gpu"
    gpu_option="--gpus=all -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all"
    ;;
    -c| --cpus)
    cpus_option="--cpus=${1}"
    shift
    ;;
    -d|--display)
    display=${1}
    shift
    ;;
    *)
    echo "Unknown option or parameter $key"
    usage >&2;
    echo "Exiting..."
    exit
    ;;
esac
done

# Stop any running container named 'trainerAI'
echo "Gracefully stopping docker container. Killing after 10s..."
echo "If no container was running, this errors out."

docker stop ray_rllib > /dev/null || true

echo "done"
echo "Running new docker container..."

# Run the image as a container with the specified option strings
docker run -it -d --rm \
        --shm-size="4g" \
        --memory="32g" \
        --name ray_rllib \
        -e DISPLAY=$display \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $(pwd)/code:/mnt/code \
        -v $(pwd)/results:/mnt/results \
        $cpus_option \
        $webcam_option \
        $gpu_option \
        pub-gitlab.iss.rwth-aachen.de:5050/niederfahrenhorst/rllib-for-students-at-the-ice:$tag > /dev/null

if $osx; then
	docker exec -it ray_rllib sh -c "export DISPLAY=host.docker.internal:0 && xhost + $(ifconfig en0 | grep 'inet[ ]' | awk '{print$2}')"
fi

echo "done"

