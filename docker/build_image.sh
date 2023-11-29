function build_image() {
    curr=$(pwd)
    pushd ${curr}
    docker build -t dataelement/bisheng-ft:0.0.1 \
        -f "$curr/docker/bisheng_ft.Dockerfile" . --no-cache
    popd
}

function build_docker() {
    IMAGE="dataelement/bisheng-ft:0.0.1"
    MOUNT="-v $HOME:$HOME -v /home/public:/home/public"
    docker run --gpus=all --net=host -itd --shm-size=10G \
        --name bisheng-ft-dev ${MOUNT} $IMAGE bash
}

function rm_docker() {
    docker stop -f bisheng-ft-dev
    docker rm -f bisheng-ft-dev
}

# build_image
build_docker
# rm_docker