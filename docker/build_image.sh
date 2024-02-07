function build_image() {
    curr=$(pwd)
    pushd ${curr}
    docker build -t dataelement/bisheng-ft:0.0.1.rc0 \
        -f "$curr/docker/bisheng_ft.Dockerfile" . --no-cache
    popd
}

function export_image() {
    image="dataelement/bisheng-ft:0.0.1.rc0"
    filename="bisheng_ft_0.0.1.rc0.tar"
    docker save -o $filename $image
    curl -k --progress-bar -o /dev/null --user 'admin:2DxY9wRGKephebOP' --upload-file $filename http://192.168.106.244:5081/repository/product/bisheng/$filename
    echo "succ"
}


function build_docker() {
    IMAGE="dataelement/bisheng-ft:0.0.1.rc0"
    MOUNT="-v $HOME:$HOME -v /home/public:/home/public"
    docker run --gpus=all --net=host -itd --shm-size=10G \
        --name bisheng-ft-dev-001rc0 ${MOUNT} $IMAGE bash
}

function rm_docker() {
    docker stop -f bisheng-ft-dev
    docker rm -f bisheng-ft-dev
}

# build_image
# export_image    
build_docker
# rm_docker