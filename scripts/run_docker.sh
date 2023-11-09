run_container() {
  echo "run docker image"
  DOCKER_IMAGE="nvcr.io/nvidia/pytorch:22.08-py3"
  DEV_MOUNT="-v /home/$(whoami):/home/$(whoami) -v /mnt/disk0/$(whoami):/mnt/disk0/$(whoami) -v /home/public:/home/public -v /nfs_106_2:/nfs_106_2"
  container_name="llama_factory_glx"

  docker run --pid=host --name ${container_name} ${DEV_MOUNT} --net=host -itd --ipc=host --gpus all $DOCKER_IMAGE bash
  if [ $? -eq 0 ]; then
      echo "run  docker image success"
  else
      echo "run docker image failed!"
      exit 1
  fi
}

run_container