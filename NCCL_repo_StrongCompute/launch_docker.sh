sudo docker run \
    --runtime=nvidia \
    --cap-add=SYS_ADMIN \
    --privileged \
    --rm \
    --gpus all \
    --network host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -it -v $(pwd):/workspace nvcr.io/nvidia/pytorch:22.07-py3

# --rm : Remove container after use
# --gpus all : Give container access to host gpus
#		(requires nvidia-container-toolkit)
# --network host: Connects container to host network
#		very important when setting master address to host ip
# --ipc, --ulimit memlock/stack : Not sure what these do yet
#		pytorch container recommended setting them like this
# -it : Interactive Terminal
# -v : Mount working directory in workspace directory in container
# nvcr.io/nvidia/pytorch:22.07-py3 : nvidia pytorch docker image
