# COMP3888_M_15_04 
# P31 - UG-USYD-02A: Improved AI/ML Communication - COMPSCI

## Group Members' Details

| Name | Email |
|------|-------|
| Reeves Spivak     | rspi3338@uni.sydney.edu.au | 
| Robert Ji         | yuji6835@uni.sydney.edu.au |
| Alex Milne        | amil3234@uni.sydney.edu.au | 
| Barrett Greaves   | bgre9982@uni.sydney.edu.au | 
| Reynardo Tjhin    | rtjh9350@uni.sydney.edu.au | 
| Adam Zhao         | azha6173@uni.sydney.edu.au |

## Project Background

Tens of billions of dollars is spent on deep learning hardware each year, however much of it is unnecessary as many algorithms can produce the same results much faster on lower spec hardware when optimized.

## Project Description

Our aim of the project:

- Review existing practices
- Profiling memory access (C/CUDA)
- Implement and profile new methods of memory access. (C/CUDA)

## Project Documentation: [Link](https://docs.google.com/document/d/1orEtajZJCvoooa7g1TOrYk64s5x_XDaHd9PcKQy3Psc/edit?usp=sharing)

## Description

<b>Model</b>: resnet50 <br>
<b>Dataset</b>: TinyImagenet <br>
<b>Communication Backend</b>: NCCL <br>
<b>Package used</b>: PyTorch

## Running or Launching the project

You can run the Dockerfile using this command
```
sudo docker run --rm --gpus all --network host \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -it -v $(pwd):/workspace nvcr.io/nvidia/pytorch:22.07-py3
```
or simply run
```
./launch_docker.sh
```

If "launch_docker.sh" cannot run, you might need to give permission to execute the file.
```
chmod +x launch_docker.sh
```

Dockerfile will install Nvidia PyTorch Image and install python3 and all the required software in `requirements.txt`.

Afterwards, you will be running our program in a container
```
python3 res50.py -b {batch_size} -e {epoch_size} -g {no_of_gpus} -lr {learning_rate}
```

Once the training has been completed, you can use Tensorboard to detect the model's performance
Write this command in the terminal
```
tensorboard --logdir=./log
```
Once the tensorboard has finished loading (it will look like it is stuck), use your desired web browser and go to this link
```
http://localhost:6006/
```
It will launch an interactive html of Tensorboard.

To install PyTorch's profiling tool
```
pip3 install torch_tb_profiler
```

Note: We use Tensorboard as our profiling tool and is not installed by Docker upon launch.


## Structure or Descriptions of our Files

- `res50.py`: this file contains code for training on the datasets, TinyImageNet, using the model "resnet50" by using NCCL as the preferred communication backend across GPU.
- `RestructureValData.py`: this file restructure the initial structure of the TinyImageNet from the internet to suit our needs.
- `SetupTinyImagenet.sh`: downloads the dataset, TinyImageNet


## Software Programs that are used

- Docker, version = 20.10.12
- python3, version = 3.8.10
- torch, version = 1.12.1 + cu116
- torchvision, version = 0.13.1
- tensorboard, version = 2.9.1
- torch-tb-profiler = 0.4.0
# NCCL_repo_template
