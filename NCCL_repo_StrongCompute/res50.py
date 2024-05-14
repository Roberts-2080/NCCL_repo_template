import os
import torch
import time
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
from torchvision.io import read_image
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from datetime import datetime

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--batch', default=64, type=int, metavar='N',
						help='batch size')
	parser.add_argument('-lr', '--rate', default=1e-4, type=float, metavar='N',
						help='learning rate')
	parser.add_argument('-m', '--mom', default=0.8, type=float, metavar='N',
						help='momentum')
	parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
						help='number of data loading workers (default: 4)')
	parser.add_argument('-g', '--gpus', default=1, type=int,
						help='number of gpus per node')
	parser.add_argument('-nr', '--nr', default=0, type=int,
						help='ranking within the nodes')
	parser.add_argument('-e', '--epochs', default=2, type=int, metavar='N',
						help='number of total epochs to run')
	parser.add_argument('--cluster', default=9, type=int, metavar='N',
						help='cluster final ip number')
	parser.add_argument('--backend', default='nccl', type=str, metavar="nccl/gloo",
						help='set gpu backend to nccl or gloo')
	args = parser.parse_args()
	args.world_size = args.gpus * args.nodes
	os.environ['MASTER_ADDR'] = '192.168.10.' + str(args.cluster)
	os.environ['MASTER_PORT'] = '29500'
	mp.spawn(train, nprocs=args.gpus, args=(args,))

def train(gpu, args):
	rank = args.nr * args.gpus + gpu
	dist.init_process_group(backend=args.backend, init_method='env://', world_size=args.world_size, rank=rank)
	torch.manual_seed(0)
	model = resnet50()
	torch.cuda.set_device(gpu)
	model.cuda(gpu)
	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda(gpu)
	optimizer = torch.optim.SGD(model.parameters(), args.rate,  args.mom)
	# Wrap the model
	model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
	# Data loading code
	train_dataset =  ImageFolder("tiny-imagenet-200/train",
								  transform=ToTensor())

	train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
																	num_replicas=args.world_size,
																	rank=rank)

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=args.batch,
											   shuffle=False,
											   num_workers=0,
											   pin_memory=True,
											   sampler=train_sampler)

	start = datetime.now()
	total_step  = len(train_loader)
	total_samples = total_step * args.batch
	avg_images_sec = 0
	avg_epoch_time = 0
	avg_util = 0
	avg_mem = 0
	print("GPU: {}\t[Training Size: {}\tBatch size: {}\tTotal steps: {}]".format(gpu, total_samples, args.batch, total_step))

	# Create TensorBoard profiler
	prof = torch.profiler.profile(
		schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
		on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/TBtest'),
		record_shapes=True,
		with_stack=True)
	prof.start()

	for epoch in range(args.epochs):
		
		trained_samples = 0
		epoch_time = 0
		epoch_start = datetime.now()
		images_sec = avg_images_sec
		util = avg_util
		mem = avg_mem
		
		for step, (images, labels) in enumerate(train_loader):
			# profiler cycles
			# if i >= (1 + 1 + 3) * 2:
			# 	break

			images = images.cuda(non_blocking=True)
			labels = labels.cuda(non_blocking=True)
			# Forward pass
			outputs = model(images)
			loss = criterion(outputs, labels)

			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			prof.step()
			
			trained_samples += args.batch
			epoch_time = datetime.now() - epoch_start
			progress = ((trained_samples / total_samples) * 100)
			images_sec = trained_samples / (epoch_time.total_seconds())
			util = torch.cuda.utilization(gpu)
			mem = torch.cuda.memory_usage(gpu)
			gpu_state = "[GPU:{} UTIL:{}% MEM:{}%]\t[EPOCH:{}\tTIME:{:.2f}sec]\t\tPROGRESS: {:.2f}% [{}/{}]\tBatch {} of {}\tIMAGES: {:.2f}/sec\tLOSS: {:.4f}".format(
    gpu, util, mem, epoch, epoch_time.total_seconds(), progress, trained_samples, total_samples, step, total_step, images_sec, loss.item())
			print(gpu_state)
		
		avg_epoch_time = (avg_epoch_time + epoch_time.total_seconds()) / 2
		avg_images_sec = (avg_images_sec + images_sec) / 2
		avg_util = (avg_util + util) / 2
		avg_mem =  (avg_mem + mem) / 2
		
	time.sleep(0.5)
	print("GPU {} Finished: \tTraining complete in: {}\t[AVG IMAGES/SEC: {}\tAVG EPOCH TIME: {:.2f}/sec AVG UTIL: {:.2f}% AVG MEM: {:.2f}%]".format(
    gpu, datetime.now() - start, avg_images_sec, avg_epoch_time, avg_util, avg_mem))
	prof.stop()

if __name__ == '__main__':
	main()
