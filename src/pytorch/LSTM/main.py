#!/usr/bin/env python
#coding: utf-8

#  Copyright 2024 Belegkarnil
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#  associated documentation files (the “Software”), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do
#  so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial
#  portions of the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
#  FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
#  OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
os.environ['TZ']='UTC'
os.environ['MASTER_ADDR']			= 'localhost'
os.environ['MASTER_PORT']			= '29500'
os.environ['GLOO_SOCKET_IFNAME']	= 'lo'
os.environ['NCCL_SOCKET_IFNAME']	= 'lo'

from datetime import datetime
import sys
import argparse

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from model import *
from dataset import *

#N_LAYER		= 1
#EPOCHS		= 10
#BATCH_SIZE	= 32
#DEVICE		= 'gpu'
#N_WORKERS	= 0
#MODE			= 'sequential'
#PIPELINE		= 5
#SIZE			= 128

def getConfiguration():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-l", "--nlayers",    dest='N_LAYER',   type=int, nargs='?', default=1, help="Number of hidden layers")
    argParser.add_argument("-s", "--size",    	dest='SIZE',   type=int, nargs='?', default=128, help="Hidden size")
    argParser.add_argument("-e", "--epochs",     dest='EPOCHS', type=int, nargs='?', default=10, help="Number of epochs")
    argParser.add_argument("-b", "--batch",      dest='BATCH_SIZE', type=int, nargs='?', default=32, help="Batch size")
    argParser.add_argument("-d", "--device",     dest='DEVICE', choices=['cpu','gpu'], nargs='?', default='cpu', help="Compute device")
    argParser.add_argument("-w", "--nworkers",   dest='N_WORKERS', type=int, nargs='?', default=0, help="Number of workers")
    argParser.add_argument("-m", "--mode",       dest='MODE', choices=['sequential','model','pipeline','data'], nargs='?', default='sequential', help="Running mode")
    argParser.add_argument("-p", "--pipeline",   dest='PIPELINE', type=int, nargs='?', default=2, help="Pipeline size")
    argParser.add_argument("-r", "--run",   		dest='GLOBAL_WORLD', type=int, nargs='?', default=1, help="Global rank (only use when mode is data without MPI)")
    #
    args = argParser.parse_args(sys.argv[1:]).__dict__
    args['GLOBAL_RANK']	= 0
    args['LOCAL_RANK'],args['LOCAL_WORLD']		= args['GLOBAL_RANK'],args['GLOBAL_WORLD']
    args['DISTRIBUTED']=len(list(filter(lambda x:'MPI_' in x,os.environ)))>0
    if(args['DISTRIBUTED']):
        if('OMPI_COMM_WORLD_RANK' in os.environ): args['GLOBAL_RANK']=int(os.environ['OMPI_COMM_WORLD_RANK'])
        if('OMPI_COMM_WORLD_SIZE' in os.environ): args['GLOBAL_WORLD']=int(os.environ['OMPI_COMM_WORLD_SIZE'])
        if('OMPI_COMM_WORLD_LOCAL_RANK' in os.environ): args['LOCAL_RANK']=int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        if('OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ): args['LOCAL_WORLD']=int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    return args

def generate_sampler(seed,indices):
    seed			= int(seed)
    generator	= torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.SubsetRandomSampler(indices, generator=generator)

def worker(model,out_device,criterion,optimizer,epochs,trainset,validationset,testset,sync,verbose=False):
    for epoch in range(1,epochs+1):
        # train
        model.train()
        if(verbose): print('"train epoch %d begins at %f"' % (epoch,datetime.now().timestamp()))
        total_loss, total_accuracy, counter = 0.0, 0.0, 0
        for (x,y) in trainset:
            y = y.to(out_device)
            optimizer.zero_grad()
            prediction		 = model(x)
            loss				 = criterion(prediction,y)
            loss.backward()
            optimizer.step()
            accuracy			 = prediction.argmax(dim=1) == y.argmax(dim=1)
            total_accuracy	+= accuracy.int().sum()
            total_loss		+= loss.item()
            counter			+= len(x)
        total_accuracy	= total_accuracy*100.0/counter
        total_loss		= total_loss/counter
        if(verbose): print('"train epoch %d ends at %f with accuracy %0.03f and loss %0.09f"' % (epoch,datetime.now().timestamp(),total_accuracy,total_loss))
        # validation
        model.eval()
        total_loss, total_accuracy, counter = 0.0, 0.0, 0
        with torch.no_grad():
            for (x,y) in validationset:
                y = y.to(out_device)
                prediction		 = model(x)
                loss				 = criterion(prediction,y)
                accuracy			 = prediction.argmax(dim=1) == y.argmax(dim=1)
                total_accuracy	+= accuracy.int().sum()
                total_loss		+= loss.item()
                counter			+= len(x)
        total_accuracy	= total_accuracy*100.0/counter
        total_loss		= total_loss/counter
        if(verbose): print('"validation epoch %d ends at %f with accuracy %0.03f and loss %0.09f"' % (epoch,datetime.now().timestamp(),total_accuracy,total_loss))
    # test
    model.eval() # not required but it's clean :-)
    total_loss, total_accuracy, counter = 0.0, 0.0, 0
    with torch.no_grad():
        for (x,y) in testset:
            y = y.to(out_device)
            prediction		 = model(x)
            loss				 = criterion(prediction,y)
            accuracy			 = prediction.argmax(dim=1) == y.argmax(dim=1)
            total_accuracy	+= accuracy.int().sum()
            total_loss		+= loss.item()
            counter			+= len(x)
    total_accuracy	= total_accuracy*100.0/counter
    total_loss		= total_loss/counter
    if(verbose): print('"test ends at %f with accuracy %0.03f and loss %0.09f"' % (datetime.now().timestamp(),total_accuracy,total_loss))

def process(rank,world,backend,config):
    if(backend is not None):
        torch.distributed.init_process_group(backend=backend, world_size=world,rank=rank)
    torch.manual_seed(42)
    sync=(lambda model: None)
    if(config['DISTRIBUTED']):
        favg	= float(config['GLOBAL_WORLD'])
        def all_reduce(model):
            for param in model.parameters():
                torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
                param.grad.data /= favg
        sync = all_reduce
    elif(config['MODE']=='data'):
        config['GLOBAL_RANK'] = config['LOCAL_RANK'] = rank
    devices = []
    if(config['DEVICE'] == 'cpu'):
        devices.append(torch.device('cpu'))
    else:
        devices.append(torch.device('cuda:0'))
    if(config['MODE'] in ['model','pipeline']):
        if(config['DEVICE'] != 'gpu'): raise ValueError('model and pipeline modes require to use a GPU')
        devices.append(torch.device('cuda:1'))
    elif(config['MODE'] == 'data' and config['DEVICE'] == 'gpu'):
        devices = [torch.device(f"cuda:{config['LOCAL_RANK']:d}") ]
    in_device	= devices[0]
    out_device	= devices[-1]
    pipeline_size	= config['PIPELINE'] if(config['MODE'] == 'pipeline')else None
    #
    model			= Model(hidden_layers=config['N_LAYER'],hidden_params=config['SIZE'],devices=devices,pipeline_size=pipeline_size)
    # model			= torch.compile(model) # TODO compile pytorch with USE_TENSORRT
    criterion	= torch.nn.L1Loss()
    optimizer	= torch.optim.Adam(model.parameters())
    #
    dataset		= Dataset(device=in_device)
    #
    indices				= torch.randperm(len(dataset)).to(in_device)
    train_end			= int(len(dataset)*0.7)
    validation_end		= int(len(dataset)*0.1) + train_end
    #
    trainset			= generate_sampler(torch.randint(low=0,high=torch.iinfo(torch.int64).max,size=(1,)), indices[:train_end])
    validationset	= generate_sampler(torch.randint(low=0,high=torch.iinfo(torch.int64).max,size=(1,)), indices[train_end:validation_end])
    testset			= generate_sampler(torch.randint(low=0,high=torch.iinfo(torch.int64).max,size=(1,)), indices[validation_end:])
    #
    trainset			= torch.utils.data.distributed.DistributedSampler(trainset,num_replicas=world,rank=rank,shuffle=False)
    validationset	= torch.utils.data.distributed.DistributedSampler(validationset,num_replicas=world,rank=rank,shuffle=False)
    testset			= torch.utils.data.distributed.DistributedSampler(testset,num_replicas=world,rank=rank,shuffle=False)
    #
    trainset			= torch.utils.data.DataLoader(dataset,batch_size=config['BATCH_SIZE'],sampler=trainset,shuffle=False,num_workers=config['N_WORKERS'],pin_memory=False) #,pin_memory_device=device) # TODO can change num_workers (0=main_thread)
    validationset	= torch.utils.data.DataLoader(dataset,batch_size=config['BATCH_SIZE'],sampler=validationset,shuffle=False,num_workers=config['N_WORKERS'],pin_memory=False) #,pin_memory_device=device) # TODO can change num_workers (0=main_thread)
    testset			= torch.utils.data.DataLoader(dataset,batch_size=config['BATCH_SIZE'],sampler=testset,shuffle=False,num_workers=config['N_WORKERS'],pin_memory=False) #,pin_memory_device=device) # TODO can change num_workers (0=main_thread)
    #
    worker(model,out_device,criterion,optimizer,config['EPOCHS'],trainset,validationset,testset,sync,rank==0)
    #
    if(torch.distributed.is_initialized()):
        torch.distributed.barrier()


def main():
    config = getConfiguration()
    backend = None
    if(config['DISTRIBUTED']):
        if(config['DEVICE'] == 'gpu'):
            os.environ['MASTER_ADDR']			= 'rtx2080-1.mit'
            os.environ['NCCL_SOCKET_IFNAME']	= 'enp3s0'
            torch.distributed.init_process_group(backend=torch.distributed.Backend.NCCL,world_size=config['GLOBAL_WORLD'],rank=config['GLOBAL_RANK'])
        #os.environ['RANK'] = str(config['GLOBAL_RANK'])
        #os.environ['WORLD_SIZE'] = str(config['GLOBAL_WORLD'])
        #torch.distributed.init_process_group(backend='cuda:mpi')
        else:
            torch.distributed.init_process_group(backend=torch.distributed.Backend.MPI)
    elif(config['MODE']=='data'):
        if(config['DEVICE'] == 'cpu'):
            backend = torch.distributed.Backend.GLOO
        else:
            backend = torch.distributed.Backend.NCCL
        torch.multiprocessing.spawn(process, args=(config['GLOBAL_WORLD'],backend,config), nprocs=config['GLOBAL_WORLD'],join=True)
        return
    process(config['GLOBAL_RANK'],config['GLOBAL_WORLD'],backend,config)

if( __name__ == '__main__'):
    main()
