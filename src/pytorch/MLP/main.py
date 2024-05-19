#!/usr/bin/env python
#coding: utf-8
import os
os.environ['TZ']='UTC'
from datetime import datetime
import sys
import argparse

import torch

from model import *
from dataset import *

N_LAYER		= 1
EPOCHS		= 10
BATCH_SIZE	= 32
DEVICE		= 'gpu'
N_WORKERS	= 0
MODE			= 'sequential'
PIPELINE		= 5

argParser = argparse.ArgumentParser()
argParser.add_argument("-l", "--nlayers",    dest='N_LAYER',   type=int, nargs='?', default=1, help="Number of hidden layers")
argParser.add_argument("-e", "--epochs",     dest='EPOCHS', type=int, nargs='?', default=10, help="Number of epochs")
argParser.add_argument("-b", "--batch",      dest='BATCH_SIZE', type=int, nargs='?', default=32, help="Batch size")
argParser.add_argument("-d", "--device",     dest='DEVICE', choices=['cpu','gpu'], nargs='?', default='cpu', help="Compute device")
argParser.add_argument("-w", "--nworkers",   dest='N_WORKERS', type=int, nargs='?', default=0, help="Number of workers")
argParser.add_argument("-m", "--mode",       dest='MODE', choices=['sequential','model','pipeline'], nargs='?', default=1, help="Running mode")
argParser.add_argument("-p", "--pipeline",   dest='PIPELINE', type=int, nargs='?', default=2, help="Pipeline size")

args = argParser.parse_args(sys.argv[1:]).__dict__
#argParser.print_help()
progVar = locals()
for var in args:
    if(var in progVar):
        progVar[var] = args[var]

devices = [torch.device('cpu' if(DEVICE == 'cpu')else 'cuda:0')]
if(MODE in ['model','pipeline']): devices.append(torch.device('cuda:1'))
in_device	= devices[0]
out_device	= devices[-1]
pipeline_size	= PIPELINE if(MODE == 'pipeline')else None

model			= Model(input_size=48, hidden_layers=N_LAYER,devices=devices,pipeline_size=pipeline_size)
model			= torch.compile(model)
criterion	= torch.nn.CrossEntropyLoss().to(out_device)
optimizer	= torch.optim.Adam(model.parameters())

dataset		= Dataset(device=in_device)

torch.manual_seed(42)

def generate_sampler(seed,indices):
    seed			= int(seed)
    generator	= torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.SubsetRandomSampler(indices, generator=generator)


indices				= torch.randperm(len(dataset)).to(in_device)
train_end			= int(len(dataset)*0.7)
validation_end		= int(len(dataset)*0.1) + train_end

trainset			= generate_sampler(torch.randint(low=0,high=torch.iinfo(torch.int64).max,size=(1,)), indices[:train_end])
validationset	= generate_sampler(torch.randint(low=0,high=torch.iinfo(torch.int64).max,size=(1,)), indices[train_end:validation_end])
testset			= generate_sampler(torch.randint(low=0,high=torch.iinfo(torch.int64).max,size=(1,)), indices[validation_end:])

trainset			= torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,sampler=trainset,shuffle=False,num_workers=N_WORKERS,pin_memory=False) #,pin_memory_device=device) # TODO can change num_workers (0=main_thread)
validationset	= torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,sampler=validationset,shuffle=False,num_workers=N_WORKERS,pin_memory=False) #,pin_memory_device=device) # TODO can change num_workers (0=main_thread)
testset			= torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,sampler=testset,shuffle=False,num_workers=N_WORKERS,pin_memory=False) #,pin_memory_device=device) # TODO can change num_workers (0=main_thread)

for epoch in range(1,EPOCHS+1):
    # train
    model.train()
    print('"train epoch %d begins at %f"' % (epoch,datetime.now().timestamp()))
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
    print('"train epoch %d ends at %f with accuracy %0.03f and loss %0.09f"' % (epoch,datetime.now().timestamp(),total_accuracy,total_loss))
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
    print('"validation epoch %d ends at %f with accuracy %0.03f and loss %0.09f"' % (epoch,datetime.now().timestamp(),total_accuracy,total_loss))

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
print('"test ends at %f with accuracy %0.03f and loss %0.09f"' % (datetime.now().timestamp(),total_accuracy,total_loss))

