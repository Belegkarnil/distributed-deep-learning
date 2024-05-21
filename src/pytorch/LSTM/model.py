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

import torch

class ExtractOutputFromLSTM(torch.nn.Module):
    def __init__(self):
        super(ExtractOutputFromLSTM, self).__init__()
    def forward(self, x):
        out,(x,cell) = x
        return out

class ExtractFinalStateFromLSTM(torch.nn.Module):
    def __init__(self):
        super(ExtractFinalStateFromLSTM, self).__init__()
    def forward(self, x):
        out,(x,cell) = x
        x.squeeze_(0)
        return x

class Model(torch.nn.Sequential): #Module):
    def __init__(self,hidden_layers=1,hidden_params=128,classes=5,devices=[torch.device('cpu')],pipeline_size=None):
        super(Model, self).__init__()
        if(devices is None and pipeline_size is not None):
            ValueError("To apply pipeline, model parallelism is required. Need to specify a devices list")
        if(devices is not None and not isinstance(devices,list)):
            raise TypeError("The devices argument must be a list of device")
        if(devices is None):
            raise TypeError("Devices are not defined")
        if(hidden_layers < 1):
            raise TypeError("Model requires at least one hidden layer")
        layers = [self]
        partitions = self.model_partition(hidden_layers+3,1)
        self.devices = devices
        if(len(self.devices)>1):
            layers = [torch.nn.Sequential() for _ in self.devices]
            for layer in layers: self.append(layer)
            partitions = self.model_partition(hidden_layers+3,len(self.devices))
        if(pipeline_size is not None):
            self.pipeline_size	= pipeline_size
            self.forward			= self.pipelinedModelParallelismForward
        elif(len(self.devices) > 1):
            self.forward = self.modelParallelismForward
        self.partitions = partitions
        #one-dimensional convolution layer produces a three-dimensional output vector (None, 10, 64), with 64 being the size of the convolution layer filters
        #Convolutional layer filters	64
        #Convolutional kernel size	1
        #Convolutional layer activation function	ReLU
        #Convolutional layer padding	Same
        #		self.conv		= torch.nn.Conv1d(10, 64, 1, stride=1, padding='same', bias=True) # in: N,C,L, out:N,C,L
        #		self.conva		= torch.nn.ReLU(inplace=True)
        layer_id	 = 0
        layers[partitions[layer_id]].append(torch.nn.Conv1d(10, 64, 1, stride=1, padding='same', bias=True).to(self.deviceForLayer(layer_id))) # in: N,C,L, out:N,C,L
        layers[partitions[layer_id]].append(torch.nn.ReLU(inplace=True).to(self.deviceForLayer(layer_id)))
        #the pooling layer, where it is converted into a three-dimensional output vector (None, 10, 64)
        #Pooling layer pool size	1
        #Pooling layer padding	Same
        #Pooling layer activation function	ReLU
        layer_id += 1
        layers[partitions[layer_id]].append(torch.nn.MaxPool1d(1, stride=None, padding=0).to(self.deviceForLayer(layer_id))) # TODO ensure padding same
        layers[partitions[layer_id]].append(torch.nn.ReLU(inplace=True).to(self.deviceForLayer(layer_id)))
        #The output vector is then fed into the LSTM layer for training (128 is the number of hidden units in the LSTM layer)
        layer_id += 1
        layers[partitions[layer_id]].append(torch.nn.LSTM(32,hidden_size=hidden_params,num_layers=1,bias=True,batch_first=True,dropout=0,bidirectional=False).to(self.deviceForLayer(layer_id)))
        for i in range(1,hidden_layers):
            layers[partitions[layer_id]].append(ExtractOutputFromLSTM().to(self.deviceForLayer(layer_id)))
            layer_id += 1
            layers[partitions[layer_id]].append(torch.nn.LSTM(input_size=hidden_params,hidden_size=hidden_params,num_layers=1,bias=True,batch_first=True,dropout=0,bidirectional=False).to(self.deviceForLayer(layer_id)))
        layers[partitions[layer_id]].append(ExtractFinalStateFromLSTM().to(self.deviceForLayer(layer_id)))
        #Number of LSTM hidden cells	128
        #Number of skip connections	2
        #LSTM activation function	tanh
        #Skip connections are added between LSTM cells that receive input from the previous layers.
        # The input to the fully connected layer is the hidden states of the skip connection at time t, which is represented
        layer_id += 1
        # layers[partitions[layer_id]].append(ExtractOutputFromLSTM().to(self.deviceForLayer(layer_id)))
        layers[partitions[layer_id]].append(torch.nn.Linear(hidden_params, classes).to(self.deviceForLayer(layer_id)))
    #and the output data (None, 128) from the previous is fed into another complete connection layer after training to acquire the output value
    def deviceForLayer(self,layerID):
        return self.devices[self.partitions[layerID]]
    def model_partition(self,layers,ndevices):
        if(layers == ndevices):
            return {i:i for i in range(layers)}
        nhidden	= layers - 3
        step		= nhidden // ndevices
        rest		= nhidden - step*ndevices
        split		= {0:0}
        #
        current			= step
        partition_id	= 0
        if(step < 1):
            partition_id += 1
            current = 1
        for (lstm_id, layer_id) in enumerate(range(2,nhidden+2)):
            split[layer_id] = partition_id
            current -= 1
            if(current < 1):
                current = step
                partition_id += 1
                if(rest > 0):
                    current	+= 1
                    rest		-= 1
        # last layer
        split[layers-1]	= min(ndevices-1, max(split.values()) + 1)
        # second layer (pooling)
        split[1] = (split[2] - split[0])//2
        return split
    def modelParallelismForward(self,input):
        for (i,module) in enumerate(self):
            input = module(input.to(self.devices[i]))
        return input
    def pipelinedModelParallelismForward(self,input):
        # The process is divided in 3 steps : load, process and flush
        # 1) The load step is loading one chunk per layer if enough chunks
        # 2) The process step is loading a new chunk until all chunk are on a layer (all layers processes at each iter)
        # 3) the flush step is to flow remaining chunks that already are on a layer (all chunks on a layer, the last one is on the first layer)
        #
        chunks			 	= input.split(self.pipeline_size, dim=0)
        nchunks				= len(chunks)
        minLayersChunks	= min(nchunks, len(self))
        chunks				= iter(chunks)
        microbatch			= []
        on_devices			= [None] * len(self)
        # load step: load min(len(chunks), len(nodes)) chunks
        for chunk_id in range(minLayersChunks):
            # load chunk_id on first (0) layer after flowing previous chunks on their next layer
            for layerID in range(chunk_id,0,-1): # move previous chunks
                on_devices[layerID] = self[layerID](on_devices[layerID-1].to(self.devices[layerID]))
            # load chunk_id on layer 0
            layerID = 0
            on_devices[layerID] = self[layerID](next(chunks).to(self.devices[layerID]))
        # Now we know that either all the chunks are loaded on a layer (more layers than chunks) => go to the flush step
        # or each layer has at least one chunk => go to the process step
        #
        # if less layers than chunk, it exists a chunk that is already fully processed
        # back it up into microbatch
        if(on_devices[-1] is not None):
            microbatch.append(on_devices[-1]) # NOTE we can reset microbatch[-1] to None but it will be overwritten
        #
        # the process step (more chunks than layers)
        for chunk in chunks: # <=> for chunk_id in range(minLayersChunks,nchunks):
            # backup output of the last layer into microbatch
            microbatch.append(on_devices[-1]) # NOTE we can reset microbatch[-1] to None but it will be overwritten
            # move each chunk from last layer to first layer
            for layerID in range(len(self)-1,0,-1): # move previous chunks
                on_devices[layerID] = self[layerID](on_devices[layerID-1].to(self.devices[layerID]))
            # then load a new chunk on the first layer
            layerID = 0
            on_devices[layerID] = self[layerID](chunk.to(self.devices[layerID]))
        #
        # the flush step (all chunks on a layer, the last one is on the first layer)
        minLayersChunks += 1 # void to compute +1 at each step (second loop)
        for lastChunkIsOnLayer in range(len(self)-1):
            start = min(len(self),lastChunkIsOnLayer+minLayersChunks) # here minLayersChunks+1 but already inc previously
            for layerID in range(start-1,lastChunkIsOnLayer,-1):
                on_devices[layerID] = self[layerID](on_devices[layerID-1].to(self.devices[layerID]))
            # backup output of the last layer into microbatch if required
            if(start == len(self)):
                microbatch.append(on_devices[-1]) # NOTE we can reset microbatch[-1] to None but it will be overwritten
        #
        return torch.cat(microbatch)

if(__name__ == '__main__'):
    hidden_layers = 3
    ndevices = 4
    devices = [torch.device('cpu')] * ndevices
    batch_size = 2
    features_size = 32
    time_size = 10
    model = Model(hidden_layers=hidden_layers,devices=devices)
    print(model)
    rnd = torch.Generator(device='cpu').manual_seed(42)
    input = torch.rand((batch_size,time_size,features_size), generator=rnd)
    output = model(input)
    print(output.shape)
