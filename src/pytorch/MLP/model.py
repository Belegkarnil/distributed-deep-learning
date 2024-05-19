#coding: utf-8

import torch

class Model(torch.nn.Sequential):
    def __init__(self,input_size=52,hidden_layers=1,hidden_size=38,classes=5,devices=[torch.device('cpu')],pipeline_size=None):
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
        partitions = self.model_partition(hidden_layers+2,1)
        self.devices = devices
        if(len(self.devices)>1):
            layers = [torch.nn.Sequential() for _ in self.devices]
            for layer in layers: self.append(layer)
            partitions = self.model_partition(hidden_layers+2,len(self.devices))
        if(pipeline_size is not None):
            self.pipeline_size	= pipeline_size
            self.forward			= self.pipelinedModelParallelismForward
        elif(len(self.devices) > 1):
            self.forward = self.modelParallelismForward
        self.partitions = partitions
        # already defined in the super class
        #else:
        #	self.forward = self.sequentialForward
        layer_id	 = 0
        layers[partitions[layer_id]].append(torch.nn.Linear(input_size,hidden_size).to(self.deviceForLayer(layer_id)))
        layers[partitions[layer_id]].append(torch.nn.ReLU(inplace=True).to(self.deviceForLayer(layer_id)))
        layer_id	+= 1
        for i in range(0,hidden_layers):
            layers[partitions[layer_id]].append(torch.nn.Linear(hidden_size,hidden_size).to(self.deviceForLayer(layer_id)))
            layers[partitions[layer_id]].append(torch.nn.ReLU(inplace=True).to(self.deviceForLayer(layer_id)))
            layer_id	+= 1
        layers[partitions[layer_id]].append(torch.nn.Linear(hidden_size,classes).to(self.deviceForLayer(layer_id)))
        layers[partitions[layer_id]].append((torch.nn.Sigmoid() if(classes < 2)else torch.nn.Softmax(dim=-1)).to(self.deviceForLayer(layer_id)))
    def deviceForLayer(self,layerID):
        return self.devices[self.partitions[layerID]]
    def model_partition(self,layers,ndevices):
        step,rest = layers//ndevices, layers % ndevices
        shift = 1 if(rest > 1)else 0
        last = rest - shift
        first = ndevices-last
        sep_layer = first*step+shift
        #
        split = {0:0}
        #
        for layerID in range(shift,sep_layer):
            split[layerID] = (layerID-shift)//step
        step += 1
        for layerID in range(sep_layer,layers):
            split[layerID] = first + (layerID-sep_layer)//step
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
