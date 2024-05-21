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

"""
Based on pytorch implementation of DenseNet (https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/densenet.py)
Customized like the source available at https://raw.githubusercontent.com/MukundSai7907/PCB-Defects-Classification-Using-Deep-Learning/main/Code/Train.py
"""

#import re
from collections import OrderedDict
#from functools import partial
#from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

#from ..transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
#from ._api import register_model, Weights, WeightsEnum
#from ._meta import _IMAGENET_CATEGORIES
#from ._utils import _ovewrite_named_param, handle_legacy_interface

class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate,self).__init__()
    def forward(self, inputs: list) -> Tensor:
        return torch.cat(inputs, 1)

class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int):
        super(DenseLayer,self).__init__(
            Concatenate(),
            nn.BatchNorm2d(num_input_features,eps=0.001,momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate,eps=0.001,momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False
    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        def closure(*inputs):
            return self.bn_function(inputs)
        return cp.checkpoint(closure, *input)

class WrapperTriton(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int):
        super(WrapperTriton,self).__init__()
        self.layer = DenseLayer(num_input_features, growth_rate, bn_size)
        self.add_module('DenseLayer',self.layer)
    def forward(self,input: list) -> Tensor:
        return input + [ self.layer(input) ]

class DenseBlock(nn.Module):
    def __init__(self,num_layers: int,num_input_features: int,bn_size: int,growth_rate: int):
        super(DenseBlock,self).__init__()
        #self.layers = [ DenseLayer(num_input_features + i * growth_rate,growth_rate=growth_rate,bn_size=bn_size) for i in range(num_layers) ]
        self.layers = [ WrapperTriton(num_input_features + i * growth_rate,growth_rate=growth_rate,bn_size=bn_size) for i in range(num_layers) ]
        self.concat = Concatenate()
        for (i,layer) in enumerate(self.layers):
            self.add_module(f"WrapperTriton_{i:d}",layer)
        self.add_module('Concatenate',self.concat)
    def forward(self,input: Tensor) -> Tensor:
        features = [input]
        for layer in self.layers:
            features = layer(features)
        return self.concat(features)

class Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__(
            nn.BatchNorm2d(num_input_features,eps=0.001,momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

class Model(nn.Sequential):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        num_classes (int) - number of classification classes
    """
    def __init__(
            self,
            growth_rate:	int = 32,
            dense_blocks:	int = 2,# kind of hidden_layers
            dense_layers:	int = 6,
            bn_size:			int = 4,
            classes:			int = 6,
            devices				 = [torch.device('cpu')],
            pipeline_size		 = None
    ):
        super(Model, self).__init__()
        if(devices is None and pipeline_size is not None):
            ValueError("To apply pipeline, model parallelism is required. Need to specify a devices list")
        if(devices is not None and not isinstance(devices,list)):
            raise TypeError("The devices argument must be a list of device")
        if(devices is None):
            raise TypeError("Devices are not defined")
        if(dense_blocks < 1):
            raise TypeError("Model requires at least one dense block")
        layers = [self]
        nlayers = 3 + (((dense_blocks-1)<<1) + 1) + 2
        partitions = self.model_partition(nlayers,1)
        self.devices = devices
        if(len(self.devices)>1):
            layers = [torch.nn.Sequential() for _ in self.devices]
            for layer in layers: self.append(layer)
            partitions = self.model_partition(nlayers,len(self.devices))
        if(pipeline_size is not None):
            self.pipeline_size	= pipeline_size
            self.forward			= self.pipelinedModelParallelismForward
        elif(len(self.devices) > 1):
            self.forward = self.modelParallelismForward
        self.partitions = partitions
        layer_id	 = 0
        #
        _log_api_usage_once(self)
        num_init_features = growth_rate << 1
        block_config = [dense_layers] * dense_blocks
        # First convolution
        layers[partitions[layer_id]].append(nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False).to(self.deviceForLayer(layer_id)))
        layer_id += 1
        layers[partitions[layer_id]].append(nn.BatchNorm2d(num_init_features,eps=0.001,momentum=0.99).to(self.deviceForLayer(layer_id)))
        layers[partitions[layer_id]].append(nn.ReLU(inplace=True).to(self.deviceForLayer(layer_id)))
        layer_id += 1
        layers[partitions[layer_id]].append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(self.deviceForLayer(layer_id)))
        # Each denseblock
        num_features = num_init_features
        for i in range(len(block_config)-1):
            num_layers = block_config[i]
            layer_id += 1
            layers[partitions[layer_id]].append(DenseBlock(num_layers=num_layers,num_input_features=num_features,bn_size=bn_size,growth_rate=growth_rate).to(self.deviceForLayer(layer_id)))
            #
            num_features = num_features + num_layers * growth_rate
            layer_id += 1
            layers[partitions[layer_id]].append(Transition(num_input_features=num_features, num_output_features=num_features >> 1).to(self.deviceForLayer(layer_id)))
            #
            num_features >>= 1
        i = len(block_config)-1
        num_layers = block_config[i]
        layers[partitions[layer_id]].append(DenseBlock(num_layers=num_layers,num_input_features=num_features,bn_size=bn_size,growth_rate=growth_rate).to(self.deviceForLayer(layer_id)))
        layer_id += 1
        #
        num_features = num_features + num_layers * growth_rate
        layer_id += 1
        layers[partitions[layer_id]].append(nn.AvgPool2d(kernel_size=7).to(self.deviceForLayer(layer_id)))
        layers[partitions[layer_id]].append(nn.Flatten(start_dim=1).to(self.deviceForLayer(layer_id)))
        layer_id += 1
        layers[partitions[layer_id]].append(nn.Linear(num_features, classes).to(self.deviceForLayer(layer_id)))
        layers[partitions[layer_id]].append(nn.Softmax(dim=-1).to(self.deviceForLayer(layer_id)))# TODO check if crossentropy already includes softmax
        # Official init from torch repo.
        for m in self.modules():
            if(isinstance(m, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
            elif(isinstance(m, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif(isinstance(m, nn.Linear)):
                nn.init.constant_(m.bias, 0)
    def deviceForLayer(self,layerID):
        return self.devices[self.partitions[layerID]]
    def model_partition(self,layers,ndevices):
        # TODO impl better but ok for current available machines
        if(ndevices == 1):
            return {i:0 for i in range(layers)}
        #TODO currenlty always 8,1 or 8,2
        return {i:i//4 for i in range(layers)}
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
    from torchvision import transforms
    import PIL
    import logging
    import torch._dynamo
    # torch._dynamo.config.log_level = logging.DEBUG
    torch._dynamo.config.verbose = True
    dev = torch.device('cuda:0')
    for (layer,sample) in [
        (Concatenate(),[torch.rand((1,64)),torch.rand((1,64))]),
        (DenseLayer(num_input_features=2, growth_rate=32, bn_size=4), [torch.rand((1,1,64,64)),torch.rand((1,1,64,64))]),
        (Model(devices=[dev]),torch.rand((1,3,64,64)))
    ] :
        model = layer.to(dev)
        if(type(sample) is list):
            sample = [x.to(dev) for x in sample]
        else:
            sample = sample.to(dev)
        model = torch.compile(model)
        model(sample)
    #
    img = PIL.Image.open('test.jpg')
    img = transforms.ToTensor()(img)
    img = transforms.CenterCrop(64)(img)
    img = img.unsqueeze_(0)
    dev = torch.device('cuda:0')
    model = Model(devices=[dev])
    BATCH,HEIGHT,WIDTH,CHANNEL=1,64,64,3 # FIXME grayscale to rgb
    sample = img # torch.rand((BATCH,CHANNEL,HEIGHT,WIDTH))
    sample = sample.to(dev)
    # model = torch.compile(model,backend="eager")
    model = torch.compile(model)
    explanation, out_guards, graphs, ops_per_graph,x, y = torch._dynamo.explain(model,sample)
    print(explanation)
    # print(torch.jit.trace(model,sample))
    model(sample)
# print(model)

