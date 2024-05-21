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

import pandas
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self,path='/data/PredictiveMaintenance/dataset.csv',history=10,device=torch.device('cpu')):
        super().__init__()
        self.history		= history-1
        self.instancesPm  = 8759
        self.div          = self.instancesPm - self.history
        self.len				= (self.instancesPm-self.history) * 100
        header				= list(pandas.read_csv(path, nrows=1))
        self.data_frame	= pandas.read_csv(path,usecols=header,low_memory=False,dtype='float32') #, nrows=400)
        self.device			= device
    def __len__(self):
        return self.len
    def idx2pos(self,idx):
        machineID   = idx//self.div
        base        = machineID * self.instancesPm + self.history
        return base + (idx - machineID * self.div)
    def __getitem__(self, idx):
        # check None case (pos == None)
        if(type(idx) is torch.Tensor): idx=idx.tolist()
        idx = self.idx2pos(idx)
        data = self.data_frame.iloc[idx-self.history:idx+1].values
        return torch.tensor(data[:,:-5],device=self.device), torch.tensor(data[0,-5:],device=self.device)

