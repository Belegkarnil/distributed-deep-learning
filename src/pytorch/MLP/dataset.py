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
    def __init__(self,path='/data/MQTT/dataset.csv',device=torch.device('cpu')):
        super().__init__()
        header	= list(pandas.read_csv(path, nrows=1))
        header.pop(0)
        self.data_frame = pandas.read_csv(path,usecols=header,low_memory=False,dtype='float32') #, nrows=400)
        self.device	= device
    def __len__(self):
        return len(self.data_frame)
    def __getitem__(self, idx):
        # check None case (pos == None)
        if(type(idx) is torch.Tensor): idx=idx.tolist()
        data = self.data_frame.iloc[idx].values
        return torch.tensor(data[:-5],device=self.device), torch.tensor(data[-5:],device=self.device)

if(__name__ == '__main__'):
    ds		= Dataset()
    last	= len(ds)-1
    print(ds[last])
#real    1m41.250s
#user    1m30.600s
#sys     0m13.002s
