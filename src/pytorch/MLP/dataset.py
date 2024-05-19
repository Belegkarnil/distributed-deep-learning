#coding: utf-8
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
