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
import libxml2
import torch
import torchvision
#use file + annotation
#random pixel shit between 5-10 (data augmentation)
#resize to 64x64 pixel
#

from torchvision.datasets.folder import find_classes, default_loader, VisionDataset

def bounding_boxes(path):
    doc = libxml2.parseFile(path)
    ctxt = doc.xpathNewContext()
    res = ctxt.xpathEval("/annotation/object/bndbox")
    results = []
    res = [tuple([int(x.xpathEval(k)[0].content) for k in ['xmin','xmax','ymin','ymax']]) for x in res]
    doc.freeDoc()
    ctxt.xpathFreeContext()
    return res

def make_dataset(directory,class_to_idx):
    directory	= os.path.expanduser(directory)
    annotation	= os.path.join(directory[:directory.rindex(os.sep)],'Annotations')
    is_valid_file = lambda x: x.endswith('.jpg')
    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    boxes = bounding_boxes(os.path.join(annotation,target_class,os.path.splitext(fname)[0]+'.xml'))
                    for box in boxes:
                        item = path, box, class_index
                        instances.append(item)
                        if target_class not in available_classes:
                            available_classes.add(target_class)
    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)
    return instances

class Dataset(VisionDataset):
    def __init__(self,root='/data/PCB_DATASET/',
                 transform=None,target_transform=None, # TODO
                 generator=torch.Generator(),device=torch.device('cpu')):
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = find_classes(os.path.join(self.root,'Annotations'))
        samples = make_dataset(os.path.join(self.root,'images'), class_to_idx)
        #
        self.shift = torch.randint(low=5,high=11,size=(len(samples)<<1,)).tolist()
        #transform  = transforms = torch.nn.Sequential(
        #	transforms.CenterCrop(10)
        #)
        #
        self.loader			= default_loader
        self.extensions	= ['.jpg']
        self.classes		= classes
        self.class_to_idx	= class_to_idx
        self.samples		= samples
        self.targets		= [s[1] for s in samples]
        self.device			= device
    def __len__(self):
        return len(self.samples)<<1
    def __getitem__(self, index):
        path,(xmin,ymin,xmax,ymax), target	= self.samples[index>>1]
        shift											= self.shift[index]
        top,left										= ymin+shift,xmin+shift
        height,width								= ymax-ymin, xmax-xmin
        #
        sample = self.loader(path)
        sample = torchvision.transforms.functional.resized_crop(sample,top,left,height,width,size=(64,64),interpolation=torchvision.transforms.InterpolationMode.BILINEAR,antialias=True) # if box is out, padded with 0
        #  InterpolationMode.NEAREST, InterpolationMode.NEAREST_EXACT, InterpolationMode.BILINEAR and InterpolationMode.BICUBIC
        #
        #if self.transform is not None:
        #	sample = self.transform(sample)
        #if self.target_transform is not None:
        #	target = self.target_transform(target)
        sample = torchvision.transforms.functional.pil_to_tensor(sample).float().to(self.device)
        target = torch.nn.functional.one_hot(torch.tensor(target), num_classes=len(self.classes)).float().to(self.device)
        # print(sample)
        # print(target)
        return sample, target

if(__name__ == '__main__'):
    train=3597
    val=1161
    test=1148
    print('Data augmentation produces %d images' % (train+val+test))
    print('=> double les images')
    print('In the paper: grayscale...')
    ds = Dataset()
    print(len(ds))
    x,y = ds[0]
    #
    print(x)
    print(y)
    print(type(x))
    print(type(y))


