# -*- coding: utf-8 -*-

#导入相关模块
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os

class Fonter(Dataset): #继承Dataset
    def __init__(self, srcPath, targetPath, transform=None): # __init__是初始化该类的一些基础参数
        self.src_Data = np.load(srcPath) # src_data 存了font数据
        self.target_Data = np.load(targetPath)# target_data 存了要变成的font的数据
        self.transform = transform
    
    def __len__(self):#返回整个数据集的大小
        assert len(self.src_Data) == len(self.target_Data)
        return min(len(self.src_Data), len(self.target_Data))
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        src = self.src_Data[index]
        target = self.target_Data[index]

        if self.transform:
            src = self.transform(src)
            target = self.transform(target)

        return src, target

# 验证用dataloader来试着取一个batch
if __name__=='__main__':
    data = Fonter('C:/Users/G.G s computer/Desktop/Coding/Summer2021/Rewrite-master/path_to_save_bitmap/src.npy', 'C:/Users/G.G s computer/Desktop/Coding/Summer2021/Rewrite-master/path_to_save_bitmap/tgt.npy', 0, 2000)#初始化类，设置数据集所在路径以及变换
    dataloader = DataLoader(data,batch_size=128,shuffle=True)#使用DataLoader加载数据
    for batch_id,batch_data in enumerate(dataloader):
        print(batch_id)#打印batch编号
        print(batch_data[0].size())#打印该batch里面图片的大小
        print(batch_data[1].size())#打印该batch里面图片的标签（大小）