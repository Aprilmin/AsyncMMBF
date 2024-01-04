import numpy as np
import torch
import os
import _pickle as cPickle
from torch.utils.data import Dataset

class HA4Mdataset(Dataset):
    def __init__(self,**kwargs):
        self.labels=kwargs['labels']
        self.skeleton=kwargs['skeleton']
        self.color=kwargs['color']
        self.colorA=kwargs['colorA']


class LoadDataByIID():
    def __init__(self,args):
        self.args=args
        self.iid=args.iid
        self.client_num=args.client_num
        dirPath=r'.\data'
        trainPath=os.path.join(dirPath,'train.pkl')
        testPath=os.path.join(dirPath,'test.pkl')
        validPath=os.path.join(dirPath,'valid.pkl')
        self.dataLen={
            key:0 for key in ['train','test','valid']
        }


        # self.frac=args.receive_num/args.client_num if self.args.isCentrial==False else 1
        self.frac=1
        # load test
        self.testLoader=self.getData(testPath,type='test')



        # load train data by args
        if args.isCentrial:
            self.trainLoader=self.getData(trainPath,type='train')
        else:
            self.client_data_frac=[0 for i in range(args.client_num)]
            if args.iid == 1:
                print('generating iid data...')
                self.clientData,self.TrainData=self.splitIID1(trainPath)
                self.clientValidData,self.ValidData = self.splitIID1(validPath)
            elif args.iid==0:
                print('generating non-iid data...')
                self.clientData,self.clientValidData,self.TrainData,self.ValidData=self.splitIID0(trainPath,validPath)

    def UtdLoader(self,**kwargs):
        return torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.tensor(kwargs['color']),
                torch.tensor(kwargs['colorA']),
                torch.tensor(kwargs['skeleton']),
                torch.tensor(kwargs['labels']).long()
            ),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=0
        )
    def clientLoader(self,**kwargs):
        return torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.tensor(kwargs['bsIdx'])
            ),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=0
        )

    def getData(self,path,type='train'):
        f=open(path,'rb+')
        data=cPickle.load(f)
        idx=np.random.choice(range(len(data.labels)),int(self.frac*len(data.labels)),replace=False)
        self.dataLen[type]=len(idx)
        return self.UtdLoader(
            color=data.color[idx],
            colorA=data.colorA[idx],
            skeleton=data.skeleton[idx],
            labels=data.labels[idx]
        )

    def splitIID0(self,path,validPath):
        f = open(path, 'rb+')
        data = cPickle.load(f)
        validF=open(validPath,'rb+')
        validData=cPickle.load(validF)
        client_data_Idxs = [[] for i in range(self.args.client_num)]
        client_valid_Idxs = [[] for i in range(self.args.client_num)]
        idxsLen=0

        for labelIdx in range(self.args.classNum):
            labelIdxs = [np.flatnonzero(data.labels == labelIdx) for i in range(2)] if self.args.subnet_num ==3 else [np.flatnonzero(data.labels == labelIdx) for i in range(3)] if self.args.subnet_num ==2 else [np.flatnonzero(data.labels == labelIdx) for i in range(6)]
            labelValidIdxs = [np.flatnonzero(validData.labels == labelIdx) for i in range(2)] if self.args.subnet_num ==3 else [np.flatnonzero(validData.labels == labelIdx) for i in range(3)] if self.args.subnet_num ==2 else [np.flatnonzero(validData.labels == labelIdx) for i in range(6)]
            labelIdxs = np.hstack(labelIdxs)
            labelValidIdxs = np.hstack(labelValidIdxs)
            proportions = np.random.dirichlet(np.repeat(0.5, self.args.client_num))
            idxsLen+=len(labelIdxs)

            proportionsTrain = (np.cumsum(proportions) * len(labelIdxs)).astype(int)[:-1]
            for userIdx, new_idx in enumerate(np.split(labelIdxs, proportionsTrain)):
                client_data_Idxs[userIdx].extend(new_idx)
            proportionsValid = (np.cumsum(proportions) * len(labelValidIdxs)).astype(int)[:-1]
            for userIdx, new_idx in enumerate(np.split(labelValidIdxs, proportionsValid)):
                client_valid_Idxs[userIdx].extend(new_idx)

        client_data = []
        client_valid_data = []
        for userIdx in range(self.args.client_num):
            client_data.append(self.clientLoader(
                bsIdx=client_data_Idxs[userIdx]
            ))
            client_valid_data.append(self.clientLoader(
                bsIdx=client_valid_Idxs[userIdx]
            ))

        self.client_data_frac=list(map(lambda idx:len(idx)/idxsLen,client_data_Idxs))
        return client_data,client_valid_data,data,validData




    def splitIID1(self,path):
        f = open(path, 'rb+')
        data = cPickle.load(f)
        # 索引列表

        if self.args.subnet_num == 3:
            idxs = [i for j in range(2) for i in range(len(data.labels))]
        elif self.args.subnet_num==2:
            idxs = [i for j in range(3) for i in range(len(data.labels))]
        elif self.args.subnet_num==1:
            idxs = [i for j in range(6) for i in range(len(data.labels))]

        np.random.shuffle(idxs)
        idxs=np.random.choice(idxs,len(idxs) - len(idxs) % self.args.client_num,replace=False)
        client_data_idx = np.hsplit(idxs, self.args.client_num)
        client_data = list(map(lambda idx: self.clientLoader(
            bsIdx=idx,
        ), client_data_idx))
        self.client_data_frac = list(map(lambda idx: len(idx) / len(idxs), client_data_idx))

        return client_data,data


