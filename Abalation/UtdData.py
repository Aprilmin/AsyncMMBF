import numpy as np
import torch
import os
import _pickle as cPickle
from torch.utils.data import Dataset
class UTDdataset(Dataset):
    def __init__(self,**kwargs):
        self.labels=kwargs['labels']
        self.depth=kwargs['depth']
        self.inertial=kwargs['inertial']
        self.skeleton=kwargs['skeleton']
        self.color=kwargs['color']
        self.__normalize__()


    def __normalize__(self):
        self.depth[self.depth != self.depth] = 0
        self.inertial[self.inertial != self.inertial] = 0
        self.skeleton[self.skeleton != self.skeleton] = 0
        self.color[self.color!=self.color]=0
        self.depth=self.depth/np.max(self.depth)
        self.inertial=self.inertial/np.max(self.inertial)
        self.skeleton=self.skeleton/np.max(self.skeleton)
        self.color=self.color/np.max(self.color)



class LoadDataByIID():
    def __init__(self,args):
        self.args=args
        self.iid=args.iid
        self.client_num=args.client_num
        dirPath=r'..\UTD\data'

        trainPath=os.path.join(dirPath,'trainUTD.pkl')
        testPath=os.path.join(dirPath,'testUTD.pkl')
        validPath=os.path.join(dirPath,'validUTD.pkl')
        self.dataLen={
            key:0 for key in ['train','test','valid']
        }


        # self.frac=args.receive_num/args.client_num if self.args.isCentrial==False else 1
        self.frac=1
        # load test
        self.testLoader=self.getData(testPath,type='test')
        self.validLoader=self.getData(validPath,type='valid')


        # load train data by args
        if args.client_num%5==0:
            if args.isCentrial:
                self.trainLoader=self.getData(trainPath,type='train')
            else:
                self.client_data_frac=[0 for i in range(args.client_num)]
                if args.iid == 1:
                    print('generating iid data...')
                    self.clientData=self.splitIID1(trainPath)
                    self.clientValidData = self.splitIID1(validPath)
                elif args.iid==0:
                    print('generating non-iid data...')
                    self.clientData,self.clientValidData=self.splitIID0(trainPath,validPath)
        else:
            print('Make sure client_num is a multiple of 5')

    def UtdLoader(self,**kwargs):
        return torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(
                torch.tensor(kwargs['color']),
                torch.tensor(kwargs['depth']),
                torch.tensor(kwargs['inertial']),
                torch.tensor(kwargs['skeleton']),
                torch.tensor(kwargs['labels']).long()
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
            depth=data.depth[idx],
            inertial=data.inertial[idx],
            skeleton=data.skeleton[idx],
            labels=data.labels[idx]
        )

    def splitIID0(self,path,validPath):
        if self.args.Dirichlet:
            f = open(path, 'rb+')
            data = cPickle.load(f)
            validF = open(validPath, 'rb+')
            validData = cPickle.load(validF)
            client_repeat = self.args.client_num // 5
            client_data_Idxs = [[] for i in range(self.args.client_num)]
            client_valid_Idxs = [[] for i in range(self.args.client_num)]
            for labelIdx in range(self.args.classNum):
                labelIdxs = [np.flatnonzero(data.labels == labelIdx) for i in range(4*client_repeat)] if self.args.subnet_num in [3,1] else [np.flatnonzero(data.labels == labelIdx) for i in range(6*client_repeat)]
                labelValidIdxs = [np.flatnonzero(validData.labels == labelIdx) for i in range(4*client_repeat)] if self.args.subnet_num in [3, 1] else [np.flatnonzero(validData.labels == labelIdx) for i in range(6*client_repeat)]
                labelIdxs = np.hstack(labelIdxs)
                labelValidIdxs = np.hstack(labelValidIdxs)
                proportions = np.random.dirichlet(np.repeat(self.args.Naplha, self.args.client_num))

                proportionsTrain = (np.cumsum(proportions) * len(labelIdxs)).astype(int)[:-1]
                for userIdx, new_idx in enumerate(np.split(labelIdxs, proportionsTrain)):
                    client_data_Idxs[userIdx].extend(new_idx)
                proportionsValid = (np.cumsum(proportions) * len(labelValidIdxs)).astype(int)[:-1]
                for userIdx, new_idx in enumerate(np.split(labelValidIdxs, proportionsValid)):
                    client_valid_Idxs[userIdx].extend(new_idx)
            client_data = []
            client_valid_data = []
            for userIdx in range(self.args.client_num):
                client_data.append(self.UtdLoader(
                    color=data.color[client_data_Idxs[userIdx]],
                    depth=data.depth[client_data_Idxs[userIdx]],
                    inertial=data.inertial[client_data_Idxs[userIdx]],
                    skeleton=data.skeleton[client_data_Idxs[userIdx]],
                    labels=data.labels[client_data_Idxs[userIdx]]
                ))
                client_valid_data.append(self.UtdLoader(
                    color=data.color[client_valid_Idxs[userIdx]],
                    depth=data.depth[client_valid_Idxs[userIdx]],
                    inertial=data.inertial[client_valid_Idxs[userIdx]],
                    skeleton=data.skeleton[client_valid_Idxs[userIdx]],
                    labels=data.labels[client_valid_Idxs[userIdx]]
                ))
        else:
            f = open(path, 'rb+')
            data = cPickle.load(f)
            validF=open(validPath,'rb+')
            validData=cPickle.load(validF)
            client_repeat = self.args.client_num//5

            if self.args.subnet_num == 4:
                idxs = np.arange(len(data.labels))
                validIdxs=np.arange(len(validData.labels))
            elif self.args.subnet_num in [3, 1]:
                idxs = [i for j in range(4*client_repeat) for i in range(len(data.labels))]
                validIdxs = [i for j in range(4*client_repeat) for i in range(len(validData.labels))]
            elif self.args.subnet_num == 2:
                idxs = [i for j in range(6*client_repeat) for i in range(len(data.labels))]
                validIdxs = [i for j in range(6*client_repeat) for i in range(len(validData.labels))]
            idxs = idxs[:int(len(idxs) - len(idxs) % self.args.client_num)]
            validIdxs = validIdxs[:int(len(validIdxs) - len(validIdxs) % self.args.client_num)]
            client_data_idx = np.hsplit(np.array(idxs), self.args.client_num)
            client_validData_idx = np.hsplit(np.array(validIdxs), self.args.client_num)
            client_data = list(map(lambda idx: self.UtdLoader(
                color=data.color[idx],
                depth=data.depth[idx],
                inertial=data.inertial[idx],
                skeleton=data.skeleton[idx],
                labels=data.labels[idx]
            ), client_data_idx))

            client_valid_data = list(map(lambda idx: self.UtdLoader(
                color=validData.color[validIdxs],
                depth=validData.depth[validIdxs],
                inertial=validData.inertial[validIdxs],
                skeleton=validData.skeleton[validIdxs],
                labels=validData.labels[validIdxs]
            ), client_validData_idx))
            self.client_data_frac=list(map(lambda idx:len(idx)/len(idxs),client_data_idx))
        return client_data,client_valid_data

    def splitIID1(self,path):
        f = open(path, 'rb+')
        data = cPickle.load(f)
        # 索引列表
        client_repeat = self.args.client_num // 5
        if self.args.subnet_num == 4:
            idxs=np.arange(len(data.labels))
        elif self.args.subnet_num in [3,1]:
            idxs = [i for j in range(4*client_repeat) for i in range(len(data.labels))]
        elif self.args.subnet_num==2:
            idxs = [i for j in range(6*client_repeat) for i in range(len(data.labels))]

        np.random.shuffle(idxs)
        idxs=np.random.choice(idxs,len(idxs) - len(idxs) % self.args.client_num,replace=False)
        client_data_idx = np.hsplit(idxs, self.args.client_num)
        client_data = list(map(lambda idx: self.UtdLoader(
            color=data.color[idx],
            depth=data.depth[idx],
            inertial=data.inertial[idx],
            skeleton=data.skeleton[idx],
            labels=data.labels[idx]
        ), client_data_idx))
        self.client_data_frac = list(map(lambda idx: len(idx) / len(idxs), client_data_idx))

        return client_data


