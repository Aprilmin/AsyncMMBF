from model import BlockTrainerBlend
import torch
import torch.nn as nn
import os
import copy
import pandas as pd
import gc
import numpy as np
import tqdm
import sys
from math import log
import blending
import math
import numpy as np


class GlobalSchedule(object):
    def __init__(self,args,data_loader_test,dataset_len,client_data,data_loader_valid,validLodader,client_data_frac):
        self.clientglobalValid=0
        self.clientValid=0
        self.globalValid=0
        self.client_data=client_data
        self.args=args
        self.data_loader_test=data_loader_test
        self.data_loader_valid=validLodader
        self.clientValidData=data_loader_valid
        self.testDataSize=int(self.args.receive_num/self.args.client_num * dataset_len['test'])
        self.blockAtt = [1 for i in range(self.args.chunks)]
        self.client_data_frac = client_data_frac

        self.globalValidFlow=[[0 for j in range(self.args.chunks+1)] for i in range(self.args.epochs)]
        self.client_receive_global_timeFrame=np.zeros(self.args.client_num)
        self.newest_time=0
        self.newest_valid=np.zeros(int(self.args.chunks+1))


        self.local_train_correct_list = []
        self.transPar_list=[]


        # Global testing result

        # if self.args.splitScheme in ['\'mix\'', '\'Mix\'', '\'MIX\'', 'mix', 'MIX', 'Mix']:

        self.local_test_result = {
            key: []
            for key in ['A', 'V', 'T', 'A+V', 'A+T', 'V+T', 'A+V+T']
        }

        if self.args.isGPU and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.dev = 'cuda'
        else:
            self.dev = 'cpu'


        input_dims = [33, 709, 768]
        self.globalModel = BlockTrainerBlend(args, dev=self.dev,mm_dim=self.args.hidDim, chunks=self.args.chunks,rank=self.args.rank,dropout_input=self.args.dropIn,dropout_output=self.args.dropOut)

        self.globalModel.to(self.dev)
        self.optimizer = torch.optim.Adam(self.globalModel.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.cost = nn.CrossEntropyLoss(reduction='sum')
        self.global_weights = copy.deepcopy(self.globalModel.state_dict())
        self.client_global_weight = [copy.deepcopy(self.globalModel.state_dict()) for i in range(self.args.client_num)]
        # local calculate variables
        self.client_time = [np.random.uniform(0, 1.54 * args.step) for i in range(self.args.client_num)]
        if args.splitScheme.lower() in ['\'mix\'', 'mix']:
            if self.args.subnet_num == 3:
                self.client_subnet_flag = [[1, 1, 1] for i in range(self.args.client_num)]
            elif self.args.subnet_num == 2:
                self.client_subnet_flag = [[0, 1, 1] if i < int(self.args.client_num / 3) else [1, 0, 1] if i < int(2 * self.args.client_num / 3) else [1, 1, 0] for i in range(self.args.client_num)]
            elif self.args.subnet_num == 1:
                self.client_subnet_flag = [[1, 0, 0] if i < int(self.args.client_num / 3) else [0, 1, 0] if i < int(2 * self.args.client_num / 3) else [0, 0, 1] for i in range(self.args.client_num)]
            else:
                self.client_subnet_flag = [[1, 1, 1] if i % 7 == 0 else [1, 1, 0] if i % 7 == 1 else [1, 0, 1] if i % 7 == 2 else [0, 1,1] if i % 7 == 3 else [1, 0, 0] if i % 7 == 4 else [0, 1, 0] if i % 7 == 5 else [0, 0, 1] for i in range(self.args.client_num)]
        else:
            if self.args.subnet_num != np.sum(self.args.iFlag):
                print(
                    '\033[1;32m Please enter the iFlag equal the subnet_num if you sure the splitScheme is pure\033[0m')
                sys.exit(1)
            else:
                self.client_subnet_flag = [self.args.iFlag for i in range(self.args.client_num)]
        self.local_weights = []
        self.local_grads=[]
        self.participating_client_subnet = []
        self.IterationTime = 0.0
        self.local_train_loss_tmp = 0.0
        self.local_train_correct_tmp = 0
        self.total_size = 0
        self.client_staleness = [0 for i in range(self.args.client_num)]

        self.modelLayerSize={
            key:self.global_weights[key].numel() for key in self.global_weights.keys()
        }
        self.clientTransIdx = {
            key: [] for key in self.global_weights.keys()
        }

        self.blockCorrect=[0.0 for i in range(self.args.chunks+1)]

        self.blockLosses, self.blockWeightedLosses, self.blockCorrect=[],[],[]

        self.optimal_blending_weights=[1/(self.args.chunks+1) for i in range(self.args.chunks+1)] if self.args.fusionType.lower() in ['block','\'block\''] else [1/4 for i in range(4)]
        self.clientOG=[]
        self.blockOGRo=[0 for i in range(self.args.chunks)]
        self.aggregate_op=ServerUpdate()


    def GlobalIterate(self,global_epo):
        # initialize local calculate variables
        self.local_weights = []
        self.local_grads=[]
        self.participating_client_subnet = []
        self.IterationTime = 0.0
        self.local_train_loss_tmp = 0.0
        self.local_train_correct_tmp = 0
        self.total_size = 0

        # select participating client
        self.participating_client_list=self.SelectParticipateClient() if self.args.isAsync!=1 else np.argsort(self.client_time)[:1]
        # self.participating_client_list = self.SelectParticipateClient()
        self.client_staleness=list(map(lambda idx:0 if idx in self.participating_client_list else self.client_staleness[idx]+1,range(self.args.client_num)))

        # evaluate test dataset on server

        self.globalModel.load_state_dict(self.global_weights)
        self.evaluteModel(global_epo)
        gc.collect()

        # start local updating
        self.clientOG = []
        self.blockOG=[]
        transParSize=0
        self.clientTransIdx = {
            key: [] for key in self.global_weights.keys()
        }

        pbar = tqdm.tqdm(range(len(self.participating_client_list)),desc='Global Epoch {} / {}：'.format(global_epo + 1, self.args.epochs), ncols=100)
        for idx in pbar:
            client_idx = self.participating_client_list[idx].item()
            client_local_time = self.client_time[client_idx]
            self.participating_client_subnet.append(self.client_subnet_flag[client_idx])
            self.IterationTime = max(self.IterationTime, client_local_time)
            transParSize+=self.LocalUpdateBlend(client_idx,pbar,idx,global_epo)

        self.transPar_list.append(transParSize)



        self.local_train_correct_list.append(self.local_train_correct_tmp / len(self.participating_client_list))




        # update client processed time
        self.client_time=list(map(lambda idx:
                                  self.client_time[idx]-self.IterationTime if (idx not in self.participating_client_list) and (self.client_time[idx]>self.IterationTime) else
                                  0 if (idx not in self.participating_client_list) and (self.client_time[idx]<=self.IterationTime) else
                                  np.random.uniform(0, 1.54 * self.args.step),
                                  range(self.args.client_num)
                                  ))
        # server aggregate




        self.global_weights =copy.deepcopy(self.aggregate_op.asyncAggBlend(self.participating_client_subnet,self.local_weights,self.global_weights,self.clientTransIdx,self.clientOG,self.blockOG,self.args,self.clientValid,self.globalValid,self.clientglobalValid,self.local_grads,self.participating_client_list,self.client_global_weight,self.globalValidFlow,self.client_receive_global_timeFrame[client_idx],self.newest_valid,self.newest_time))

        # update the newest global weight
        self.UpdateClientGlobalWeight()
        # print training metric
        for idx in self.participating_client_list:
            self.client_receive_global_timeFrame[idx]=global_epo



    def SelectParticipateClient(self):
        if self.args.isAggreByTime:
            # 按照指定周期进行聚合
            global_time = args.aggTime
            return np.flatnonzero(np.array(self.client_time) <= global_time)

        else:
            # 按照指定的接收数目进行选择
            sorted_client_time_idx = np.argsort(self.client_time)
            return np.array(sorted_client_time_idx[:self.args.receive_num])


    def evaluteModel(self,global_epo):
        '''
            data_size=int(frac * dataset_len['test'])
        '''
        self.globalModel.eval()
        subnet_list = [[1, 0, 0],[0, 1, 0],[0, 0, 1],[0, 1, 1],[1, 0, 1],[1, 1, 0],[1, 1, 1]]

        for subnet_flag_idx in subnet_list:
            correct=0
            totalSize=0
            batchNum=0
            with torch.no_grad():
                for _, data_batch in enumerate(self.data_loader_test):
                    batchNum += 1
                    color, depth, inertial, skeleton, labels = data_batch[0].cuda().float(), data_batch[1].cuda().float(), data_batch[2].cuda().float(), data_batch[3].cuda().float(), data_batch[4].cuda().long()
                    color = color.permute(0, 3, 1, 2)
                    depth = torch.unsqueeze(depth, 1)
                    inertial = inertial.permute(0, 2, 1)

                    output,modalityOut=self.globalModel(depth,inertial,skeleton, subnet_flag_idx,self.blockAtt,mode='test')
                    loss = self.cost(output[-1], labels)
                    _, pred = torch.max(output[-1], dim=1)
                    correct += torch.sum(pred == labels).item()
                    totalSize += len(pred)
                testAcc = correct / totalSize
            _subnet_str = []
            if subnet_flag_idx[0] == 1:
                _subnet_str.append('A')
            if subnet_flag_idx[1] == 1:
                _subnet_str.append('V')
            if subnet_flag_idx[2] == 1:
                _subnet_str.append('T')
            subnet_str = '+'.join(_subnet_str)
            # wethersubnet=self.args.iFlag if self.args.splitScheme.lower() in ['pure', '\'pure\''] else [1,1,1]
            # if subnet_flag_idx == wethersubnet:
            #     print("|----{} Test Metric:TestingAcc:{:4f}".format(subnet_str, testAcc))
            self.local_test_result[subnet_str].append(testAcc)
        if self.args.splitScheme.lower() in ['mix', '\'mix\'']:
            if self.args.subnet_num==3:
                print("|----Test Metric:TestingAcc:{:4f}".format(self.local_test_result['A+V+T'][-1]))
            elif self.args.subnet_num==2:
                print("|----Test Metric:TestingAcc:{:4f}".format((self.local_test_result['A+V'][-1]+self.local_test_result['A+T'][-1]+self.local_test_result['V+T'][-1])/3))
            else:
                print("|----Test Metric:TestingAcc:{:4f}".format((self.local_test_result['A'][-1]+self.local_test_result['T'][-1]+self.local_test_result['V'][-1])/3))
        else:
            _subnet_str = []
            if self.args.iFlag[0] == 1:
                _subnet_str.append('A')
            if self.args.iFlag[1] == 1:
                _subnet_str.append('V')
            if self.args.iFlag[2] == 1:
                _subnet_str.append('T')
            subnet_str = '+'.join(_subnet_str)
            print("|----{} Test Metric:TestingAcc:{:4f}".format(subnet_str, self.local_test_result[subnet_str][-1]))


    def LocalUpdateBlend(self,client_idx,pbar,clientOrderIdx,global_epo):

        self.globalModel.load_state_dict(self.client_global_weight[client_idx])
        for p in self.globalModel.depthSubnet.parameters():
            p.requires_grad = False if self.client_subnet_flag[client_idx][0]==0 else True

        for p in self.globalModel.inertialSubnet.parameters():
            p.requires_grad = False if self.client_subnet_flag[client_idx][1] == 0 else True

        for p in self.globalModel.skeletonSubnet.parameters():
            p.requires_grad = False if self.client_subnet_flag[client_idx][2] == 0 else True
        self.optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, self.globalModel.parameters()), lr=self.args.lr, weight_decay=self.args.weight_decay)

        aggLayerName = list(self.global_weights.keys())

        aggLayerName = list(filter(lambda x: x.find('depth') < 0, aggLayerName)) if self.client_subnet_flag[client_idx][0] == 0 else aggLayerName
        aggLayerName = list(filter(lambda x: x.find('inertial') < 0, aggLayerName)) if self.client_subnet_flag[client_idx][1] == 0 else aggLayerName
        aggLayerName = list(filter(lambda x: x.find('skeleton') < 0, aggLayerName)) if self.client_subnet_flag[client_idx][2] == 0 else aggLayerName


        # 本地模型的初始ogr
        subnetFlag = [[1,0,0],[0,1,0],[0,0,1]] if (self.args.splitScheme.lower() in ['\'mix\'', 'mix']) and (self.args.subnet_num==1) else [[1,1,0],[0,1,1],[1,0,1]] if (self.args.splitScheme.lower() in ['\'mix\'', 'mix']) and (self.args.subnet_num==2) else [[1,1,1]] if (self.args.splitScheme.lower() in ['\'mix\'', 'mix']) and (self.args.subnet_num==3) else [self.client_subnet_flag[0]]

        self.clientglobalValid,localModelOgrTermStartValid=self.eval_step(eval_data_loader=self.clientValidData[client_idx],subnetFlags=subnetFlag,model=self.client_global_weight[client_idx],client_subnet=self.client_subnet_flag[client_idx])
        self.globalValidFlow[global_epo]=self.clientglobalValid
        threshold=0.1
        LossOGR=[1/self.args.chunks for i in range(self.args.chunks)]
        LossGrad = [1/self.args.chunks for i in range(self.args.chunks)]
        modalityOGR=torch.zeros(3)
        for loc_epo in range(self.args.local_ep):
            self.blockLosses, self.blockWeightedLosses, self.blockCorrect = [0.0 for i in range(self.args.chunks+1)],[0.0 for i in range(self.args.chunks+1)],[0.0 for i in range(self.args.chunks+1)]
            self.modalityCorrect=torch.zeros(3)
            batchNum,correct,totalSize,trainLoss=0,0,0,0
            self.globalModel.train()
            for _,data_batch in enumerate(self.client_data[client_idx]):
                batchNum+=1
                color, depth, inertial, skeleton, labels = data_batch[0].cuda().float(), data_batch[1].cuda().float(), \
                                                           data_batch[2].cuda().float(), data_batch[3].cuda().float(), \
                                                           data_batch[4].cuda().long()
                color=color.permute(0,3,1,2)
                depth=torch.unsqueeze(depth,1)
                inertial = inertial.permute(0, 2, 1)
                self.optimizer.zero_grad()

                output, modalityOut = self.globalModel(depth, inertial, skeleton,self.client_subnet_flag[client_idx], self.blockAtt)
                loss = self.getLossBlend(output, labels, LossOGR, LossGrad, modalityOGR,modalityOut,self.client_subnet_flag[client_idx])



                loss.backward()
                self.optimizer.step()
                trainLoss+=loss.item()

                totalSize+=self.computBlockCorrect(output,labels,modalityOut,self.client_subnet_flag[client_idx])

                _,pred=torch.max(output[-1],dim=1)
                correct+=torch.sum(pred==labels).item()

            trainLoss=trainLoss/batchNum

            self.blockCorrect=np.array(self.blockCorrect)/totalSize

            self.modalityCorrect=self.modalityCorrect/totalSize


            trainAcc=self.blockCorrect[-1]

            pbar.set_postfix({'trainAcc': trainAcc})

            if loc_epo==0:
                # t globalModel
                localModelOgrTermStartTrain=np.array(self.blockCorrect)
                modalityStartTrain=np.array(self.modalityCorrect)
            else:
                # end localModel
                localModelOgrTermEndTrain=np.array(self.blockCorrect)
                modalityEndTrain=np.array(self.modalityCorrect)
                _,localModelOgrTermEndValid=self.eval_step(eval_data_loader=self.clientValidData[client_idx],subnetFlags=[self.client_subnet_flag[client_idx]],client_subnet=self.client_subnet_flag[client_idx])

                blockOGR=torch.zeros(self.args.chunks)
                blockGrad=torch.zeros(self.args.chunks)
                blockF1=torch.zeros(self.args.chunks)

                blockF11 = torch.zeros(self.args.chunks)
                blockOGR1=blending.get_optimal_gradient_blend_weights(
                    {
                        'eval':localModelOgrTermEndTrain[:-1],
                        'train':[localModelOgrTermEndTrain[-1] for i in range(self.args.chunks)]
                    },
                    {
                        'eval':localModelOgrTermStartTrain[:-1],
                        'train':[localModelOgrTermEndTrain[-1] for i in range(self.args.chunks)]
                    }
                )


                for blockId in range(self.args.chunks):
                    deltaO=abs(abs(localModelOgrTermEndTrain[-1]-localModelOgrTermEndTrain[blockId])-abs(localModelOgrTermStartTrain[-1]-localModelOgrTermStartTrain[blockId]))
                    deltaG=abs(localModelOgrTermEndTrain[-1]-localModelOgrTermStartTrain[-1])
                    blockOGR[blockId]=(deltaO+1e-10)/(deltaG+1e-10)

                    tmp=0
                    for dimensionIdx in range(3):
                        layerName = 'merge_linears{}.{}.weight'.format(dimensionIdx, blockId)
                        local_module_wt = self.globalModel.state_dict()[layerName]
                        global_module_wt = self.client_global_weight[client_idx][layerName]
                        tmp=tmp+torch.norm((local_module_wt-global_module_wt).float(), 2)
                    blockGrad[blockId]=tmp

                    # 要求OGR越小越好,Grad越大越好
                    blockF1[blockId]=(2*blockGrad[blockId]/blockOGR[blockId])/(blockGrad[blockId]+1/blockOGR[blockId])

                    blockF11[blockId]=((2*blockGrad[blockId]*blockOGR1[blockId])/(blockGrad[blockId]+blockOGR1[blockId]))
                self.blockAtt=self.args.chunks*blockF1/(torch.sum(blockF1))



                modalityOGR=torch.zeros(3)
                modalityF1 = torch.zeros(3)
                for modalityId in range(len(self.modalityCorrect)):
                    if self.client_subnet_flag[client_idx][modalityId]==1:
                        deltaO=abs(abs(modalityEndTrain[modalityId]-localModelOgrTermEndTrain[-1])-abs(modalityStartTrain[modalityId]-localModelOgrTermStartTrain[-1]))
                        # deltaO = abs(modalityEndTrain[modalityId] - localModelOgrTermEndTrain[-1]
                        deltaG=abs(localModelOgrTermEndTrain[-1]-localModelOgrTermStartTrain[-1])
                        modalityOGR[modalityId]=(deltaO + 1e-10) / (deltaG + 1e-10)
                        modalityF1[modalityId]=(deltaG + 1e-10)/(deltaO + 1e-10)
                modalityOGRsum=torch.sum(modalityOGR)
                modalityOGR=modalityOGR/modalityOGRsum

                if loc_epo==int(self.args.local_ep-1):
                    self.clientValid, localModelOgrTermEndValid = self.eval_step(eval_data_loader=self.clientValidData[client_idx],subnetFlags=subnetFlag,client_subnet=self.client_subnet_flag[client_idx])
                    self.local_train_loss_tmp += trainLoss
                    self.local_train_correct_tmp += trainAcc
                    self.local_weights.append(copy.deepcopy(self.globalModel.state_dict()))
                    self.local_grads.append({key: copy.deepcopy(self.globalModel.state_dict())[key] -
                                                  self.client_global_weight[client_idx][key] for key in
                                             self.global_weights.keys()})

                    subnetFlag = [[1,0,0],[0,1,0],[0,0,1]] if (self.args.splitScheme.lower() in ['\'mix\'', 'mix']) and (self.args.subnet_num==1) else [[1,1,0],[0,1,1],[1,0,1]] if (self.args.splitScheme.lower() in ['\'mix\'', 'mix']) and (self.args.subnet_num==2) else [[1,1,1]] if (self.args.splitScheme.lower() in ['\'mix\'', 'mix']) and (self.args.subnet_num==3) else [self.client_subnet_flag[0]]
                    # t+T globalModel
                    self.globalModel.load_state_dict(self.global_weights)
                    self.globalValid,globalModelOgrTermStartValid=self.eval_step(eval_data_loader=self.clientValidData[client_idx],model=self.global_weights,subnetFlags=subnetFlag,client_subnet=self.client_subnet_flag[client_idx])
                    self.globalValid=self.newest_valid

                    _,globalModelOgrTermStartTrain=self.eval_step(eval_data_loader=self.client_data[client_idx],model=self.global_weights,subnetFlags=[self.client_subnet_flag[client_idx]],client_subnet=self.client_subnet_flag[client_idx])

        if self.client_receive_global_timeFrame[client_idx] >= self.newest_time:
            self.newest_valid = self.clientglobalValid
        self.newest_time=max(self.newest_time,self.client_receive_global_timeFrame[client_idx])

        modalityGrad=torch.zeros(3)

        for i in np.flatnonzero(np.array(self.client_subnet_flag[client_idx])==1):
            subnetName=['depth','inertial','skeleton'][i]
            modalityLayer=list(filter(lambda x:x.find(subnetName)>=0,aggLayerName))
            if len(modalityLayer)>0:
                tmp=0
                for key in modalityLayer:
                    local_module_wt = self.globalModel.state_dict()[key]
                    global_module_wt = self.client_global_weight[client_idx][key]
                    tmp = tmp + torch.norm((local_module_wt - global_module_wt).float(), 2)
                modalityGrad[i]=tmp


        modalityF1=modalityF1/torch.sum(modalityF1)


        for key in self.global_weights.keys():
            if 'Subnet' in key:

                if ((self.client_subnet_flag[client_idx][0] == 1) and ('depthSubnet' in key) and (modalityF1[0]>=threshold)):
                    self.clientTransIdx[key] += [clientOrderIdx]
                elif ((self.client_subnet_flag[client_idx][1] == 1) and ('inertialSubnet' in key) and (modalityF1[1]>=threshold)):
                    self.clientTransIdx[key] += [clientOrderIdx]
                elif ((self.client_subnet_flag[client_idx][2] == 1) and ('skeletonSubnet' in key) and (modalityF1[2]>=threshold)):
                    self.clientTransIdx[key] += [clientOrderIdx]

            else:

                if 'merge_linears' in key:
                    subnetIdx=int(key.split('.')[0][-1])
                    block_idx=int(key.split('.')[1])
                    if self.blockAtt[block_idx]>=1:
                        self.clientTransIdx[key]+=[clientOrderIdx]
                else:
                    self.clientTransIdx[key]+=[clientOrderIdx]


        transParSize =sum(self.modelLayerSize[key]*len(self.clientTransIdx[key]) for key in self.global_weights.keys())
        return transParSize

    def computBlockCorrect(self,output,labels,modalityOut=None,subnet=[1,1,1]):
        for i in range(len(output)):
            _, pred = torch.max(output[i], dim=1)
            self.blockCorrect[i] += torch.sum(pred == labels).item()
        for i in range(len(subnet)):
            if subnet[i]==1:
                _, pred = torch.max(modalityOut[i], dim=1)
                self.modalityCorrect[i] += torch.sum(pred == labels).item()
        return len(pred)




    def getLossBlend(self,FusionChunksOut,labels,LossOGR,LossGrad,modalityOGR,modalityOut=None,subnet=[1,1,1]):

        LossOGR=torch.tensor(np.array(LossOGR))
        LossGrad=torch.tensor(np.array(LossGrad))
        losses=0
        blockLoss=[0 for i in range(len(FusionChunksOut))]
        blockLoss[-1]=self.cost(FusionChunksOut[-1],labels)

        if np.sum(np.array(modalityOGR)) != 0:
            for modalityId in range(3):
                if subnet[modalityId]==1:
                    loss = self.cost(modalityOut[modalityId], labels)
                    losses = losses + loss * modalityOGR[modalityId] * self.args.alpha

        return blockLoss[-1]+losses

    def eval_step(self, eval_data_loader,model=None,subnetFlags=[[1,1,1]],type='acc',client_subnet=[1,1,1]):
        """Perfome the evaluation"""
        if model is not None:

            self.globalModel.load_state_dict(model)
        self.globalModel.eval()
        avgValidAcc=0
        for subnetFlag in subnetFlags:
            correct = [0.0 for i in range(int(self.args.chunks+1))]
            totalSize = 0
            for _, data_batch in enumerate(eval_data_loader):
                color, depth, inertial, skeleton, labels = data_batch[0].cuda().float(), data_batch[1].cuda().float(), \
                                                           data_batch[2].cuda().float(), data_batch[3].cuda().float(), \
                                                           data_batch[4].cuda().long()
                color = color.permute(0, 3, 1, 2)
                depth = torch.unsqueeze(depth, 1)
                inertial = inertial.permute(0, 2, 1)
                # output = self.globalModel(depth, inertial, skeleton, subnetFlag)

                output,modalityOut = self.globalModel(depth, inertial, skeleton, subnetFlag,self.blockAtt,mode='test')

                for block_idx in range(len(output)):
                    _, pred = torch.max(output[block_idx], 1)
                    block_correct = torch.sum(pred == labels).item()
                    correct[block_idx]+=block_correct
                totalSize += len(pred)
            validAcc=np.array(correct)/totalSize
            avgValidAcc+=validAcc
            if subnetFlag==client_subnet:
                clientModalityAcc=validAcc

        return avgValidAcc/len(subnetFlags),clientModalityAcc

    def UpdateClientGlobalWeight(self):
        self.client_global_weight=list(map(lambda idx: self.global_weights if idx in self.participating_client_list else self.client_global_weight[idx], range(self.args.client_num)))




class ServerUpdate():
    def __init__(self):
        pass




    def asyncAggBlend(self,participating_client_subnet,local_weights,server_global_weight,clientTransIdx,clientOG,blockOG,args,clientValid,globalValid,clientglobalValid,local_grads,participating_client_list,client_global_weight,globalValidFlow,client_receive_global_timeFrame,newest_valid,newest_time):
        '''

        :param participating_client_subnet:
        :param local_grads:
        :param server_global_weight:
        :param clientTransIdx: 传输该层的用户
        :param clientOG: [本地模型权重，历史模型权重]
        :param blockOG: [每个block的权重]
        :param isBlockAggWeight: 平均/ogr权重
        :return:
        '''

        lamda = 1
        e = 1

        # 欧式距离
        deltaAccGlobal = abs(globalValidFlow[int(newest_time)][-1]-globalValidFlow[int(client_receive_global_timeFrame)][-1])/(newest_time-client_receive_global_timeFrame)/5 if newest_time-client_receive_global_timeFrame!=0 else 0
        deltaAccLocal = abs(clientValid[-1]-globalValidFlow[int(client_receive_global_timeFrame)][-1])/5
        delta_x = 0
        delta_grad = 0

        for key in local_weights[0].keys():
            if (('depth' in key) and (participating_client_subnet[0][0] == 0)) or (
                    ('inertial' in key) and (participating_client_subnet[0][1] == 0)) or (
                    ('skeleton' in key) and (participating_client_subnet[0][2] == 0)):
                continue
            else:
                tmp = 0
                div_len = len(clientTransIdx[key])
                if div_len != 0:

                    for orderIdx in range(div_len):
                        client_idx = participating_client_list[orderIdx]
                        delta_x += torch.norm(
                            (server_global_weight[key] - client_global_weight[client_idx][key]).float(), 2)
                        delta_grad += torch.norm(local_grads[clientTransIdx[key][orderIdx]][key].float(), 2)


        deltaG=abs(deltaAccLocal*delta_grad)
        deltaO=abs(abs(deltaAccGlobal*delta_x)-deltaG)

        elta=math.pow(0.9,abs(deltaO/(deltaG+1e-10)))


        for key in local_weights[0].keys():
            if (('depth' in key) and (participating_client_subnet[0][0] == 0)) or (
                    ('inertial' in key) and (participating_client_subnet[0][1] == 0)) or (
                    ('skeleton' in key) and (participating_client_subnet[0][2] == 0)):
                continue
            else:
                tmp = 0
                div_len = len(clientTransIdx[key])
                if div_len != 0:
                    for orderIdx in range(div_len):
                        client_idx = clientTransIdx[key][orderIdx]

                        tmp=tmp+(1-elta)*server_global_weight[key]+elta*local_weights[client_idx][key]
                    server_global_weight[key] = tmp / div_len


        return server_global_weight

