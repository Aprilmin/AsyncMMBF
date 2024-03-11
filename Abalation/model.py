from __future__ import absolute_import
from __future__ import division
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import copy
import numpy as np
##########
# Basic layers
##########
import torch
import torch.nn as nn
from scipy.interpolate import interp1d,interp2d
class skeSubnet(nn.Module):
    """
    CNN layers applied on acc sensor data to generate pre-softmax
    ---
    params for __init__():
        input_size: e.g. 1
        num_classes: e.g. 6
    forward():
        Input: data
        Output: pre-softmax
    """
    def __init__(self, input_size,output_size):
        super().__init__()

        # Extract features, 3D conv layers
        self.features = nn.Sequential(
            nn.Conv3d(input_size, 64, [5,5,2]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(64, 64, [5,5,2]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(64, 32, [5,5,1]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv3d(32, 16, [5,2,1]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            )
        self.fc=nn.Sequential(
            nn.Linear(in_features=16*4*27,out_features=output_size),
            # nn.ReLU(),
            # nn.Linear(in_features=512,out_features=256),
            # nn.ReLU(),
            # nn.Dropout(p=0.6),
            # nn.Linear(in_features=256,out_features=output_size),
            # nn.LogSoftmax(dim=1)
        )


    def forward(self, x):
        x=x.permute(0,1,3,2)
        x=torch.unsqueeze(x,1)
        x = self.features(x)
        # print(x.shape)
        x=x.view(x.shape[0],-1)

        x=self.fc(x)

        return x
class inertialSubNet(nn.Module):  # 文本的基于LSTM的子网

    def __init__(self, in_size, hidden_size,  num_layers=1, dropout=0.0, bidirectional=False):

        super(inertialSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout,
                           bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):

        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        # y_1 = self.linear_1(h)
        return h
class depthSubnet(nn.Module):
    def __init__(self,out_size,input_size=1,dropout=0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_size,out_channels=64,kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.AvgPool2d(3),

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(),


            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            )
        self.fc=nn.Sequential(
            nn.Linear(in_features=16*5*10,out_features=out_size),
            # nn.ReLU(),
            # nn.Linear(in_features=512,out_features=256),
            # nn.ReLU(),
            # nn.Dropout(p=0.6),
            # nn.Linear(in_features=256,out_features=out_size),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        return x


def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks
    sizes_list = [split_size] * chunks
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim) # Adjust last
    assert sum(sizes_list) == dim
    if sizes_list[-1]<0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j-1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list,split_size
def interplate_2d(inputMatrix, xNum, yNum):
    inputMatrix=inputMatrix.cpu().detach()
    x = np.arange(inputMatrix.shape[0])
    y = np.arange(inputMatrix.shape[1])
    # print('{};{};{}'.format(inputMatrix.shape,xNum,yNum))
    # linear_interp = interp2d(y, x, inputMatrix, kind='cubic')
    try:
        linear_interp = interp2d(y, x, inputMatrix, kind='cubic')
    except:
        linear_interp = interp2d(y, x, inputMatrix, kind='linear')
    endpointX = inputMatrix.shape[0]
    endpointY = inputMatrix.shape[1]
    x_new = np.arange(0, endpointX, endpointX / xNum)
    y_new = np.arange(0, endpointY, endpointY / yNum)
    outMatrix = torch.tensor(linear_interp(y_new, x_new)).cuda().float()
    # plt.plot(x, inputMatrix[:, 0], 'ro-', x_new, outMatrix[:,0], 'b-')
    return outMatrix
def get_chunks(x,sizes):
    out = []
    begin = 0
    if len(x.shape)==1:
        x=torch.unsqueeze(x,0)
    for s in sizes:
        y = x.narrow(1,begin,s)
        out.append(y)
        begin = begin+s
    return out
class BlockTrainerBlend(nn.Module):
    def __init__(self,args,dev,
                 mm_dim=1600,
                 chunks=20,
                 rank=5,
                dropout_input=0,
                dropout_output=0,
                pos_norm='before_cat',

                ):
        super(BlockTrainerBlend, self).__init__()

        self.args=args
        self.dev=dev
        self.output_dim = 27





        self.inertialSubnet = inertialSubNet(120,mm_dim,dropout=0.5) # inertial lr=0.01 ac=0.625
        self.skeletonSubnet = skeSubnet(input_size=1, output_size=mm_dim) # lr=0.01 ac=0.7
        self.depthSubnet=depthSubnet(mm_dim) # depth 0.001 ac=0.5




        self.x_0 = torch.ones(1, int(mm_dim), dtype=torch.float).cuda()
        self.x_1 = torch.ones(1, int(mm_dim), dtype=torch.float).cuda()
        self.x_2 = torch.ones(1, int(mm_dim), dtype=torch.float).cuda()



        self.mm_dim = mm_dim
        self.chunks = chunks
        self.rank = rank
        self.dropout_input = dropout_input

        self.dropout_output = dropout_output
        assert(pos_norm in ['before_cat', 'after_cat'])
        self.pos_norm = pos_norm
        # Modules

        merge_linears0, merge_linears1,merge_linears2 = [], [],[]
        self.sizes_list,self.split_size = get_sizes_list(mm_dim, chunks)
        for size in self.sizes_list:
            ml0 = nn.Linear(size, size*rank)
            merge_linears0.append(ml0)
            ml1 = nn.Linear(size, size*rank)
            merge_linears1.append(ml1)
            ml2=nn.Linear(size,size*rank)
            merge_linears2.append(ml2)
        # self.chunksVar=nn.Parameter(torch.Tensor(1,self.split_size,self.output_dim),requires_grad=False)
        # nn.init.xavier_normal_(self.chunksVar)

        self.merge_linears0 = nn.ModuleList(merge_linears0)
        self.merge_linears1 = nn.ModuleList(merge_linears1)
        self.merge_linears2=nn.ModuleList(merge_linears2)
        self.linear_out = nn.Linear(mm_dim, self.output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    # def assing_weights(self, net_name, weights):
    #     """ Assign the weights to the specific network """
    #     self.name_net_mapper[net_name].load_state_dict(weights, strict=True)
    def forward(self,  depth,inertial,skeleton, subnet_flag,blockAtt,mode='train'):
        # referClassifier=copy.deepcopy(self.linear_out).to(self.dev)
        # for param in referClassifier.parameters():
        #     param.requires_grad = False

        bsize = depth.shape[0]

        x0 = self.depthSubnet(depth) if subnet_flag[0]==1 else self.x_0
        x1 = self.inertialSubnet(inertial) if subnet_flag[1]==1 else self.x_1
        x2 = self.skeletonSubnet(skeleton) if subnet_flag[2]==1 else self.x_2



        x0_chunks = get_chunks(x0, self.sizes_list)
        x1_chunks = get_chunks(x1, self.sizes_list)
        x2_chunks = get_chunks(x2, self.sizes_list)
        referChunks=[]
        if (mode=='train') and (np.sum(np.array(subnet_flag))!=1):
            referChunks.append(get_chunks(self.x_0, self.sizes_list))
            referChunks.append(get_chunks(self.x_1, self.sizes_list))
            referChunks.append(get_chunks(self.x_2, self.sizes_list))



        zs = []
        chunks_out=[]
        modalityOut=[[],[],[]]

        chunksSet=[x0_chunks,x1_chunks,x2_chunks]
        startIdx=0

        for chunk_id, m0, m1, m2 in zip(range(len(self.sizes_list)),
                                    self.merge_linears0,
                                    self.merge_linears1,
                                    self.merge_linears2):

            merge_linears=[m0,m1,m2]
            m=1


            m=m0(x0_chunks[chunk_id])*m1(x1_chunks[chunk_id])*m2(x2_chunks[chunk_id])
            m = m.view(bsize, self.rank, -1)
            z = torch.sum(m, 1)
            if (self.args.isBlockAtt==True) and (self.args.isBlend==1):
                z=blockAtt[chunk_id]*z
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
                z = F.normalize(z,p=2)
            zs.append(z)

            if (mode=='train') and (np.sum(np.array(subnet_flag))!=1):

                if subnet_flag[0]==1:
                    modality0=m0(x0_chunks[chunk_id])*m1(referChunks[1][chunk_id])*m2(referChunks[2][chunk_id])
                    modality0 = modality0.view(bsize, self.rank, -1)
                    modality0 = torch.sum(modality0, 1)
                    if self.pos_norm == 'before_cat':
                        modality0 = torch.sqrt(F.relu(modality0)) - torch.sqrt(F.relu(-modality0))
                        modality0 = F.normalize(modality0, p=2)
                    modalityOut[0].append(modality0)

                if subnet_flag[1]==1:
                    modality1=m0(referChunks[0][chunk_id])*m1(x1_chunks[chunk_id])*m2(referChunks[2][chunk_id])
                    modality1 = modality1.view(bsize, self.rank, -1)
                    modality1 = torch.sum(modality1, 1)
                    if self.pos_norm == 'before_cat':
                        modality1 = torch.sqrt(F.relu(modality1)) - torch.sqrt(F.relu(-modality1))
                        modality1 = F.normalize(modality1, p=2)
                    modalityOut[1].append(modality1)

                if subnet_flag[2]==1:
                    modality2=m0(referChunks[0][chunk_id])*m1(referChunks[1][chunk_id])*m2(x2_chunks[chunk_id])
                    modality2=modality2.view(bsize,self.rank,-1)
                    modality2 = torch.sum(modality2, 1)
                    if self.pos_norm == 'before_cat':
                        modality2 = torch.sqrt(F.relu(modality2)) - torch.sqrt(F.relu(-modality2))
                        modality2 = F.normalize(modality2, p=2)
                    modalityOut[2].append(modality2)



            inter_z=torch.zeros(z.shape[0],self.mm_dim).cuda()
            # print(z)
            # print(inter_z)
            inter_z[:,startIdx:startIdx+self.sizes_list[chunk_id]]=z
            startIdx+=self.sizes_list[chunk_id]


            # inter_z=interplate_2d(z,z.shape[0],self.split_size)
            # inter_z=torch.unsqueeze(inter_z,1)
            # chunk_out=F.dropout(torch.matmul(inter_z,self.chunksVar),p=self.dropout_output)
            # chunk_out=torch.squeeze(chunk_out)
            # chunks_out.append(F.softmax(chunk_out,dim=1))
            chunks_out.append(F.softmax(self.linear_out(inter_z),dim=1))

        z = torch.cat(zs,1)
        if (mode=='train') and (np.sum(np.array(subnet_flag))!=1):
            if subnet_flag[0] == 1:
                modalityOut[0] = torch.cat(modalityOut[0], 1)
                if self.pos_norm == 'after_cat':
                    modalityOut[0] = torch.sqrt(F.relu(modalityOut[0])) - torch.sqrt(F.relu(-modalityOut[0]))
                    modalityOut[0] = F.normalize(modalityOut[0], p=2)
                modalityOut[0] = F.dropout(self.linear_out(modalityOut[0]), p=self.dropout_output,training=self.training)
                modalityOut[0] = F.softmax(modalityOut[0], dim=1)
            else:
                modalityOut[0]=0
            if subnet_flag[1] == 1:
                modalityOut[1] = torch.cat(modalityOut[1], 1)
                if self.pos_norm == 'after_cat':
                    modalityOut[1] = torch.sqrt(F.relu(modalityOut[1])) - torch.sqrt(F.relu(-modalityOut[1]))
                    modalityOut[1] = F.normalize(modalityOut[1], p=2)
                modalityOut[1] = F.dropout(self.linear_out(modalityOut[1]), p=self.dropout_output,
                                           training=self.training)
                modalityOut[1] = F.softmax(modalityOut[1], dim=1)
            else:
                modalityOut[1]=0

            if subnet_flag[2] == 1:
                modalityOut[2] = torch.cat(modalityOut[2], 1)
                if self.pos_norm == 'after_cat':
                    modalityOut[2] = torch.sqrt(F.relu(modalityOut[2])) - torch.sqrt(F.relu(-modalityOut[2]))
                    modalityOut[2] = F.normalize(modalityOut[2], p=2)
                modalityOut[2] = F.dropout(self.linear_out(modalityOut[2]), p=self.dropout_output, training=self.training)
                modalityOut[2] = F.softmax(modalityOut[2], dim=1)
            else:
                modalityOut[2]=0

        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z,p=2)




        z = F.dropout(self.linear_out(z),p=self.dropout_output,training=self.training)


        z = F.softmax(z, dim=1)

        chunks_out.append(z)



        return chunks_out,modalityOut
