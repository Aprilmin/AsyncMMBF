#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
from UtdData import LoadDataByIID,UTDdataset
from options import args_parser
from GlobalScheduling import GlobalSchedule

import time
import os
import numpy as np
import torch
import warnings
import torch.nn as nn
import pandas as pd
import random

# 1、loss=f(x)+|ogr|/||grad||_2


# 分布式训练
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def exp_details(args):
    mode = 'semi_async' if args.isAsync == 0.5 else 'async' if args.isAsync == 1 else 'sync' if args.isAsync == 0 else 'our_scheme'
    print('Experimental details:')
    print(f'    Learning  : {args.lr}')
    print(f'    Dataset   : {args.datasetName}')
    print('Federated parameters:')
    if args.iid==0:
        print(f'    IID   : {args.iid}  --unequal   : {args.unequal}')
    else:
        print(f'    IID   : {args.iid}')
    print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Train Mode       : {mode}')
    print(f'    Client Number : {args.client_num}')
    print(f'    Participated Client Number : {args.client_num}')
    return
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    setup_seed(3407)
    args,argsDict=args_parser()
    exp_details(args)
    mode = 'AsyncMMBF'
    dataMode='M{}'.format(args.subnet_num) if args.splitScheme.lower() in ['\'mix\'','mix'] else 'P{}'.format(args.iFlag)


    if args.isGPU and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        dev='cuda'
    else:
        dev='cpu'
    print('getting data')
    utdOp=LoadDataByIID(args)


    testLoader = utdOp.testLoader
    validLoader = utdOp.validLoader
    clientValidData=utdOp.clientValidData
    client_data_frac = utdOp.client_data_frac
    clientData=utdOp.clientData
    dataset_len=utdOp.dataLen

    server=GlobalSchedule(args,testLoader,dataset_len,clientData,clientValidData,validLoader,client_data_frac)
    model = server.globalModel
    modelSize = sum(p.numel() for p in model.parameters()) / 1e6
    # 创建结果文件夹
    dirname="../results/UTD/Client{}_Naplha{}".format(args.client_num,args.Naplha) if args.Dirichlet else "../results/UTD/Client{}_IID{}".format(args.client_num,args.iid)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    print('model size: {:.5f}M'.format(modelSize))

    filename = '{}-{}-lr{}-alpha{}-{:.2f}M-{}c-R{}'.format(mode, dataMode, args.lr,args.alpha, modelSize, args.chunks,args.hidDim)

    dirname = os.path.join(dirname, filename)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    testAccGMIS = np.zeros(5)
    for global_epo in range(args.epochs):
        server.GlobalIterate(global_epo)
        if args.isStopByAcc:
            testAccGMIS[global_epo%5]=(server.local_test_result['A+V'][-1]+server.local_test_result['A+T'][-1]+server.local_test_result['V+T'][-1])/3 if args.subnet_num==2 else server.local_test_result['A+V+T'][-1] if args.subnet_num==3 else NotImplementedError
            tmp=np.flatnonzero(testAccGMIS!=0)
            if np.mean(testAccGMIS[tmp])>args.thresholdAcc:
                break
      

    torch.save({'model': model.state_dict()}, os.path.join(dirname, 'model.pth'))
    outpath = os.path.join(dirname, '{}.xlsx'.format(filename))
    writer = pd.ExcelWriter(outpath, engine='openpyxl')


    server.local_test_result['TrainAcc']=server.local_train_correct_list
    server.local_test_result['transPar']=server.transPar_list
    server.local_test_result= pd.DataFrame.from_dict(server.local_test_result,orient='index').T


    server.local_test_result.to_excel(writer, sheet_name='result_log')


    # writer.save()
    workbook = writer.book
    writer.close()
    print('model size: {:.5f}M'.format(modelSize))












