#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
from HA4MData import LoadDataByIID,HA4Mdataset
from options import args_parser
from GlobalScheduling import GlobalSchedule
import os
import numpy as np
import torch
import warnings
import pandas as pd
import random

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
    mode='AsyncMMBF'
    dataMode='M{}'.format(args.subnet_num) if args.splitScheme.lower() in ['\'mix\'','mix'] else 'P{}'.format(args.iFlag)


    if args.isGPU and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        dev='cuda'
    else:
        dev='cpu'
    print('getting data')
    utdOp=LoadDataByIID(args)


    testLoader = utdOp.testLoader
    trainData=utdOp.TrainData
    validData=utdOp.ValidData
    clientValidData=utdOp.clientValidData

    client_data_frac = utdOp.client_data_frac

    clientData=utdOp.clientData
    dataset_len=utdOp.dataLen


    server=GlobalSchedule(args,testLoader,dataset_len,clientData,clientValidData,validData,trainData,client_data_frac)
    model = server.globalModel
    modelSize = sum(p.numel() for p in model.parameters()) / 1e6
    print('############################ Start training : ModelSize: {}M ############################'.format(modelSize))
    dirname="../results/HA4M/IID{}".format(args.iid)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    print('model size: {:.5f}M'.format(modelSize))


    filename = '{}-{}-lr{}-alpha{}-{:.2f}M-{}c-R{}-t{}-ep{}-p{}'.format(mode, dataMode, args.lr,
                                                                args.alpha, modelSize, args.chunks,
                                                                args.hidDim,args.thred,args.local_ep,args.p)
    # filename = '{}-Acc{}-{}'.format(filename, args.thresholdAcc,timestr) if args.isStopByAcc == True else '{}-{}'.format(filename,timestr)
    dirname = os.path.join(dirname, filename)

    if not os.path.exists(dirname):
        os.mkdir(dirname)


    for global_epo in range(args.epochs):
        server.GlobalIterate(global_epo)


    modelSize = sum(p.numel() for p in model.parameters()) / 1e6


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











