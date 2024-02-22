import argparse
import distutils.util
def args_parser():
    parser = argparse.ArgumentParser()
    # wandb arguments
    # parser.add_argument('--wandb_enable',type=bool,default=True,help='whether to use wandb')
    # parser.add_argument('--project',type=str,default='Create-new',help='project name')
    # parser.add_argument('--name',type=str,default='1',help='name')

    # global arguments

    parser.add_argument('--isGPU',type=bool,default=True)
    parser.add_argument('--isBlend',type=int,default=1,help='1:True;0:False')
    parser.add_argument('--isAsync',type=float,default=1,help='async 1, sync 0,semi 0.5 my:2')
    parser.add_argument('--max_delay',type=int,default=5,help='max delay of clients')
    parser.add_argument('--datasetName', type=str, default='sims',help='support mosi/mosei/sims')
    parser.add_argument('--client_num',type=int,default=5,help='number of users')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay',type=float,default=0.004)
    parser.add_argument('--unequal',type=bool,default=False,help='when sample non-I.I.D client data from dataset s.t clients have unequal amount of data')
    parser.add_argument('--isStopByAcc',type=bool,default=False,help='the model finish train when the test accuracy achieve the threshold')
    parser.add_argument('--thresholdAcc',type=float,default=0.8,help='test accuracy threshold')
    parser.add_argument('--isBlockAtt', type=lambda x: bool(distutils.util.strtobool(x)), default='True')
    parser.add_argument('--train_mode', type=str, default="classification", help='regression / classification')
    parser.add_argument('--step',type=int,default=20)
    parser.add_argument('--isCentrial',type=lambda x:bool(distutils.util.strtobool(x)),default='False')
    parser.add_argument('--Dirichlet', type=lambda x: bool(distutils.util.strtobool(x)), default='False')
    parser.add_argument('--Naplha', type=float, default=1)



    # async arguments
    parser.add_argument('--isAggreByTime', type=bool, default=False,help='Aggregate the global model by the given time or receives the given number M of local models')
    parser.add_argument('--aggTime', type=int, default=10, help='Aggregate the global model by the given time')
    parser.add_argument('--receive_num', type=int, default=3,help='server receives receive_num local model to aggregate')
    parser.add_argument('--maxStaleness',type=int,default=5,help='synchronize stale local model while the client staleness outer the maxStaleness')

    # server arguments
    parser.add_argument('--epochs', type=int, default=50,help="number of rounds of training")
    parser.add_argument('--model', type=str, default='CNN', help='model name')
    parser.add_argument('--schedule',type=str, default='RandomSchedule',help='node selection method')

    # client arguments
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")

    # fusion arguments

    parser.add_argument('--hidDim',type=int,default=100)
    parser.add_argument('--chunks',type=int,default=4)
    parser.add_argument('--dropIn',type=float,default=0.,)
    parser.add_argument('--dropOut',type=float,default=0.)
    parser.add_argument('--rank',type=int,default=4)
    parser.add_argument('--div', type=int, default=1)


    parser.add_argument('--fusionType',type=str,default='block',help='concate or tucker')
    # data split scheme
    parser.add_argument('--subnet_num',type=int,default=3,help='n : n model at each client else subnet_num=4 presents every client have one modal')
    parser.add_argument('--classNum',type=int,default=27,help='two class or five class or three class')
    parser.add_argument('--iid', type=float, default=0,help='iid:1 、 non-iid：0、 non-iid-0.9: 10% of data is iid and the other 90% is non-iid')
    parser.add_argument('--splitScheme',type=str,default='mix',help='mix:global dataset = [a,v,t] or pure: global dataset = a/v/t')
    # parser.add_argument('--iFlag',type=list,default=[1,1,1],help='is the subnet flag if the splitScheme is pure')
    parser.add_argument('--iFlag',nargs='+',type=int,default=[1,1,1])
    parser.add_argument('--alpha',type=float,default=1)
    parser.add_argument('--ratioOGR',type=float,default=1)
    parser.add_argument('--ratioW', type=float, default=0.7)
    parser.add_argument('--mu',type=float,default=0.2)




    args = parser.parse_args()
    # args = parser.parse_args(args=[])
    argsDict = args.__dict__


    return args,argsDict
