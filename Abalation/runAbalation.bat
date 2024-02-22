% device number%
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=1000 --lr=0.0001 --thresholdAcc=0.8 --isStopByAcc=True
python abalation_UTD.py --iid=0 --client_num=10 --splitScheme='mix' --subnet_num=2 --epochs=1000 --lr=0.0001 --thresholdAcc=0.8 --isStopByAcc=True
python abalation_UTD.py --iid=0 --client_num=20 --splitScheme='mix' --subnet_num=2 --epochs=1000 --lr=0.0001 --thresholdAcc=0.8 --isStopByAcc=True

python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=3 --epochs=1000 --lr=0.0001 --thresholdAcc=0.8 --isStopByAcc=True
python abalation_UTD.py --iid=0 --client_num=10 --splitScheme='mix' --subnet_num=3 --epochs=1000 --lr=0.0001 --thresholdAcc=0.8 --isStopByAcc=True
python abalation_UTD.py --iid=0 --client_num=20 --splitScheme='mix' --subnet_num=3 --epochs=1000 --lr=0.0001 --thresholdAcc=0.8 --isStopByAcc=True



% Dirichlet %
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=150 --lr=0.0001 --Dirichlet=True --Naplha=0.01
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=150 --lr=0.0001 --Dirichlet=True --Naplha=0.05
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=150 --lr=0.0001 --Dirichlet=True --Naplha=0.1
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=150 --lr=0.0001 --Dirichlet=True --Naplha=0.5
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=150 --lr=0.0001 --Dirichlet=True --Naplha=1

python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=3 --epochs=150 --lr=0.0001 --Dirichlet=True --Naplha=0.01
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=3 --epochs=150 --lr=0.0001 --Dirichlet=True --Naplha=0.05
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=3 --epochs=150 --lr=0.0001 --Dirichlet=True --Naplha=0.1
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=3 --epochs=150 --lr=0.0001 --Dirichlet=True --Naplha=0.5
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=3 --epochs=150 --lr=0.0001 --Dirichlet=True --Naplha=1




%sensitivity%
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=150 --lr=0.0001 --hidDim=200
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=150 --lr=0.0001 --hidDim=300
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=150 --lr=0.0001 --hidDim=400
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=150 --lr=0.0001 --hidDim=500

python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=150 --lr=0.0001 --chunks=2
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=150 --lr=0.0001 --chunks=6
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=150 --lr=0.0001 --chunks=8
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=150 --lr=0.0001 --chunks=10




