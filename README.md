# AsyncMMBF
The implementation of "Multimodal Fusion with Block Term Decomposition for Asynchronous Federated Learning".  All simulations are conducted using the PyTorch framework on a computer equipped with an Intel Core i7-12700 CPU, 32GB of memory, and an NVIDIA GeForce RTX 3090 GPU.

# Project Strcuture
```
|--UTD // codes running on UTD dataset

	|-- data                    // download corresponding dataset and save here
	|-- requirements.txt    // the required environment on UTD dataset
	|-- run.bat    // running script of AsyncMMBF in 4 cases on UTD dataset
	|-- mian_UTD.py    // main file of AsyncMMBF on UTD dataset
	|-- GlobalScheduling.py	// set up training
	|-- UtdData.py		// load the data for clients in FL
	|-- model.py 	// model configuration for UTD datasets 
	|-- blending.py		// utility functions
	|-- options.py		// configuration for AsyncMMBF


|--HA4M // codes running on HA4M dataset

	|-- data                    // download corresponding dataset and save here
	|-- requirements.txt    // the required environment on HA4M dataset
	|-- run.bat    // running script of AsyncMMBF in 4 cases on HA4M dataset
	|-- mian_HA4M.py    // main file of AsyncMMBF on HA4M dataset
	|-- GlobalScheduling.py	// set up training
	|-- HA4MData.py		// load the data for clients in FL
	|-- model.py 	// model configuration for HA4M datasets 
	|-- blending.py		// utility functions
	|-- options.py		// configuration for AsyncMMBF


|--Abalation // Abalations study on UTD dataset

	|-- requirements.txt    // the required environment on UTD dataset
	|-- runAbalation.bat    // running script of Abalations study on UTD dataset
	|-- abalation_UTD.py    // main file of AsyncMMBF on UTD dataset
	|-- GlobalScheduling.py	// set up training
	|-- UtdData.py		// load the data for clients in FL
	|-- model.py 	// model configuration for UTD datasets 
	|-- blending.py		// utility functions
	|-- options.py		// configuration for Abalations study

```
# Environment
* Create and activate conda environment for each dataset
```
conda create --name envname python=3.8
conda activate envname
```
* Enter UTD/HA4M folder and install packages according to the corresponding requirements.txt
```
pip install -r requirements.txt
```

# Download datasets
* UTD-MHAD
    * UTD-MHAD dataset follows the previous work " <a href="https://github.com/xmouyang/Cosmo"> Cosmo: Contrastive Fusion Learning with Small Data for Multimodal Human Activity Recognition </a>" and pre-processed dataset can be downloaded in the [google driver folder](https://drive.google.com/drive/folders/1-0qV95bWhVT8rNwh-pBNTAaqKgiGvOeQ?usp=sharing).
    * Put the dataset into the UTD/data folder.
* HA4M
    * The raw HA4M dataset can be downloaded in the https://www.scidb.cn/en/detail?dataSetId=c8d743ad2ea549dfa938cea320a38c46. The pre-processed dataset and sample index can downloaded in the [google driver folder](https://drive.google.com/drive/folders/11jBsTZgz9P1HyxEvkuCvaWvIFRSIBCJd?usp=sharing).
    * Put the dataset into the HA4M/data folder.


# Quick Start 
* Activate the corresponding conda environment for dataset 
```
conda activate envname
```
* Run the script on your machine. Note that iid=0, subnetnet_num=2 represent label-based Non-IID and modality-based Non-IID respectively, vice versa.
```
run.bat
```

# Abalation 
* Ensure the UTD-MHAD dataset has been downloaded and put into the UTD/data folder. Then enter the Abalation folder.
* Run the script on your machine. Note that, client_num, Naplha, hidDim, and chunks represent the number of devices, the values of Dirichlet parameters, Tucker-rank, and CP-rank, respectively. The specifical commands can be shown in the script.
```
conda runAbalation.bat
```
* Executing the code on different numbers of devices. Note that client_num controls the number of devices, thresholdAcc represents the accuracy threshold at which the global model stops training.
```
python abalation_UTD.py --iid=0 --client_num=5 --splitScheme='mix' --subnet_num=2 --epochs=1000 --lr=0.0001 --thresholdAcc=0.8 --isStopByAcc=True
```

 
# Citation
The code and datasets of this project are made available for non-commercial, academic research only. If you find this work useful to you, please cite the following papers:
```  
@article{AsyncMMBF2023,
  title={Multimodal Fusion with Block Term Decomposition for Asynchronous Federated Learning},
  author={},
  journal={},
  year={},
  publisher={}
}
```

