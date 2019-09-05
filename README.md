# DeepZip

## Description
DNA_compression using neural networks


## Requirements
0. GPU
1. python 2
2. numpy
3. sklearn
4. keras 2.2.2
5. tensorflow (gpu) 1.8

## Code
To run a compression experiment: 

### Data Preparation
1. All the datasets can be downloaded using the scripts provided in the folder [Datasets](./Datasets)


Run the following command
```bash get_data.sh
```

### Running models
1. All the models are listed in models.py
2. Pick a model, to run compression experiment on all the data files in the data/files_to_be_compressed directory

```
cd src
./run_experiments.sh biLSTM
```

