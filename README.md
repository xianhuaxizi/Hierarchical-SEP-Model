# Hierarchical-SEP-Model

We proposed a new hierarchical neural network model for more accurate script event prediction. 

## Prerequisites
- linux
- Python 3.6
- CPU or NVIDIA GPU + CUDA CuDNN
- \>= PyTorch 1.8.0 


## Paper data, models and Code

We use the same dataset as [SGNN](https://github.com/eecrazy/ConstructingNEEG_IJCAI_2018). You can download the dataset as follows:
- NYT dataset [[Google Drive]](https://drive.google.com/file/d/1zXTBHeBCWESX7kaAG6Q01YhUJrEl3V1j/view?usp=sharing) [[Baidu Drive]](https://pan.baidu.com/s/1pOBlOtxNIjU_ywf1_6Witg)(eg89)

The trained models can be downloaded via:
- Hierarchical-SEP-Model_best_acc_73.44.pth [[Google Drive]](https://drive.google.com/file/d/1JlSA8IfZ9sP5_rtqPjwKjVBAW2zIWqU0/view?usp=sharing) [[Baidu Drive]](https://pan.baidu.com/s/1hXGraw4f7ZcgB-RRrStK1Q)(pvsa)

**Codes Structures:**

    +--- data
    |   +--- metadata
    |   |   +--- deepwalk_128_unweighted_with_args.txt
    |   |   +--- dev_index.pickle
    |   |   +--- event_chain_dev.json
    |   |   +--- event_chain_test.json
    |   |   +--- event_chain_test.txt
    |   |   +--- event_chain_train.json
    |   |   +--- test_index.pickle
    |   |   +--- test_index.txt
    |   |   +--- vocab.json
    |   |   +--- vocab_index_dev.data
    |   |   +--- vocab_index_test.data
    |   |   +--- vocab_index_train.data
    |   |   +--- word_embedding.npy
    +--- Hierarchical-SEP-Model
    |   +--- checkpoints
    |   |   +--- Hierarchical-SEP-Model_best_acc_73.44.pth
    |   +--- config.py
    |   +--- script_inference.py  # running this file to predict
    |   +--- setup.py
    |   +--- src
    |   |   +--- datasources
    |   |   |   +--- ScriptData.py
    |   |   |   +--- __init__.py
    |   |   +--- models
    |   |   |   +--- base_model.py
    |   |   |   +--- my_modules.py
    |   |   |   +--- ScriptNet.py
    |   |   |   +--- STPredictor.py
    |   |   |   +--- transformer.py
    |   |   |   +--- __init__.py
    |   |   +--- util
    |   |   |   +--- osutils.py
    |   |   |   +--- tools.py

## How to run the code?

Just download the dataset, models and codes, and place them as the above settings. Then you can run the corresponding files to predict.
It is very easy! Just enjoy it!

## Official implementation of some baseline models
- [SGNN](https://github.com/eecrazy/ConstructingNEEG_IJCAI_2018)
- [MCer](https://github.com/YueAWu/MCer)
- [EMDF-Net](https://github.com/xianhuaxizi/EMDF-Net)

## Questions
Please contact zhoupengpeng@bupt.edu.cn. 
