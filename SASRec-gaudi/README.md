This is Gaudi-v2 version implementation for [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781). The original code for SASRec can be found here: [TensorFlow version](https://github.com/kang205/SASRec), and [CUDA version](https://github.com/pmixer/SASRec.pytorch).

## Datasets:

We have implemented `data_preprocess.py` and `utils.py` to handle the Amazon 2023 datasets.

By running `main.py`, the datasets are automatically downloaded and preprocessed.


## Model Training:

E.g., train SASRec on `raw_Industrial_and_Scientific`
```
python main_rawdata.py --dataset Industrial_and_Scientific --maxlen 10 --device hpu
```

E.g., train SASRec on `5_core_Industrial_and_Scientific`
```
python main.py --dataset Industrial_and_Scientific --maxlen 10 --device hpu
```


- **Issue**: When using the HPU, the loss does not decrease during training.

## Run (CPU):
```
python main_rawdata.py --dataset Industrial_and_Scientific --maxlen 10 --device cpu
```

Or (5-Core)
```
python main.py --dataset Industrial_and_Scientific --maxlen 10 --device cpu
```

## Run (CUDA):
```
python main_rawdata.py --dataset Industrial_and_Scientific --maxlen 10 --device 0
```

Or (5-Core)
```
python main.py --dataset Industrial_and_Scientific --maxlen 10 --device 0
```


- **Results**: Training works correctly on the CPU, with the loss decreasing as expected.

## Run (HPU with `nn.Parameter`):
```
python main_rawdata.py --dataset Industrial_and_Scientific --maxlen 10 --device hpu --nn_parameter
```

Or (5-Core)
```
python main.py --dataset Industrial_and_Scientific --maxlen 10 --device hpu --nn_parameter
```
- **Results**: When using `nn.Parameter` instead of `nn.Embedding`, training works correctly on the HPU, and the loss decreases as expected.

