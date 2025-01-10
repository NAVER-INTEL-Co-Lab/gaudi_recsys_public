This is Gaudi-v2 version implementation for [Probabilistic Matrix Factorization](https://dl.acm.org/doi/10.5555/2981562.2981720).

## Datasets:

We have implemented `data_preprocess.py` and `utils.py` to handle the Amazon 2023 datasets.

By running `main.py`, the datasets are automatically downloaded and preprocessed.

## Model Training:

E.g., train SASRec on `raw_Industrial_and_Scientific `
```
python main.py --dataset Industrial_and_Scientific  --device hpu
```

- **Issue**: When using the HPU, the loss does not decrease during training.


## Run (CPU):
```
python main.py --dataset Industrial_and_Scientific  --device cpu
```

- **Results**: Training works correctly on the CPU, with the loss decreasing as expected.

## Run (HPU with `nn.Parameter`):
```
python main.py --dataset Industrial_and_Scientific  --device hpu --nn_parameter
```
- **Results**: When using `nn.Parameter` instead of `nn.Embedding`, training works correctly on the HPU, and the loss decreases as expected.
