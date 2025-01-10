## Gaudi-v2: For recommendation models

This repository is designed for implementing and testing llm-based recommendation models using Gaudi-v2.

In this repository, we have implemented three llm-based recommendation models:
- [**TALLRec**](https://arxiv.org/abs/2305.00447)
- [**LLaRA**](https://arxiv.org/abs/2312.02445)
- [**CoLLM**](https://arxiv.org/abs/2310.19488)
- We use LLaMA-3.2-3b-instruct.

## Pre-train CF-RecSys (SASRec)

Due to **LLaRA** and **CoLLM** use CF-RecSys, be sure to pretrain CF-RecSys model before train baselines.

```
cd SeqRec/sasrec
python main.py --device hpu --dataset Industrial_and_Scientific --nn_parameter
```


## Baseline Train - Item Title Generation

If you train SASRec using the `--nn_parameter` flag, be sure to use `--nn_parameter` for both training and inference of baseline.

To choose a model among **TALLRec**, **CoLLM**, **LLaRA**, and **A-LLMRec** use the `--baseline` argument with one of the following options:

- `tallrec`
- `Collm`
- `llara`
- `a-llmrec`

```
python title_generation_main_baseline.py --device hpu --pretrain_stage2 --rec_pre_trained_data Industrial_and_Scientific --nn_parameter --baseline tallrec --save_dir tallrec --bath_size2 2
```

```
python title_generation_main_baseline.py --device hpu --pretrain_stage2 --rec_pre_trained_data Industrial_and_Scientific --nn_parameter --baseline Collm --save_dir Collm --bath_size2 2
```

```
python title_generation_main_baseline.py --device hpu --pretrain_stage2 --rec_pre_trained_data Industrial_and_Scientific --nn_parameter --baseline llara --save_dir llara --bath_size2 2
```

```
python title_generation_main_baseline.py --device hpu --pretrain_stage2 --rec_pre_trained_data Industrial_and_Scientific --nn_parameter --baseline a-llmrec --save_dir a-llmrec
```

### Evaluation
Inference stage generates "recommendation_output.txt" file and write the recommendation result generated from the LLMs into the file. To evaluate the result, run the eval.py file.

```
python title_generation_main_baseline.py --device hpu --inference --rec_pre_trained_data Industrial_and_Scientific --nn_parameter --baseline llara --save_dir llarabest
python eval.py
```

## Baseline Train - Item Retrieval

If you train SASRec using the `--nn_parameter` flag, be sure to use `--nn_parameter` for both training and inference of baseline.

In item retrieval, special tokens are generated using `nn.Embedding`. However, due to issues with training `nn.Embedding` on Gaudi-v2, tokens need to be generated using `nn.Parameter` instead for proper training.

The current code is configured to automatically use `nn.Parameter` when running on HPU.

To choose a model among **TALLRec**, **CoLLM**, **LLaRA**, and **A-LLMRec** use the `--baseline` argument with one of the following options:

- `tallrec`
- `Collm`
- `llara`
- `a-llmrec`

```
python main_baseline.py --device hpu --pretrain_stage2 --rec_pre_trained_data Industrial_and_Scientific --nn_parameter --baseline tallrec --save_dir tallrec --bath_size2 2
```

```
python main_baseline.py --device hpu --pretrain_stage2 --rec_pre_trained_data Industrial_and_Scientific --nn_parameter --baseline Collm --save_dir Collm --bath_size2 2
```

```
python main_baseline.py --device hpu --pretrain_stage2 --rec_pre_trained_data Industrial_and_Scientific --nn_parameter --baseline llara --save_dir llara --bath_size2 2
```

```
python main_baseline.py --device hpu --pretrain_stage2 --rec_pre_trained_data Industrial_and_Scientific --nn_parameter --baseline a-llmrec --save_dir a-llmrec --bath_size2 2
```

### Evalutaion

```
python main_baseline.py --device hpu --inference --rec_pre_trained_data Industrial_and_Scientific --nn_parameter --baseline tallrec --save_dir tallrecbest
```