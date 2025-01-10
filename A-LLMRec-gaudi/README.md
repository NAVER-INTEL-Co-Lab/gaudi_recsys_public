The Gaudi-v2 implementation for A-LLMRec : Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System [paper](https://arxiv.org/abs/2404.11343), accepted at **KDD 2024**.

## Dataset
Download [dataset of 2018 Amazon Review dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) for the experiment. Should download metadata and reviews files and place them into data/amazon direcotory.

```
cd data/amazon
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Movies_and_TV.json.gz  # download review dataset
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz  # download metadata
gzip -d meta_Movies_and_TV.json.gz
```

## Pre-train CF-RecSys (SASRec)
```
cd pre_train/sasrec
python main.py --device hpu --dataset Movies_and_TV --nn_parameter
```

Due to [issues](https://github.com/Sein-Kim/Gaudi-Git/issues/2) with `nn.Embeddding`, use the `--nn_parameter` flag to train SASRec.

## A-LLMRec Train

If you train SASRec using the `--nn_parameter` flag, be sure to use `--nn_parameter` for both training and inference of A-LLMRec.

- train stage1
```
cd ../../
python main.py --pretrain_stage1 --rec_pre_trained_data Movies_and_TV --nn_parameter
```

```
#For DDP
python main.py --pretrain_stage1 --rec_pre_trained_data Movies_and_TV --nn_parameter --multi_gpu --world_size 8
```

- train stage2
```
python main.py --pretrain_stage2 --rec_pre_trained_data Movies_and_TV --nn_parameter
```

```
#For DDP
python main.py --pretrain_stage2 --rec_pre_trained_data Movies_and_TV --nn_parameter --multi_gpu --world_size 8
```

## Evaluation
Inference stage generates "recommendation_output.txt" file and write the recommendation result generated from the LLMs into the file. To evaluate the result, run the eval.py file.

```
python main.py --inference --rec_pre_trained_data Movies_and_TV --nn_parameter
python eval.py
```
