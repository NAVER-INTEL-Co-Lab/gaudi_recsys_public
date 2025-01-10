## Gaudi-v2: For recommendation models

This repository is designed for implementing and testing recommendation models using Gaudi-v2.

We demonstrate our implementation of recommendation models, focusing on integrating multimodal inputs and utilizing Intel Gaudi-v2 devices for efficient computation. The primary goal of this repository is to provide the following contributions:

1. Implementaion of Traditional Recommendation Models
    - Models such as Probabilistic Matrix Factorization (PMF) and SASRec have been implemented and optimized for training within the Gaudi-v2 environment.
2. Development of an LLM-based Recommendation Model
    - We have implemented LLM-based recommendation models, called TALLRec, LLaRA, CoLLM, and A-LLMRec using Gaudi-v2.
3. Exploration of MLLM-based Recommendation
    - By leveraging Gaudi-v2, we aim to integrate multimodal data (e.g., visual, audio) to enhance item representations beyond text-based features. This exploration provides insights into improving recommendation performance through multimodal learning.
4. Issue Identification
    - While implementing and training these models on Gaudi-v2, we identified several challenges related to the Gaudi-v2

-------

Currently, we have implemented two traditional recommendation models and four LLM-based recommendation models:
- Traditional recommendation models
    - PMF: [Paper](https://dl.acm.org/doi/10.5555/2981562.2981720) / [Code](https://github.com/NAVER-INTEL-Co-Lab/gaudi-recsys/tree/main/MF-gaudi)
    - SASRec: [Paper](https://arxiv.org/abs/1808.09781) / [Code](https://github.com/NAVER-INTEL-Co-Lab/gaudi-recsys/tree/main/SASRec-gaudi)

- LLM-based recommendation models: [Codes](https://github.com/NAVER-INTEL-Co-Lab/gaudi-recsys/tree/main/Seq_Exp/new_recsys)
    - TALLRec: [Paper](https://arxiv.org/abs/2305.00447)
    - LLaRA: [Paper](https://arxiv.org/abs/2312.02445)
    - CoLLM: [Paper](https://arxiv.org/abs/2310.19488)
    - A-LLMRec: [Paper](https://arxiv.org/abs/2404.11343) / (Code for [Opt-6.7b](https://github.com/NAVER-INTEL-Co-Lab/gaudi-recsys/tree/main/A-LLMRec-gaudi))


------


Checklists for improving implementation of recommendation models
- [x] Distributed Data Parallel: A-LLMRec Stage 1
- [x] Distributed Data Parallel: A-LLMRec Stage 2
- [x] Distributed Data Parallel: A-LLMRec Inference
- [ ] Distributed Data Parallel: Automatically find `world_size` (initialize_distributed_hpu)
- [ ] `nn.Parameter` -> `nn.Embedding`