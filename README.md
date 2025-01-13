## Gaudi-v2: For recommendation models

This repository is designed for implementing and testing recommendation models using Gaudi-v2.

We demonstrate our implementation of recommendation models, focusing on integrating multimodal inputs and utilizing Intel Gaudi-v2 devices for efficient computation. The primary goal of this repository is to provide the following contributions:

1. Implementaion of Traditional Recommendation Models
    - Models such as Matrix Factorization (MF) and SASRec have been implemented and optimized for training within the Gaudi-v2 environment.
2. Development of an LLM-based Recommendation Model
    - We have implemented LLM-based recommendation models, called TALLRec, LLaRA, CoLLM, and A-LLMRec using Gaudi-v2.
3. Exploration of MLLM-based Recommendation
    - By leveraging Gaudi-v2, we aim to integrate multimodal data (e.g., visual, audio) to enhance item representations beyond text-based features. This exploration provides insights into improving recommendation performance through multimodal learning.
4. Issue Identification
    - While implementing and training these models on Gaudi-v2, we identified several challenges related to the Gaudi-v2
        - About `nn.Embedding` [Issue](https://github.com/NAVER-INTEL-Co-Lab/gaudi_recsys_public/tree/master/SASRec-gaudi).
        - About `LoRA` [Link](https://github.com/NAVER-INTEL-Co-Lab/gaudi_recsys_public/blob/master/On_Going_for_Gaudi/bit_load_error.ipynb)
        
            Huggingface doesn’t support 8-bit operations on Gaudi-v2 
        - About `save_pretrained` of Huggingface [Link](https://github.com/NAVER-INTEL-Co-Lab/gaudi_recsys_public/blob/master/On_Going_for_Gaudi/bit_load_error.ipynb)

            Gaudi-v2 raises an error if model weights are not converted to CPU before saving.

-------

Currently, we have implemented two traditional recommendation models and four LLM-based recommendation models:
- Traditional recommendation models
    - PMF: [Code](https://github.com/NAVER-INTEL-Co-Lab/gaudi_recsys_public/tree/master/MF-gaudi)
    - SASRec: [Paper](https://arxiv.org/abs/1808.09781) / [Code](https://github.com/NAVER-INTEL-Co-Lab/gaudi_recsys_public/tree/master/SASRec-gaudi)

- LLM-based recommendation models: [Codes](https://github.com/NAVER-INTEL-Co-Lab/gaudi_recsys_public/tree/master/On_Going_for_Gaudi)
    - TALLRec: [Paper](https://arxiv.org/abs/2305.00447)
    - LLaRA: [Paper](https://arxiv.org/abs/2312.02445)
    - CoLLM: [Paper](https://arxiv.org/abs/2310.19488)
    - A-LLMRec: [Paper](https://arxiv.org/abs/2404.11343) / (Code for [Opt-6.7b](https://github.com/NAVER-INTEL-Co-Lab/gaudi_recsys_public/tree/master/A-LLMRec-gaudi))


------


Checklists for improving implementation of recommendation models
- [x] Distributed Data Parallel: A-LLMRec Stage 1
- [x] Distributed Data Parallel: A-LLMRec Stage 2
- [x] Distributed Data Parallel: A-LLMRec Inference
- [ ] Distributed Data Parallel: Automatically find `world_size` (initialize_distributed_hpu)
- [ ] `nn.Parameter` -> `nn.Embedding`