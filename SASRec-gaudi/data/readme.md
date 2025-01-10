Get datasets (Amazon 2023)

Amazon 2023 citation

@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}


wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz

wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Amazon_Fashion.jsonl.gz


wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz


```
from datasets import load_dataset

dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
print(dataset["full"][0])

```