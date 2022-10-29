## LPD
Pytorch implementation of our EMNLP 2022 paper [Better Few-Shot Relation Extraction with Label Prompt Dropout](https://arxiv.org/abs/2210.13733)
```
@inproceedings{zhang-etal-2022-better,
  title = "Better Few-Shot Relation Extraction with Label Prompt Dropout",
  author = "Zhang, Peiyuan and Lu, Wei",
  booktitle = "Proceedings of EMNLP", 
  year = {2022},
}
```

This codebase is adapted from https://github.com/thunlp/RE-Context-or-Names. 
```
@article{peng2020learning,
  title={Learning from Context or Names? An Empirical Study on Neural Relation Extraction},
  author={Peng, Hao and Gao, Tianyu and Han, Xu and Lin, Yankai and Li, Peng and Liu, Zhiyuan and Sun, Maosong and Zhou, Jie},
  journal={arXiv preprint arXiv:2010.01923},
  year={2020}
}
```

### Quick Start

You can quickly run our code by following steps:

- Install dependencies as described in following section. 
- cd to `pretrain` or `finetune` directory then download and pre-processing data for pre-traing or finetuning.    


### 1. Dependencies

We run our experiments on cuda 11.1.
Run the following script to install dependencies.

```shell
pip install -r requirement.txt
```

**You need install transformers and apex manually.**

**transformers**
We use huggingface transformers to implement Bert.  And for convenience, we have downloaded  [transformers](https://github.com/huggingface/transformers) into `utils/`. And we have also modified some lines in the class `BertForMaskedLM` in `src/transformers/modeling_bert.py` while keep the other codes unchanged. 

You just need run 
```
pip install .
```
to install transformers manually.

**apex**
Install [apex](https://github.com/NVIDIA/apex) under the offical guidance.

### 2. More details
You can check the readme file in `pretrain` or `finetune` to learn more details about pre-training or finetuning.







