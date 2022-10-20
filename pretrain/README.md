This directory contains code and data for pre-training step.

### 1. Dataset 

We follow [Peng et al., 2020](https://arxiv.org/abs/2010.01923) and use the same pretraining dataset to pre-train LPD. You can follow this [link](https://github.com/thunlp/RE-Context-or-Names/blob/master/pretrain/README.md) to prepare the pretraining dataset. Sucessful preperation will generate file `./data/CP/rel2cope.json` and `./data/CP/cpdata`. 

The file `./data/CP/rel2cope.json` contains relation to sentences mappings for sentences in  `./data/CP/cpdata`. However, Peng et al., 2020 did not exclude relation types that appear both in the pretraining dataset and the fewrel 1.0 dataset (they only excude the overlapping relation **instnaces**, not the entire relation **type**).

To exclude the overlapping relation types, first go to https://competitions.codalab.org/competitions/27980 to download the fewrel 1.0 test set. Put the data under `./finetune/data`, then run 
```
python data/CP/exclude_fewfel.py
```
A new mapping `./data/CP/rel2cope_excluded.json` that contains no overlapping relation types will be generated.

### 2. Pretrained Model

You can download the [checkpoint](https://drive.google.com/file/d/1HAU6NHoK01Msj-35e_RIr8p-f3ZnrZTl/view?usp=sharing) of LPD pretrained on the original Wikipedia pretraining dataset ), or the [checkpoint](https://drive.google.com/file/d/1zS-xvb5eH6aU8RhEOMvCo4-XeIVBQEAp/view?usp=sharing) pretrained on the filtered out dataset. 


### 3. Pretrain
To pretrain on the original Wikipedia pretraining dataset, use
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2  code/main.py \
	--cuda 0,1  \
	--model CP \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch 20 \
	--gradient_accumulation_steps 16 \
	--max_length 64 \
	--save_step 500 \
	--alpha 0.3 \
	--temperature 0.05 \
	--train_sample \
	--save_dir ckpt_lpd_0.6  --label_mask_prob 0.6 --seed 42 
```

To pretrain on the filtered out dataset, use
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2  code/main.py \
	--cuda 0,1  \
	--model CP \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch 20 \
	--gradient_accumulation_steps 16 \
	--max_length 64 \
	--save_step 500 \
	--alpha 0.3 \
	--temperature 0.05 \
	--train_sample \
	--save_dir ckpt_lpd_0.6_excluded  --label_mask_prob 0.6 --seed 42 --exclude_fewrel
```

### Reference
```
@article{peng2020learning,
  title={Learning from Context or Names? An Empirical Study on Neural Relation Extraction},
  author={Peng, Hao and Gao, Tianyu and Han, Xu and Lin, Yankai and Li, Peng and Liu, Zhiyuan and Sun, Maosong and Zhou, Jie},
  journal={arXiv preprint arXiv:2010.01923},
  year={2020}
}
```