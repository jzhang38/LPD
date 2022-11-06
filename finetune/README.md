## Finetuned Model

For better reproducibility, we have uploaded our model's [finetuned checkpoint](https://drive.google.com/drive/folders/1IRzpQPY1pSOrJJRvA9hYotfTUdkrcATw?usp=sharing) as well as the test set [prediciton file](https://drive.google.com/drive/folders/1SRB0M-hrQWK__JkAzypyc-TBxMGWlLaL?usp=sharing) to google drive.

The following sections provide script to finetune LPD on your machine.

## Finetune on FewRel 1.0

To train LPD from scratch on FewRel 1.0, run

```shell
for seed in 43
do
	for Nway in 10
	do
	for Kshot in 1
	do
	for prob in  0.4 
	do
	CUDA_VISIBLE_DEVICES=2 python train_demo.py \
	--trainN ${Nway} --N ${Nway} --K ${Kshot} --Q 1 \
	--val val_wiki --test val_wiki \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
	--mode CM \
  	--batch_size 4 --fp16 \
	--seed ${seed} \
    --save_ckpt checkpoint/[name your checkpoint]   --label_mask_prob ${prob} 
	done
	done
	done
done
```

To finetine LPD based on a pretrained checkpoint:

```shell
for seed in 43
do
	for Nway in 10
	do
	for Kshot in 1
	do
	for prob in  0.4 
	do
	CUDA_VISIBLE_DEVICES=2 python train_demo.py \
	--trainN ${Nway} --N ${Nway} --K ${Kshot} --Q 1 \
	--val val_wiki --test val_wiki \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
	--mode CM \
  	--batch_size 4 --fp16 \
	--seed ${seed} \
	--path  [path_to_pretrained_checkpoint] --save_ckpt checkpoint/[name your checkpoint]  --label_mask_prob ${prob}
	done
	done
	done
done
```


To evaluate on the FewRel 1.0 test set, first go to https://competitions.codalab.org/competitions/27980 to download the unlabelled fewrel 1.0 test data. Run the following script to generate the prediction file:

```shell
for seed in 43
do
	for Nway in 10
	do
	for Kshot in 1
	do
	CUDA_VISIBLE_DEVICES=2 python train_demo.py \
	--trainN ${Nway} --N ${Nway} --K ${Kshot} --Q 1 \
	--val val_wiki --test data/test_wiki_input-${Nway}-${Kshot}.json \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
	--mode CM \
  	--batch_size 4 --fp16 \
	--seed ${seed} \
    --load_ckpt [path_to_the_saved_checkpoint_generated_after_finetuning]   --do_prediction --prediction_save_path prediction/[name your prediction file]/
	done
	done
done
```

The predictions will be saved to `./finetune/prediction`. Follow the guide on https://competitions.codalab.org/competitions/27980 to submit your prediction file.

## For FewRel 2.0 Domain Adaptation

To finetune LPD with early stopping based on FewRel 2.0 validation accuracy, run 

```shell
for seed in 43
do
	for Nway in 10
	do
	for Kshot in 1
	do
	for prob in 0.8
	do
	for checkpoint in 1500
	do
	CUDA_VISIBLE_DEVICES=2 python train_demo.py \
	--trainN ${Nway} --N ${Nway} --K ${Kshot} --Q 1 \
	--val val_pubmed --test val_pubmed \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
	--mode CM \
  	--batch_size 4 --fp16 \
	--seed ${seed} \
	--path  [path_to_pretrained_checkpoint] --save_ckpt checkpoint/[name your checkpoint]   --label_mask_prob ${prob}  --train_iter 1000 --test_iter 1000
	done
	done
	done
	done
done
```

To evaluate on the FewRel 2.0 test set, first go to https://codalab.lisn.upsaclay.fr/competitions/7397 to download the unlabelled fewrel 2.0 test data.  Run the following script to generate the prediction file:

```shell
for seed in 43
do
	for Nway in 10 5
	do
	for Kshot in 5 1
	do
	CUDA_VISIBLE_DEVICES=2 python train_demo.py \
	--trainN ${Nway} --N ${Nway} --K ${Kshot} --Q 1 \
	--val val_wiki --test data/test_pubmed_input-${Nway}-${Kshot}.json \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
	--mode CM \
  	--batch_size 4 --fp16 \
	--seed ${seed} \
    --load_ckpt  [path_to_the_saved_checkpoint_generated_after_finetuning]  --do_prediction  --prediction_save_path prediction/[name your prediction file]/
	done
	done
	done
done
```

The predictions will be saved to `./finetune/prediction`. Follow the guide on https://codalab.lisn.upsaclay.fr/competitions/7397 to submit your prediction file.








