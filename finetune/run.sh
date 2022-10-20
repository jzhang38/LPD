
for seed in 42
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
	--path  ../ckpt/ckpt_lpd_0.6_excluded/ckpt_of_step_1620 --save_ckpt checkpoint/ckpt_cp_0.6_${prob}_seed_42_excluded_seed_${seed}_${Nway}N_${Kshot}K   --label_mask_prob ${prob}   
	done
	done
	done
done




for seed in 43
do
	for Nway in 10
	do
	for Kshot in 1
	do
	for prob in  0.4 
	do
	CUDA_VISIBLE_DEVICES=1 python train_demo.py \
	--trainN ${Nway} --N ${Nway} --K ${Kshot} --Q 1 \
	--val val_wiki --test val_wiki \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
	--mode CM \
  	--batch_size 4 --fp16 \
	--seed ${seed} \
    --save_ckpt checkpoint/no_pretrain_seed_${seed}_${Nway}N_${Kshot}K   --label_mask_prob ${prob} 
	done
	done
	done
done


for seed in 42 43 44
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
	--path  ../ckpt/ckpt_lpd_0.6/ckpt_of_step_3780 --save_ckpt checkpoint/ckpt_cp_0.6_${prob}_seed_42_seed_${seed}_${Nway}N_${Kshot}K   --label_mask_prob ${prob}
	done
	done
	done
done



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
	--load_ckpt checkpoint/ckpt_cp_0.6_${prob}_seed_42_excluded_seed_${seed}_${Nway}N_${Kshot}K   --label_mask_prob ${prob}   --only_test
	done
	done
	done
done



for seed in 43
do
	for Nway in 10
	do
	for Kshot in 1
	do
	for pretrain_prob in 0.6
	do
	for prob in 0.8
	do
	for checkpoint in 1620
	do
	CUDA_VISIBLE_DEVICES=2 python train_demo.py \
	--trainN ${Nway} --N ${Nway} --K ${Kshot} --Q 1 \
	--val val_pubmed --test val_pubmed \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
	--mode CM \
  	--batch_size 4 --fp16 \
	--seed ${seed} \
	--path  ../ckpt/to_submit/ckpt_of_step_${checkpoint} --save_ckpt checkpoint/ckpt_lpd_${pretrain_prob}_${prob}_${checkpoint}_excluded_pubmed_dev_seed_${seed}_1620  --label_mask_prob ${prob}  --train_iter 1000 --test_iter 1000
	done
	done
	done
	done
	done
done