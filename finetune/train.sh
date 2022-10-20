CUDA_VISIBLE_DEVICES=$1 python train_demo.py \
	--trainN 5 --N 5 --K 1 --Q 1 \
	--val val_wiki --test val_wiki \
	--model proto --encoder bert --hidden_size 768 --val_step 1000 \
	--mode OC \
  	--batch_size 4 --fp16 \
	--seed $2 \
	--path $3  
