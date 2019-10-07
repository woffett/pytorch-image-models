PYTHON=/home/jonathanli/anaconda3/envs/ml/bin/python
MAIN=/home/jonathanli/Documents/stanford_fall2019/hazy/pytorch-image-models/train.py
DATA=/home/jonathanli/datasets
OUTPUT=/home/jonathanli/Documents/stanford_fall2019/hazy/pytorch-image-models/cifar10_results
SEED=1

$PYTHON $MAIN $DATA \
	--model efficientnet_b0 \
	--num-classes 10 \
	--img-size 32 \
	--batch-size 128 \
	--use-cifar \
	--lr 0.1 \
	--epochs 200 \
	--decay-epochs 80 \
	--warmup-epochs 0 \
	--seed $SEED \
	--output $OUTPUT \
	--num-gpu 0

