DATA_ROOT=~/4DShell
TRAIN_SET=$DATA_ROOT/datasets/Monkaa/data/320/
python train.py $TRAIN_SET \
--resnet-layers 18 \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epoch-size 1000 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--name Monkaa_r18 \
--image-type 2
