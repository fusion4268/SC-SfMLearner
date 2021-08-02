DATASET_DIR=/home/fusion4268/4DShell/datasets/v4/
OUTPUT_DIR=vo_results/v4.txt

POSE_NET=checkpoints/resnet50_pose_256/exp_pose_model_best.pth.tar

python test_ours.py \
--img-height 256 --img-width 832 \
--sequence 1280 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

