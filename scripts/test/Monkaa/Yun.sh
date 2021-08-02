DATASET_DIR=/home/fusion4268/4DShell/datasets/
OUTPUT_DIR=vo_results/Monkaa/

POSE_NET=checkpoints/Monkaa_r18/07-27-19:01/exp_pose_model_best.pth.tar

python test_ours.py \
--img-height 160 --img-width 320 \
--sequence "v1" \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR \
--dir_suffix "/640/image_0/"

python test_ours.py \
--img-height 160 --img-width 320 \
--sequence "v2" \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR \
--dir_suffix "/640/image_0/"

python test_ours.py \
--img-height 160 --img-width 320 \
--sequence "v3" \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR \
--dir_suffix "/640/image_0/"

python test_ours.py \
--img-height 160 --img-width 320 \
--sequence "v4" \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR \
--dir_suffix "/640/image_0/"
