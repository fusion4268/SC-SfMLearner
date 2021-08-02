DATASET_DIR=/home/fusion4268/4DShell/datasets/Monkaa/data/320/
OUTPUT_DIR=vo_results/Monkaa/

POSE_NET=checkpoints/Monkaa_r18/07-27-19:01/exp_pose_model_best.pth.tar

python test_ours.py \
--img-height 160 --img-width 320 \
--sequence "treeflight_x2" \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

python test_ours.py \
--img-height 160 --img-width 320 \
--sequence "flower_storm_augmented1_x2" \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

python test_ours.py \
--img-height 160 --img-width 320 \
--sequence "lonetree_difftex_x2" \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR

python test_ours.py \
--img-height 160 --img-width 320 \
--sequence "lonetree_difftex2_x2" \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR
