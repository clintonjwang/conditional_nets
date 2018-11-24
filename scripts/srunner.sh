#!/bin/bash
export HOME=/data/vision/polina/users/clintonw
export PATH=$PATH:/data/vision/polina/shared_software/anaconda3/bin
export PROJ=/data/vision/polina/projects/wmh/clintonw
source activate clinton
# -w $3
case $1 in
    cls)
        srun --pty -p gpu -t 0 --mem-per-cpu 6000M --gres=gpu:4 python "$HOME/code/vision_final/scripts/train_classifier.py" $2;;
    regr)
        srun --pty -p gpu -t 0 --mem-per-cpu 6000M --gres=gpu:4 python "$HOME/code/vision_final/scripts/train_regressor.py" $2
esac