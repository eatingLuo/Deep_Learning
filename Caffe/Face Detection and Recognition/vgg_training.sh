#!/usr/bin/env sh

/home/wangfei/caffe/build/tools/caffe train \
    --solver=/home/wangfei/ML2FinalProject/project_full_code/data/solver.prototxt \
    --weights=/home/wangfei/ML2FinalProject/project_full_code/data/VGG_FACE.caffemodel -gpu=0 2>&1 | tee -a /home/wangfei/ML2FinalProject/project_full_code/data/log/my_model.log

/home/wangfei/caffe/tools/extra/plot_training_log.py.example 0 test_accuracy.png /home/wangfei/ML2FinalProject/project_full_code/data/log/my_model.log
/home/wangfei/caffe/tools/extra/plot_training_log.py.example 4 learning_rate.png /home/wangfei/ML2FinalProject/project_full_code/data/log/my_model.log
/home/wangfei/caffe/tools/extra/plot_training_log.py.example 6 train_loss.png /home/wangfei/ML2FinalProject/project_full_code/data/log/my_model.log
