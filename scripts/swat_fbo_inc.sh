python train.py fbo \
    -project fbo \
    -dataset swat \
    -dataroot 'TEEN/data/swat' \
    -model_dir 'checkpoint/swat/fbo/ft_dot-avg_cos-data_init-start_0/0110-10-37-24-172-Epo_100-Bs_128-sgd-Lr_0.1-decay0.0005-Mom_0.9-Max_100-NormF-T_16.00-multi_proto_num_3-knn_epoch_5-alpha1_1/session0_max_acc.pth' \
    -base_mode 'ft_cos' \
    -new_mode 'avg_cos' \
    -lr_base 0.1 \
    -decay 0.0005 \
    -epochs_base 0 \
    -batch_size_base 128 \
    -test_batch_size 128 \
    -schedule Cosine \
    -tmax 100 \
    -gpu '0' \
    -temperature 16 \
    -multi_proto_num 3 \
    -knn_epoch 5 \
    -alpha1 1 \
    -seed 3407