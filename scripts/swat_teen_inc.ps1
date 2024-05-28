c:/Users/27220/AppData/Local/miniconda3/envs/python3.8_pytorch1.12.1/python.exe train.py teen `
    -project teen `
    -dataset swat `
    -dataroot 'TEEN/data/swat' `
    -model_dir 'checkpoint\swat\warp\ft_dot-avg_cos-data_init-start_0\0528-15-49-31-575-Epo_100-Bs_128-sgd-Lr_0.1-decay0.0005-Mom_0.9-Max_100-NormF-T_16.00-alpha_0.05\session0_max_acc.pth' `
    -base_mode 'ft_dot' `
    -new_mode 'avg_cos' `
    -lr_base 0.1 `
    -lr_new 0.01 `
    -decay 0.0005 `
    -epochs_base 0 `
    -batch_size_base 128 `
    -test_batch_size 128 `
    -schedule Cosine `
    -tmax 100 `
    -gpu '0' `
    -temperature 16 `
    -shift_weight 0.1 `
    -soft_mode 'soft_proto' `
    -seed 3407