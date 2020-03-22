"""
Various important parameters of our model and training procedure.
"""


def get_general_params():
    param = {}
    dn = 1
    param['IMG_HEIGHT'] = int(256/dn)
    param['IMG_WIDTH'] = int(256/dn)
    param['obj_scale_factor'] = 1.14/dn
    param['scale_max'] = 1.05  # Augmentation scaling
    param['scale_min'] = 0.90
    param['max_rotate_degree'] = 5
    param['max_sat_factor'] = 0.05
    param['max_px_shift'] = 10
    param['posemap_downsample'] = 2
    param['sigma_joint'] = 7/4.0

    # old version
    # param['n_joints'] = 14
    # param['n_limbs'] = 10
    # # Using MPII-style joints: head (0), neck (1), r-shoulder (2), r-elbow (3), r-wrist (4), l-shoulder (5),
    # # l-elbow (6), l-wrist (7), r-hip (8), r-knee (9), r-ankle (10), l-hip (11), l-knee (12), l-ankle (13)
    # param['limbs'] = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [2, 5, 8, 11]]

    # # new version COCO datasets.
    param['n_joints'] = 16
    param['n_limbs'] = 15
    # Using COCO joints: 0-'nose', 1-'neck', 2-'Rsho', 3-'Relb', 4-'Rwri',5-'Lsho', 6-'Lelb', 7-'Lwri', 8-'Rhip',
    # 9-'Rkne', 10-'Rank', 11-'Lhip', 12-'Lkne', 13-'Lank', 14-'Leye', 15-'Reye', 16-'Lear', 17-'Rear', 18-'pt19'
    param['limbs'] = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [1, 11], [8, 9], [9, 10], [11, 12], [12, 13], [0, 14],
                       [0, 15]]


    param['n_training_iter'] = 200000
    param['test_interval'] = 500
    param['model_save_interval'] = 1000
    param['project_dir'] = r'..'
    # param['project_dir'] = r'drive/My Drive'
    param['model_save_dir'] = param['project_dir'] + '/models'
    # param['data_dir'] = param['project_dir']+'/animeWarp'
    param['data_dir'] = r'D:/download_cache/animeWarp'
    param['batch_size'] = 16
    param['load_weights'] = param['model_save_dir']+'/gan5000.h5'
    return param

