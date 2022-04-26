class FLAGS(object):
    """ """
    """
    General settings
    """
    input_size = 256
    heatmap_size = 32
    cpm_stages = 3
    joint_gaussian_variance = 1.0
    center_radius = 17
    num_of_joints = 17
    color_channel = 'RGB'
    normalize_img = True
    use_gpu = True
    gpu_id = 0


    """
    Demo settings
    """
    # 'MULTI': show multiple stage heatmaps
    # 'SINGLE': show last stage heatmap
    # 'Joint_HM': show last stage heatmap for each joint
    # 'image or video path': show detection on single image or video
    DEMO_TYPE = 'SINGLE'

    model_path = 'cpm_body'
    cam_id = 0

    webcam_height = 480
    webcam_width = 640

    use_kalman = True
    kalman_noise = 0.03


    """
    Training settings
    """
    network_def = 'cpm_body'
    train_img_dir = 'omc_dataset_train.tfrecords'
    val_img_dir = 'omc_dataset_val.tfrecords'
    bg_img_dir = ''
    pretrained_model = ''
    batch_size = 8
    init_lr = 0.001
    lr_decay_rate = 0.5
    lr_decay_step = 10000
    training_iters = 100000
    # training_iters = 3000
    verbose_iters = 10
    validation_iters = 5
    model_save_iters = 2500