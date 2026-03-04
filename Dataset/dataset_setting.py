#This function save the parameter of the dataset

dataset_params = {
########################### Kinect T Shirt Dataset ###################################################
    "base_dir": "/home/gax/NRSfM_dataset",
    "dataset_name": "dense_dataset/Dense_paper",
    #"results_base_folder" :"/home/gax/NRSfM_dataset/results",# "results/",
    "preprocessed_W_filename" : "/home/gax/NRSfM_dataset/nnrsfm_datasets/KINECT_TSHIRT/measurement_matrix_W/W.txt",#"KINECT_TSHIRT/measurement_matrix_W/W.txt",

    "dataset_size": 300,
    "point_tracks_used_for_rigid_init": 3000,
    "downsample": True,
    "K_inv": [528,528],#[528,528],challenge为[1,1]
    "downsample_size": 2000,
    #"full_Mat_folder": "/home/gax/NRSfM_dataset/tradition_dataset/Flag/matlab.mat",
    "load_data": True,


    #"gt_images_location": "/KINECT_TSHIRT/seq/",
    #"gt_images_file" : "gt_files/gt_images_kinect_seq.txt",
    "save_or_load" : "load" #"save" "load"
#####################################################################################################
}
