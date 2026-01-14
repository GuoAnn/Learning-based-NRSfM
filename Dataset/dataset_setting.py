#This function save the parameter of the dataset

dataset_params = {
########################### Kinect T Shirt Dataset ###################################################
    "results_base_folder" :"KINECT_TSHIRT",# "results/",
    "preprocessed_W_filename" : "KINECT_TSHIRT/measurement_matrix_W/W.txt",#"KINECT_TSHIRT/measurement_matrix_W/W.txt",

    "dataset_size": 300,
    "point_tracks_used_for_rigid_init": 300,
    "downsample": True,
    "K_inv": [528,528],
    "downsample_size": 2000,
    "full_Mat_folder": "D:/NRSfM/NIPS2022_Yongbo/nnrsfm_datasets/KINECT_TSHIRT/mat_file/matlab.mat",
    "load_data": True,


    #"gt_images_location": "/KINECT_TSHIRT/seq/",
    #"gt_images_file" : "gt_files/gt_images_kinect_seq.txt",
    "save_or_load" :  "load"#"save"
#####################################################################################################
}