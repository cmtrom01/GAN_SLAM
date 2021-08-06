python associate.py /home/chris/GAN_SLAM/data/rgbd_dataset_freiburg3_sitting_xyz/rgb.txt /home/chris/GAN_SLAM/data/rgbd_dataset_freiburg3_sitting_xyz/depth.txt > ORB_SLAM2/associations.txt

ORB_SLAM2/Examples/RGB-D/rgbd_tum ORB_SLAM2/Vocabulary/ORBvoc.txt ORB_SLAM2/Examples/RGB-D/TUM1.yaml /home/chris/GAN_SLAM/data/rgbd_dataset_freiburg3_sitting_xyz ORB_SLAM2/associations.txt

evo_traj tum '/home/chris/GAN_SLAM/ORB_SLAM2/CameraTrajectory.txt' --ref='/home/chris/GAN_SLAM/data/rgbd_dataset_freiburg3_sitting_xyz/groundtruth.txt' -p --plot_mode=xz

evo_ape tum '/home/chris/GAN_SLAM/data/rgbd_dataset_freiburg3_sitting_xyz/groundtruth.txt' '/home/chris/GAN_SLAM/ORB_SLAM2/CameraTrajectory.txt' -va --plot --plot_mode xz --save_results results/ORB.zip

evo_res results/*.zip -p --save_table results/table.csv

