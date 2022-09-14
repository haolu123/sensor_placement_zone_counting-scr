import argparse
import os

argparser = argparse.ArgumentParser(description='sensor placement for zone counting with ToF sensors')

# folder directions
argparser.add_argument('--temp_result_dir', type=str, default='./temp_result/')
argparser.add_argument('--data_dir', type=str, default='./data/')
argparser.add_argument('--result_dir', type=str, default='./result/', help='the place to save the result')
# room parameters
argparser.add_argument('--room_width', type=float, default=500*1.1, help='the real width of the room (cm)')
argparser.add_argument('--room_length', type=float, default=2200*1.1, help='the real length of the room (cm)')

# A* parameters
argparser.add_argument('--random_obstacle_dense', type=float, default=0.2, help='the dense of random obstacles in blank space')
argparser.add_argument('--doorway_penalty', type=float, default=50, help='door way penalty')
argparser.add_argument('--wall_penalty', type=float, default=20, help='penalty close to the wall'  )
argparser.add_argument('--path_num', type=float, default=1000, help='how many paths will simulate with A* algorithm')

# control flags
argparser.add_argument('--use_saved_base_color', action='store_false', help='Use the previous base color or not?')
argparser.add_argument('--save_hotmap', action='store_false', help='Save hotmap or not?')
argparser.add_argument('--save_temp_result', action='store_false', help='Save temp_result or not?')
argparser.add_argument('--save_unity3d_result', action='store_false', help='Save result for unity3d or not?')
argparser.add_argument('--slice_path', action='store_false', help="where sliced the pathes around the doorways or not")
argparser.add_argument('--use_save_temp_result', action='store_false', help='Use the saved temp_result or not?')

# unity simulation parameters
argparser.add_argument('--human_num', type=int, default=3, help="the number of people used in the simulation of Unity")
argparser.add_argument('--frame_num', type=int, default=200000, help="How many frame simulated by unity")
argparser.add_argument('--unity_data_dir', type=str, default=r"unity3d_outputs/", help="where are the simulation result saved")
argparser.add_argument('--unity_requirement_dir', type=str, default=r'unity3d_req/', help='where to save the files unity used')
argparser.add_argument('--environmnet_id', type=str, default='office1', help='Which environment are we working on')
# 






# argparser.add_argument('--train_file', type=str, default='../../social-interactions/data/split/train.list', help='Train list')
# argparser.add_argument('--val_file', type=str, default='../../social-interactions/data/split/val.list', help='Validation list')
# # argparser.add_argument('--test_file', type=str, default='./test.list', help='Test list')
# argparser.add_argument('--train_stride', type=int, default=3, help='Train subsampling rate')
# argparser.add_argument('--val_stride', type=int, default=1, help='Validation subsampling rate')
# argparser.add_argument('--test_stride', type=int, default=1, help='Test subsampling rate')
# argparser.add_argument('--epochs', type=int, default=40, help='Maximum epoch')
# argparser.add_argument('--batch_size', type=int, default=400, help='Batch size')
# argparser.add_argument('--num_workers', type=int, default=0, help='Num workers')
# argparser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
# argparser.add_argument('--weights', type=list, default=[0.266, 0.734], help='Class weight')
# argparser.add_argument('--eval', action='store_true', help='Running type')
# argparser.add_argument('--model', type=str, default='BaselineLSTM', help='Model architecture')
# argparser.add_argument('--rank', type=int, default=0, help='Rank id')
# argparser.add_argument('--device_id', type=int, default=0, help='Device id')
# argparser.add_argument('--exp_path', type=str, default='evalai_test/output', help='Path to results')
# argparser.add_argument('--checkpoint', type=str, default='evalai_test/best.pth', help='Checkpoint to load')
