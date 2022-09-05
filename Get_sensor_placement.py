from common.config import argparser
from common.read_labeled_floor_plan import Get_BaseColor,Label_gridFloorPlan,get_grid_width
from common.engine import get_all_matrix, ILP_solver, A_star_simulation
import pickle,os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = argparser.parse_args()
    data_dir = args.data_dir
    tempdir  = args.temp_result_dir
    base_color_fig_name = data_dir + 'baseColor.png'
    floor_plan_zone_fig_name = data_dir + 'fp1_zone.png'
    floor_plan_code_fig_name = data_dir + 'fp1_code.png'

    room_width = args.room_width
    room_length = args.room_length
    grid_width = get_grid_width(room_width, room_length)

    # get base color
    flag = os.path.exists(tempdir+'baseColor.pickle')
    if args.use_saved_base_color and os.path.exists(tempdir+'baseColor.pickle'):
        with open(tempdir+'baseColor.pickle', 'rb') as f:
            baseColor = pickle.load(f)
    else:
        baseColor = Get_BaseColor(base_color_fig_name, tempdir + 'baseColor.pickle', 16)
    
    # get fp_grid
    flag = os.path.exists(tempdir + 'fp_grid.pickle')
    if args.save_temp_result and os.path.exists(tempdir + 'fp_grid.pickle'):
        with open(args.temp_result_dir + 'fp_grid.pickle','rb') as f:
            fp_grid = pickle.load(f)
        # with open(args.temp_result_dir + 'fp_grid_groundtruth.pickle','rb') as f:
        #     fp_grid_zone = pickle.load(f)
    else:
        fp_grid, fp_grid_gt =  Label_gridFloorPlan(floor_plan_code_fig_name, floor_plan_zone_fig_name, room_width, room_length, grid_width, tempdir, baseColor)
    
    # get simulation results
    flag = os.path.exists(tempdir + 'path_all.pickle')
    if args.save_temp_result and os.path.exists(tempdir + 'path_all.pickle'):
        with open(tempdir + 'path_all.pickle', 'rb') as f:
            path_all = pickle.load(f)
        with open(tempdir+'bound_slices.pickle', 'rb') as f:
            bound_slices = pickle.load(f)
    else:
        path_all, bound_slices = A_star_simulation(args, fp_grid, grid_width)

    # get constant matrixes for ILP
    flag = os.path.exists(tempdir + 'b_matrix_2.pickle')
    if args.save_temp_result and os.path.exists(tempdir + 'b_matrix_2.pickle'):
         # save b
        with open(args.temp_result_dir + 'b_matrix_2.pickle','rb') as file:
            b = pickle.load(file)
        # save P
        with open(args.temp_result_dir + 'P_matrix_2.pickle','rb') as file:
            P = pickle.load(file)
        # save G
        with open(args.temp_result_dir + 'G_matrix_2.pickle','rb') as file:
            G = pickle.load(file)
        # save b_mid
        with open(args.temp_result_dir + 'b_mid_matrix_2.pickle','rb') as file:
            b_mid = pickle.load(file)
        # save P_sliced
        with open(args.temp_result_dir + 'P_sliced.pickle','rb') as file:
            P_sliced = pickle.load(file)
    else:
        b, P, G, b_mid, P_sliced = get_all_matrix(args, fp_grid, bound_slices, path_all, grid_width)

    # ILP
    if args.slice_path:
        P_matrix = P_sliced
        f_name = 'sensor_placement_result_sliced'
    else:
        P_matrix = P
        f_name = 'sensor_placement_result_no_sliced'
    n = fp_grid.shape[0] * fp_grid.shape[1]
    n_p = P_matrix.shape[1]
    # print(n)
    # print(n_p)
    # print(G.shape)
    # print(P_matrix)
    # print(b_mid)
    # print(b)
    # print(f_name)
    sesor_placement = ILP_solver(args, n, n_p, G, P_matrix, b_mid, b, f_name)
    for i in sesor_placement:
        plt.imshow(10*i.reshape(fp_grid.shape)+fp_grid)
        plt.show()