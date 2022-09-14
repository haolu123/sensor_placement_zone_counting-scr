from common.config import argparser
from common.read_labeled_floor_plan import Get_BaseColor,Label_gridFloorPlan,get_grid_width
from common.engine import get_all_matrix, ILP_solver, A_star_simulation, get_b_mid_d_matrix, ILP_solver_bmd
import pickle,os
import matplotlib.pyplot as plt
import numpy as np

def output_for_unity(args, fp_grid):
    os.makedirs(args.unity_requirement_dir, exist_ok=True)
    output_folder = args.unity_requirement_dir + args.environmnet_id + '/'
    os.makedirs(output_folder, exist_ok=True)

    # save the walls
    walls = np.where(fp_grid == 1)
    wall_list = np.zeros((walls[0].shape[0],2))
    for i in range(walls[0].shape[0]):
        wall_list[i,:] = [walls[0][i], walls[1][i]]
    wall_list = wall_list - np.array(fp_grid.shape)//2
    with open(output_folder + 'fp_grid_walls.txt','w+') as f:
        f.write("numWall,%d" % walls[0].shape[0])
    with open(output_folder + 'fp_grid_walls.txt','ab') as f:
        f.write(b'\n')
        np.savetxt(f, wall_list, fmt='%d', delimiter=',')

    # save the obstacles
    obstacles = np.where(fp_grid == 2)
    obstacle_list = np.zeros((obstacles[0].shape[0],2))
    for i in range(obstacles[0].shape[0]):
        obstacle_list[i,:] = [obstacles[0][i], obstacles[1][i]]
    obstacle_list = obstacle_list - np.array(fp_grid.shape)//2
    with open(output_folder + 'fp_grid_obstacles.txt','w+') as f:
        f.write("numWall,%d" % obstacles[0].shape[0])
    with open(output_folder + 'fp_grid_obstacles.txt','ab') as f:
        f.write(b'\n')
        np.savetxt(f, obstacle_list, fmt='%d', delimiter=',')
    
    # save the area of interest
    AoI = np.where(fp_grid == 4)
    AoI_list = np.zeros((AoI[0].shape[0],2))
    for i in range(AoI[0].shape[0]):
        AoI_list[i,:] = [AoI[0][i], AoI[1][i]]
    AoI_list = AoI_list - np.array(fp_grid.shape)//2
    with open(output_folder + 'fp_grid_interest.txt','w+') as f:
        f.write("numInterest,%d" % AoI[0].shape[0])
    with open(output_folder + 'fp_grid_interest.txt','ab') as f:
        f.write(b'\n')
        np.savetxt(f, AoI_list, fmt='%d', delimiter=',')

def output_gt_for_evaluations(args, fp_grid_groundtruth, b_mid_list):
    fp_grid_groundtruth = fp_grid_groundtruth - 6
    fp_grid_groundtruth[fp_grid_groundtruth < 0] = 0

    os.makedirs(args.unity_data_dir, exist_ok=True)
    output_folder = args.unity_data_dir + args.environmnet_id + '/'
    os.makedirs(output_folder, exist_ok=True)

    # save area of zone_ground_truth
    idx_zone = 1
    i_max = int(fp_grid_groundtruth.max())
    
    for idx_zone in range(1,i_max+1):
        zone_i_where = np.where(fp_grid_groundtruth==idx_zone)
        
        # save ground truth of zone division
        zone_i_list = np.zeros((zone_i_where[0].shape[0],2))
        for i in range(zone_i_where[0].shape[0]):
            zone_i_list[i,:] = [zone_i_where[0][i],zone_i_where[1][i]]
        zone_i_list = zone_i_list - np.array(fp_grid_groundtruth.shape)//2
        with open(output_folder + "fp_grid_zone_%s.txt" %str(idx_zone), 'w+') as f:
            f.write("numZoneGrids,%d" % zone_i_where[0].shape[0])
        with open(output_folder + "fp_grid_zone_%s.txt" %str(idx_zone), "ab") as f:
            f.write(b"\n")
            np.savetxt(f, zone_i_list, fmt='%d',delimiter=',')

    # save number of zone_division
    with open(output_folder + 'zone_num.pickle', 'wb') as f:
        pickle.dump(i_max, f)

    b_mid_list_save = np.zeros((len(b_mid_list),2))
    for i in range(len(b_mid_list)):
        b_mid_list_save[i,:] = [b_mid_list[i][1],b_mid_list[i][0]]

    b_mid_list_save = b_mid_list_save - np.array(fp_grid_groundtruth.shape)//2
    print(b_mid_list_save)
    with open(output_folder + "b_mid.txt", 'w+') as f:
            f.write("numBound,%d" % zone_i_where[0].shape[0])
    with open(output_folder + "b_mid.txt" , "ab") as f:
        f.write(b"\n")
        np.savetxt(f, b_mid_list_save, fmt='%d',delimiter=',')
    return

def save_placement_result(args,sensor_placement,fp_grid, P,b, G):

    os.makedirs(args.unity_data_dir, exist_ok=True)
    output_folder = args.unity_data_dir + args.environmnet_id + '/'
    os.makedirs(output_folder, exist_ok=True)

    # save ground truth of sensor placement
    for sensor_num in range(len(sensor_placement)):
        sensor_i = sensor_placement[sensor_num].reshape(fp_grid.shape)
        sensor_where = np.where(sensor_i!=0)
        sensor_list = np.zeros((sensor_where[0].shape[0],2))
        for i in range(sensor_where[0].shape[0]):
            sensor_list[i,:] = [sensor_where[0][i],sensor_where[1][i]]
        sensor_list = sensor_list - np.array(fp_grid.shape)//2
        
        with open(output_folder + "sensor_placement_%s.txt" %str(sensor_num+1), 'w+') as f:
            f.write("sensorNum,%d" % sensor_where[0].shape[0])
            
        with open(output_folder + "sensor_placement_%s.txt" %str(sensor_num+1), "ab") as f:
            f.write(b"\n")
            np.savetxt(f, sensor_list, fmt='%d',delimiter=',')
    
    # save cover rate
    cover_rate = []
    for i in range(len(sensor_placement)):
        cover =  (P @ np.diag((b.T @ P).flatten()) ).T @ G @ sensor_placement[i]
        cover_rate_i = np.sum(cover!=0)/np.sum(b.T @ P!=0)
        
        cover_rate.append(cover_rate_i)
    
    print(cover_rate)
    with open(output_folder + "cover_rate.pickle", 'wb') as f:
        pickle.dump(cover_rate, f)
    
    return

def get_fp_grid(args, fp_zone_fn = 'fp1_zone.png', fp_code_fn='fp1_code.png', baseColor_fn = 'baseColor.png'):
    base_color_fig_name = args.data_dir + baseColor_fn
    floor_plan_zone_fig_name = args.data_dir + fp_zone_fn
    floor_plan_code_fig_name = args.data_dir + fp_code_fn

    room_width = args.room_width
    room_length = args.room_length
    grid_width = get_grid_width(room_width, room_length)

    # get base color
    flag = os.path.exists(args.temp_result_dir+'baseColor.pickle')
    if args.use_saved_base_color and os.path.exists(args.temp_result_dir+'baseColor.pickle'):
        with open(args.temp_result_dir+'baseColor.pickle', 'rb') as f:
            baseColor = pickle.load(f)
    else:
        baseColor = Get_BaseColor(base_color_fig_name, args.temp_result_dir + 'baseColor.pickle', 16)
    
    # get fp_grid
    flag = os.path.exists(args.temp_result_dir + 'fp_grid.pickle')
    if args.use_save_temp_result and os.path.exists(args.temp_result_dir + 'fp_grid.pickle'):
        with open(args.temp_result_dir + 'fp_grid.pickle','rb') as f:
            fp_grid = pickle.load(f)
        with open(args.temp_result_dir + 'fp_grid_groundtruth.pickle','rb') as f:
            fp_grid_gt = pickle.load(f)
    else:
        fp_grid, fp_grid_gt =  Label_gridFloorPlan(floor_plan_code_fig_name, floor_plan_zone_fig_name, room_width, room_length, grid_width, args.temp_result_dir, baseColor)
    
    return fp_grid, fp_grid_gt

def run ():
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
    if args.use_save_temp_result and os.path.exists(tempdir + 'fp_grid.pickle'):
        with open(args.temp_result_dir + 'fp_grid.pickle','rb') as f:
            fp_grid = pickle.load(f)
        with open(args.temp_result_dir + 'fp_grid_groundtruth.pickle','rb') as f:
            fp_grid_gt = pickle.load(f)
    else:
        fp_grid, fp_grid_gt =  Label_gridFloorPlan(floor_plan_code_fig_name, floor_plan_zone_fig_name, room_width, room_length, grid_width, tempdir, baseColor)
    
    # get simulation results
    flag = os.path.exists(tempdir + 'path_all.pickle')
    if args.use_save_temp_result and os.path.exists(tempdir + 'path_all.pickle'):
        with open(tempdir + 'path_all.pickle', 'rb') as f:
            path_all = pickle.load(f)
        with open(tempdir+'bound_slices.pickle', 'rb') as f:
            bound_slices = pickle.load(f)
        with open(tempdir + 'b_mid_list.pickle', 'rb') as f:
            b_mid_list = pickle.load(f)
    else:
        path_all, bound_slices,b_mid_list = A_star_simulation(args, fp_grid, grid_width, range_b=200)
        with open(tempdir + 'b_mid_list.pickle', 'wb') as f:
            pickle.dump(b_mid_list,f)
    # print(b_mid_list)
    # print(fp_grid.shape)
    # print(bound_slices)

    # get constant matrixes for ILP
    flag = os.path.exists(tempdir + 'b_matrix_2.pickle')
    if args.use_save_temp_result and os.path.exists(tempdir + 'b_matrix_2.pickle'):
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
        b, P, G, b_mid, P_sliced = get_all_matrix(args, fp_grid, bound_slices, path_all, grid_width, b_mid_list)

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
    bmd = get_b_mid_d_matrix(b_mid_list, fp_grid)
    sensor_placement = ILP_solver_bmd(args, 0.00001,0.01, n, n_p, G, P_matrix, b_mid, b,bmd, f_name)

    #sensor_placement = ILP_solver(args, n, n_p, G, P_matrix, b_mid, b, f_name)
    for i in sensor_placement:
        plt.imshow(10*i.reshape(fp_grid.shape)+fp_grid)
        plt.show()
    
    os.makedirs(args.result_dir, exist_ok=True)
    with open(args.result_dir+'sensor_placement.pickle','wb') as f:
        pickle.dump(sensor_placement,f)

    output_for_unity(args, fp_grid)
    print(b_mid_list)
    print(fp_grid.shape)
    output_gt_for_evaluations(args, fp_grid_gt, b_mid_list)
    save_placement_result(args, sensor_placement,fp_grid, P,b, G)

if __name__ == '__main__':
    run()
    # args = argparser.parse_args()
    # tempdir  = args.temp_result_dir
    # with open(tempdir + 'b_mid_list.pickle', 'rb') as f:
    #     b_mid_list = pickle.load(f)
    # with open(args.temp_result_dir + 'fp_grid.pickle','rb') as f:
    #     fp_grid = pickle.load(f)
    
    # bmd = get_b_mid_d_matrix(b_mid_list, fp_grid)