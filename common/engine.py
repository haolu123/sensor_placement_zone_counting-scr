#%%
import pickle
from common.A_star import Env, AStar, Env_sensor_coverage
from common.config import argparser
from common.read_labeled_floor_plan import Get_BaseColor,Label_gridFloorPlan,get_grid_width
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import cvxpy as cp

def A_star_simulation(args, fp_grid, grid_width):

    walk_freq_map = np.zeros(fp_grid.shape)
    # build enviroment for A* algorithm
    fp_env = Env(fp_grid)
    bound_slices = fp_env.get_boundary_areas(grid_width)

    # begin simulation
    path_all = []
    for loop_num in tqdm.tqdm(range(args.path_num)):
        # random select two points as start and end point
        (s_start, s_goal) = fp_env.get_start_end_point()

        # add random small obstacles
        fp_env.add_random_obs(args.random_obstacle_dense)

        # get the shortest path between start point and end point
        astar = AStar(s_start, s_goal, "euclidean", fp_env)
        path, visited = astar.searching(args.doorway_penalty, args.wall_penalty)

        # save shortest paths
        path_all.append(path)

        # mark in the walk frequence map
        if args.save_hotmap:
            for p in path:
                walk_freq_map[p[0],p[1]] += 1

    if args.save_hotmap:
        # plot walk_freq_map
        plt.imshow(walk_freq_map, cmap='hot', interpolation='nearest')
        plt.title("obstacle dense:%.3f, doorway penalty:%d , wall_penalty:%d" % (args.random_obstacle_dense, args.doorway_penalty, args.wall_penalty))
        img_name = "obstacle_dense_" + str(args.random_obstacle_dense) + "doorway_penalty" + str(args.doorway_penalty) + "wall_penalty" + str(args.wall_penalty) +'.png'
        folder_name = args.temp_result_dir+'hotmap/'
        os.makedirs(folder_name, exist_ok=True)
        plt.savefig(folder_name + img_name)
        # plt.show()

        with open(args.temp_result_dir+"walk_freq_map.pickle", 'wb') as f:
            pickle.dump(walk_freq_map,f)

    if args.save_temp_result:
        # save simulated paths
        with open(args.temp_result_dir+'path_all.pickle', 'wb') as f:
            pickle.dump(path_all, f)
        with open(args.temp_result_dir+'bound_slices.pickle', 'wb') as f:
            pickle.dump(bound_slices, f)
    
    if args.save_unity3d_result:
        folder_name_unity ='unity3d_result/'
        os.makedirs(folder_name_unity, exist_ok=True)

        # save simulated information for unity 3d simulation
        fp_grid_unity = copy.deepcopy(fp_grid)
        fp_grid_unity[fp_grid_unity==2] = 1
        
        # save obstacles
        obstacle_where = np.where(fp_grid_unity==1)
        obstacle_list = np.zeros((obstacle_where[0].shape[0],2))
        for i in range(obstacle_where[0].shape[0]):
            obstacle_list[i,:] = [obstacle_where[0][i],obstacle_where[1][i]]
            
        obstacle_list = obstacle_list - np.array(fp_grid_unity.shape)//2
        with open(folder_name_unity + "fp_grid_obstacle.txt", 'w+') as f:
            f.write("numWall,%d" % obstacle_where[0].shape[0])
        with open(folder_name_unity + "fp_grid_obstacle.txt", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, obstacle_list, fmt='%d',delimiter=',')
        
        # with open(r"..\..\Unity3d\AutoFloorPlan\Assets\fp_grid_obstacle.txt", 'w+') as f:
        #     f.write("numWall,%d" % obstacle_where[0].shape[0])
        # with open(r"..\..\Unity3d\AutoFloorPlan\Assets\fp_grid_obstacle.txt", "ab") as f:
        #     f.write(b"\n")
        #     np.savetxt(f, obstacle_list, fmt='%d',delimiter=',')

        # save area of interestings
        interest_where = np.where(fp_grid_unity==4)
        
        area_interest_list = np.zeros((interest_where[0].shape[0],2))
        for i in range(interest_where[0].shape[0]):
            area_interest_list[i,:] = [interest_where[0][i],interest_where[1][i]]
        area_interest_list = area_interest_list - np.array(fp_grid_unity.shape)//2
        with open(folder_name_unity + "fp_grid_interest.txt", 'w+') as f:
            f.write("numInterest,%d" % interest_where[0].shape[0])
        with open(folder_name_unity + "fp_grid_interest.txt", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, area_interest_list, fmt='%d',delimiter=',')
        
        # with open(r"..\..\Unity3d\AutoFloorPlan\Assets\fp_grid_interest.txt", 'w+') as f:
        #     f.write("numInterest,%d" % interest_where[0].shape[0])
        # with open(r"..\..\Unity3d\AutoFloorPlan\Assets\fp_grid_interest.txt", "ab") as f:
        #     f.write(b"\n")
        #     np.savetxt(f, area_interest_list, fmt='%d',delimiter=',')
    return path_all, bound_slices

def get_all_matrix(args, fp_grid, bound_slices, path_all, grid_width):

    print("begin to get all metrix")
    fp_grid_bound = copy.deepcopy(fp_grid)
    fp_grid_bound[fp_grid_bound != 5] = 0
    fp_grid_bound[fp_grid_bound >= 5] = 1
    
    # b: boundary matrix
    print("geting b")
    b = fp_grid_bound.reshape([-1,1])
    
    # get matrix P: paths matrix
    print("geting P")
    x_shape, y_shape = fp_grid.shape
    P = np.zeros((x_shape * y_shape, len(path_all)) )
    
    for i, path in enumerate(path_all):
        path_2d = np.zeros(fp_grid.shape)
        for j, grid in enumerate(path):
            path_2d[grid[0],grid[1]] = 1
        path_col = path_2d.reshape([-1])
        P[:,i] = path_col
        

    
    # get matrix G ( sensor coverage matrix)
    print("geting G")
    fp_grid_cover = copy.deepcopy(fp_grid)
    fp_grid_cover[fp_grid_cover > 2] = 0
    
    x_range = fp_grid_cover.shape[0]
    y_range = fp_grid_cover.shape[1]
    
    G = -np.ones((x_range,y_range,x_range,y_range)).astype(np.int8)
    
    for x_i in range(x_range):
        for x_j in range(y_range):
            # print((x_i,x_j))
            if fp_grid_cover[x_i,x_j] != 0:
                continue
            
            for quad in range(4):
                if quad == 0:
                    sign_i = 1
                    sign_j = 1
                elif quad == 1:
                    sign_i = -1
                    sign_j = 1
                elif quad == 2:
                    sign_i = -1
                    sign_j = -1
                else:
                    sign_i = 1
                    sign_j = -1
                
            # sign_i = -1
            # sign_j = 1
                obs_slop = []
                for i in range(100//grid_width):
                    for j in range(100//grid_width):
                        # if i == 4 and j == 2:
                        #     aaa = 0
                            
                        sc_grid_i = x_i + i*sign_i
                        sc_grid_j = x_j + j*sign_j
                        
                        if sc_grid_i<0 or sc_grid_i>=x_range:
                            continue
                        if sc_grid_j<0 or sc_grid_j>=y_range:
                            continue
                        
                        if fp_grid_cover[sc_grid_i,sc_grid_j] != 0: 
                            # if this grid is obstacle
                            # set G
                            G[x_i,x_j,sc_grid_i,sc_grid_j] = 0
                            
                            # update shaddow
                            obs_r = np.sqrt(i**2 + j**2)
                            if i != 0:
                                obs_a = np.arctan(j/i)
                            else:
                                obs_a = np.pi/2
                            obs_slop.append((obs_r,obs_a))
                        else: 
                            # if this gird is not obstacle
                            
                            # compute the polar coordinate of this grid
                            g_r = np.sqrt(i**2 + j**2)
                            if i!=0:
                                g_a = np.arctan(j/i)
                            else:
                                g_a = np.pi/2
                            # set G[]=1 (can be detected) first then kick off the shaddows.
                            G[x_i,x_j,sc_grid_i,sc_grid_j] = 1
                            # whether this grid in in shaddow
                            for obs_i in obs_slop:
                                if g_r > obs_i[0] and obs_i[1]-0.9/obs_i[0] <= g_a <= obs_i[1] + 0.9/obs_i[0]:
                                    # if it is in shaddow, set G[] = 0
                                    G[x_i,x_j,sc_grid_i,sc_grid_j] = 0
                                    break
                    
    G = G.reshape(x_range*y_range, x_range*y_range).astype(np.int8)
    
    # get b_mid
    bound_mid_point = []
    b_mid = np.zeros(b.shape)
    
    for i in range(len(bound_slices)):
        x_mid = (bound_slices[i][0] + bound_slices[i][2])//2
        y_mid = (bound_slices[i][1] + bound_slices[i][3])//2
        bound_mid_point.append([x_mid,y_mid])
        b_mid[y_mid*fp_grid.shape[1]+x_mid] = 1
    
    
    # chop matrix P with bound_slices
    print("geting chop P")
    P_sliced = np.zeros((P.shape[0], 0))
    b_slices_list = []
    for i in range(len(bound_slices)):
        b_slices_mask = np.zeros((x_range,y_range))
        b_slices_mask[bound_slices[i][1]:bound_slices[i][3], bound_slices[i][0]:bound_slices[i][2]] = 1
        b_slices_mask = np.reshape(b_slices_mask, (1,-1))
        p_temp = copy.deepcopy(P)
        p_temp_sliced = b_slices_mask.T * p_temp
        P_sliced = np.hstack((P_sliced,p_temp_sliced))
        b_slices_list.append(b_slices_mask)
        
    idx = np.argwhere(np.all(P_sliced[..., :] == 0, axis=0))
    P_sliced = np.delete(P_sliced, idx, axis=1)
    
    # for i in range(len(b_slices_list)):
    #     plt.imshow(10*b_slices_list[i].reshape(fp_grid.shape)+fp_grid)
    #     plt.show()
    print("finished get all matrix")

    # save temp_result
    if args.save_temp_result:
        # save b
        with open(args.temp_result_dir + 'b_matrix_2.pickle','wb') as file:
            pickle.dump(b,file)
        # save P
        with open(args.temp_result_dir + 'P_matrix_2.pickle','wb') as file:
            pickle.dump(P,file)
        # save G
        with open(args.temp_result_dir + 'G_matrix_2.pickle','wb') as file:
            pickle.dump(G,file)
        # save b_mid
        with open(args.temp_result_dir + 'b_mid_matrix_2.pickle','wb') as file:
            pickle.dump(b_mid,file)
        # save P_sliced
        with open(args.temp_result_dir + 'P_sliced.pickle','wb') as file:
            pickle.dump(P_sliced,file)

    return b, P, G, b_mid, P_sliced

def ILP_solver(args, n, n_p, G, P, b_mid, b, result_same_name):
    G[G == -1] = 0
    G = G.T

    sensor_placement = []
    x_initial = np.concatenate((b_mid, np.ones((n_p,1))), axis=0)
    m = n+n_p
    x = cp.Variable(m, boolean = True)
    x.value = x_initial.reshape(-1)
    
    gamma = cp.Parameter(nonneg=True)
    gamma_vals = np.array([1,2,3,4,5,6])
    
    # objective weights
    # f = -np.concatenate((-np.ones((1,n)), np.ones((1,n_p))), axis = 1).T
    
    # constrains weights A
    temp_m = (P @ np.diag((b.T @ P).flatten()) ).T @ G
    temp_line_1 = np.concatenate((np.ones((1,n)), np.zeros((1,n_p))),axis = 1)
    temp_line_2 = np.concatenate((-temp_m, np.eye(n_p)),axis = 1)
    # A = np.concatenate((temp_line_1, temp_line_2), axis = 0)
    A = temp_line_2
    
    # sensor number k
    # k = 4
    
    # constrains bias B
    # B = np.concatenate((np.array([k]).reshape((1,1)), np.zeros((1,n_p))),axis=1).T
    B = np.zeros((1,n_p)).T
    
    objective = 0.01* np.concatenate((np.ones((1,n)),np.zeros((1,n_p))), axis = 1) @ x - np.concatenate((np.zeros((1,n)),np.ones((1,n_p))), axis = 1) @ x
    
    constrains = []
    for i in range(len(A)):
        constrains.append(A[i] @ x <= B[i])
    constrains.append(temp_line_1 @ x <= gamma)
    
    
    problem = cp.Problem(cp.Minimize(objective), constrains)
    
    for val in gamma_vals:
        gamma.value = val
        problem.solve(solver = 'CBC',verbose=True, warm_start = False, maximumSeconds=4000)
        sensor_placement.append(x.value[:n])
    
        # plt.imshow(10*x.value[:n].reshape(fp_grid.shape)+fp_grid)
        # plt.pause(0.1)

    if args.save_temp_result:
        with open(args.temp_result_dir + result_same_name, 'wb') as f:
            pickle.dump(sensor_placement,f)
    
    return sensor_placement
#%%
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
#%%