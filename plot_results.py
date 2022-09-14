from common.config import argparser
from common.read_labeled_floor_plan import Get_BaseColor,Label_gridFloorPlan,get_grid_width
from common.engine import get_all_matrix, ILP_solver, A_star_simulation
from common.A_star import Env
from common.plot_utls import result_plot
import pickle,os
import matplotlib.pyplot as plt
import numpy as np
from Get_sensor_placement import get_fp_grid

def plot_figure2(args):
    color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[255,255,255],4:[255,255,255],5:[255,255,255], 6:[128,128,128], 7:[0,0,204]}
    fp_grid, fp_grid_gt = get_fp_grid(args)

    fp_env = Env(fp_grid)
    # 
    # (s_start, s_goal) = fp_env.get_start_end_point()
    doorway_penalty = args.doorway_penalty
    wall_penalty = args.wall_penalty
    rplot = result_plot(fp_grid,(35,31),(30,160),"euclidean",fp_env,color_map)
    rplot.plot_fp()
    rplot.add_path(doorway_penalty,wall_penalty,[0,0,255])
    rplot.add_s_start_goal_plot()

    fp_env.add_random_obs(args.random_obstacle_dense)
    rplot.reload_fp_grid(fp_env)
    rplot.re_initialize()
    rplot.add_path(doorway_penalty,wall_penalty,[255,155,0])
    rplot.plot_show('Figure2.jpg')

def plot_figure3(args):
    color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[0,255,0],4:[255,255,255],5:[255,255,255], 6:[128,128,128], 7:[0,0,204]}
    fp_grid, fp_grid_gt = get_fp_grid(args)

    fp_env = Env(fp_grid)
    fp_env.add_random_obs(args.random_obstacle_dense)
    # (s_start, s_goal) = fp_env.get_start_end_point()

    rplot = result_plot(fp_grid,(27,51),(10,150),"euclidean",fp_env,color_map)
    rplot.plot_fp()
    rplot.add_path(0,20, [0,0,255])
    rplot.add_s_start_goal_plot()
    # rplot.plot_show()
    # rplot.change_s_start_goal((44,117),(8,150))
    rplot.re_initialize()
    rplot.add_path(100,20, [255,155,0])
    # rplot.add_s_start_goal_plot()
    rplot.plot_show('Figure3.jpg')

def plot_figure4(args):
    color_map = {0:[255,255,255],1:[0,0,0],2:[0,0,0],3:[255,255,255],4:[255,255,255],5:[255,255,255], 6:[128,128,128], 7:[0,0,204]}
    fp_grid, fp_grid_gt = get_fp_grid(args)

    fp_env = Env(fp_grid)
    # fp_env.add_random_obs(args.random_obstacle_dense)
    # 
    # (s_start, s_goal) = fp_env.get_start_end_point()
    doorway_penalty = args.doorway_penalty
    wall_penalty = args.wall_penalty
    rplot = result_plot(fp_grid,(46,46),(46,178),"euclidean",fp_env,color_map)
    rplot.plot_fp()
    rplot.add_path(doorway_penalty,0,[0,0,255])
    

    doorway_penalty = args.doorway_penalty
    rplot.reload_fp_grid(fp_env)
    rplot.re_initialize()
    rplot.add_path(doorway_penalty,1,[255,155,0])
    rplot.add_s_start_goal_plot()
    rplot.plot_show('Figure4.jpg')

if __name__ == '__main__':
    args = argparser.parse_args()
    # plot_figure2(args)
    # plot_figure3(args)
    plot_figure4(args)