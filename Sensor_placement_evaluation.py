import pickle
from common.config import argparser
from common.evaluation_utls import Get_people_positions, Get_fp_zone_grids, Get_sensor_placement, Is_sensor_covered, read_b_mid, compute_direction, distance, Is_toward_bound
import copy 
import numpy as np
import matplotlib.pyplot as plt

def Get_frame_num(args):
    human_num = args.human_num
    frame_num = args.frame_num

    for i in range(human_num):
        file_name = args.unity_data_dir + args.environmnet_id +'/' + "position%d.txt"%(i)
        people_position_list, _ = Get_people_positions(file_name)
        if len(people_position_list) < frame_num:
            frame_num = len(people_position_list)

    return frame_num

def Get_position_gt(args, frame_num):
    people_list = []
    people_list_int = []
    for i in range(args.human_num):
        file_name = args.unity_data_dir + args.environmnet_id + '/' + "position%d.txt"%(i)
        people_position_list, people_position_list_int = Get_people_positions(file_name)
        people_list.append(people_position_list[:frame_num])
        people_list_int.append(people_position_list_int[:frame_num])
    return people_list, people_list_int

def Get_zone_division_gt(args):
    folder_name = args.unity_data_dir + args.environmnet_id + '/'
    # with open(folder_name + 'zone_num.pickle', 'rb') as f:
    #     zone_num = pickle.load(f)
    zone_num = 4

    zone_grids_dict = {}

    for i in range(zone_num):
        file_name = folder_name + "fp_grid_zone_%d.txt" % (i+1)
        zone_grids = Get_fp_zone_grids(file_name)
        for j in range(zone_grids.shape[0]):
            zone_grids_dict[(int(zone_grids[j,0]),int(zone_grids[j,1]))] = i
    return zone_grids_dict, zone_num

def Get_people_zone_label(frame_num, human_num, people_list_int, zone_grids_dict):
    GT_people = []
    for i in range(frame_num):
        temp_GT = []
        for j in range(human_num):
            try:
                temp_GT.append(zone_grids_dict[people_list_int[j][i]])
            except KeyError:
                if people_list_int[j][i][0] < -30 or people_list_int[j][i][1] < -114 or people_list_int[j][i][0]>30 or people_list_int[j][i][1]>115:
                    temp_GT.append(3)
                elif -6< people_list_int[j][i][0] < 17 and -23< people_list_int[j][i][1] < 12:
                    temp_GT.append(1)
                else: 
                    temp_GT.append(-1)
        GT_people.append(temp_GT)
    return GT_people

def Get_zone_count_gt(GT_people, zone_num, human_num, frame_num):
    GT_zone_count = []
    for i in range(frame_num):
        temp_GT_count = np.zeros(zone_num)
        for j in range(human_num):
            if GT_people[i][j] <zone_num and  GT_people[i][j] >0:
                temp_GT_count[GT_people[i][j]] += 1
            else:
                temp_GT_count[GT_people[i-1][j]] += 1
                GT_people[i][j] = GT_people[i-1][j]
        GT_zone_count.append(temp_GT_count)

    GT_count_activation = np.zeros(frame_num)

    temp_position = copy.copy(GT_people[0])
    for i in range(frame_num):
        if GT_people[i] != temp_position:
            GT_count_activation[i] = 1
            temp_position = copy.copy(GT_people[i])
    return GT_zone_count, GT_count_activation

def Get_sensor_placement_pack(args, sensor_num):
    folder_name = args.unity_data_dir + args.environmnet_id + '/'
    file_name = folder_name + "sensor_placement_%d.txt" % sensor_num
    sensor_place = Get_sensor_placement(file_name)
    return sensor_place

def Get_detected_people(sensor_place, frame_num, sensor_radius, human_num, people_list):
    people_detected = []
    for i in range(frame_num):
        temp_detected = []
        for j in range(human_num):
            if Is_sensor_covered(sensor_place, people_list[j][i], sensor_radius):
                temp_detected.append(people_list[j][i])
            else:
                temp_detected.append((1000, 1000))
        people_detected.append(temp_detected)
    return people_detected

def Get_zone_label_for_detected_people(frame_num, human_num, people_detected, zone_grids_dict):
    see_people = []
    for i in range(frame_num):
        temp_see = []
        for j in range(human_num):
            try:
                x = round(people_detected[i][j][0])
                y = round(people_detected[i][j][1])
                # if x<-15 and (y<9 and y>=-2):
                #     temp_see.append(5)
                # else:
                temp_see.append(zone_grids_dict[(x,y)])
            except KeyError:
                temp_see.append(-1)
        see_people.append(temp_see)
    return see_people

def Zone_counting(zone_count_init, see_people_list, zone_num, frame_num, human_num, sensor_num, people_detected, b_mid):
    zone_count = np.zeros(zone_num+1)
    zone_count[:zone_num] = zone_count_init
    see_people = np.array(copy.deepcopy(see_people_list))
    count_activation = np.zeros(frame_num)

    for i in range(human_num):
        now_position = -1
        for j in range(1,frame_num-10):
            # if sensor_num == 6 and i == 0 and (j>37 and j < 40):
            #     print(now_position)
            #     print(see_people[j][i])
            #     print(see_people[j][i] != now_position)
            #     print(now_position == -1)                
                
            if see_people[j][i] != now_position:
                # if double_detect_check != -1:
                #     double_detect_count += 1
                
                if now_position == -1:
                    now_position = see_people[j][i]
                    speed_v,position_matrix = compute_direction(people_detected, i,j, 'f')
                    IsAction,bound_i = Is_toward_bound(b_mid, people_detected[j][i],speed_v,'g')
                    if IsAction:
                        count_activation[j] = 1

                if  see_people[j][i]==-1:
                    now_position = see_people[j][i]
                    speed_v,_ = compute_direction(people_detected, i,j, 'b')
                    IsAction,bound_i = Is_toward_bound(b_mid, people_detected[j][i],speed_v,'c')
                    if IsAction:
                        count_activation[j] = 1

                if see_people[j][i]!=-1 and now_position != -1: 
                    count_activation[j] = 1

                    now_position = see_people[j][i]
                    # if sensor_num == 6 and i == 0 and (j>37 and j < 40):
                    #     print(count_activation[j])
                #print(zone_count)
    return count_activation

def CCR_wcc_compute(window_size, count_act_gt, count_act_est, frame_num):
    """
    CCR_wcc = TP/(TP + TF + FN + FP)

    GT_count_activation
    count_activation
    """
    window_size = 100
    en = []
    N = []
    TP = 0 
    all_event = 0
    for i in range(frame_num):
        min_en = 100
        for j in range(-window_size, window_size+1):
            idx_e = max(0, i+j)
            idx_e = min(frame_num-1, i+j)
            if abs(count_act_est[idx_e] - count_act_gt[i]) < min_en : 
                min_en = abs(count_act_est[idx_e] - count_act_gt[i])
                min_position = idx_e
        
        if count_act_gt[i]!= 0:
            count_act_est[min_position] = 0
            all_event += 1
            if min_en == 0:
                TP += 1
        else:
            if min_en != 0:
                all_event += 1
        N.append(min_position)
        en.append(min_en)

    for i in range(frame_num):
        if N[i] == 0 and count_act_est[i] != 0:
            all_event += 1
            
    print(TP/all_event)
    CCR_wcc = TP/all_event
    return CCR_wcc
def run():
    args = argparser.parse_args()
    human_num = args.human_num
    sensor_radius = 8
    # determine the frame_num
    frame_num = Get_frame_num(args)

    # Get the position ground truth of all persons
    people_list, people_list_int = Get_position_gt(args,frame_num)

    # read ground truth of zone grids. [the ground truth of zone division should get from sensor_labeling part, need change here and the Get_sensor_placement part]
    zone_grids_dict, zone_num = Get_zone_division_gt(args)

    # get the ground truth of which zone the people are.
    GT_people = Get_people_zone_label(frame_num, human_num, people_list_int, zone_grids_dict)

    # get the zone counting ground truth
    GT_zone_count, GT_count_activation = Get_zone_count_gt(GT_people, zone_num, human_num, frame_num)

    CCR_wcc_list = []
    for sensor_num in range(1,7): # will change to for loop later

        # get_sensor_placement: input: number of sensor
        sensor_placement = Get_sensor_placement_pack(args, sensor_num)

        # get_detected people: input: sensor_placement, sensor_radius
        people_detected = Get_detected_people(sensor_placement, frame_num, sensor_radius, human_num, people_list)

        # get detected zone id
        see_people = Get_zone_label_for_detected_people(frame_num, human_num, people_detected, zone_grids_dict)

        # get mid point of boundary
        b_mid = read_b_mid(args.unity_data_dir + args.environmnet_id + '/' + 'b_mid.txt')
        # zone count
        count_activation = Zone_counting(GT_zone_count[0], see_people, zone_num, frame_num, human_num, sensor_num, people_detected, b_mid)

        # CCR_wcc
        CCR_wcc = CCR_wcc_compute(50, GT_count_activation, count_activation, frame_num)
        CCR_wcc_list.append(CCR_wcc)

    x = [i+1 for i in range(6)]
    plt.plot(x, CCR_wcc_list,'bs', label = "classification correct rate")
    plt.xlabel("sensor number")
    plt.ylabel("percentage")
    plt.title("sensor coverage and classification correct rate VS sensor numbers")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    run()