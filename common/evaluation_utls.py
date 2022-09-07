from array import array
import numpy as np
from sklearn.decomposition import PCA
import math

def Get_people_positions(file_name):
    """read the position of people in unity

        Args: 
            file_name: strings, the file name of the people position 

        Returns: 
            people_position_list: list[(x,y)], the list of position of one people
            people_position_list_int: list[(x_int, y_int)], the list of position of one people,
    """
    people_position_list = []
    people_position_list_int = []
    with open(file_name, 'r') as f:
        i = 0
        frame = 0
        while True:
            i = i+1
            line = f.readline().split()
            if not line:
                line = f.readline().split()
                break
            
            if line[0] == "Position":
                x = float(line[1])
                y = float(line[2])
                people_position_list.append((x,y))
                people_position_list_int.append((int(x),int(y)))
    return people_position_list, people_position_list_int


def Get_fp_zone_grids(file_name):
    """read zone ground truth from files:

        Args:
            file_name: strings, the name of file contains the zone ground truth
        
        Returns:
            zone_grids: numpy.array(n,2), the n grids position belongs to zone i.
    """
    with open(file_name, 'r') as f:
        num_grids = 0

        line = f.readline().split(',')

        if line[0] == 'numZoneGrids':
            num_grids = int(line[1])
        else:
            print("error format")
        zone_grids = np.loadtxt(f,delimiter = ',')
    
    return zone_grids    


def Get_sensor_placement(file_name):
    """read sensor placement from the files:

        Args:
            file_name: strings, the name of file contains the sensor placement
        
        Returns:
            sensor_placement: np.array(n,2), the sensor placement position
    """
    with open(file_name, 'r') as f:
        num_grids = 0

        line = f.readline().split(',')

        if line[0] == 'sensorNum':
            num_grids = int(line[1])
        else:
            print("error format")
        sensor_place = np.loadtxt(f,delimiter = ',')
    return sensor_place    


def Is_sensor_covered(sensor_place, position, sensor_radius) -> bool:
    """ get wether this position is covered by one of the sensor:

        Args:
            sensor_placement: np.array(n,2), 
            position: (x_int,y_int)
            sensor_radius: float, the radius of the sensor coverage
        
        Returns:
            True or False
    """
    cover_range = sensor_radius
    if len(sensor_place.shape) ==2:
        for i in range(len(sensor_place)):
            if abs(position[0]-sensor_place[i][0])<cover_range and abs(position[1]-sensor_place[i][1])<cover_range:
                return True
        return False
    else:
        if abs(position[0]-sensor_place[0])<cover_range and abs(position[1]-sensor_place[1])<cover_range:
            return True
        return False

def read_b_mid(file_name) -> array:
    """read mid point of boundaries from the files:

        Args:
            file_name: strings, the name of file contains the sensor placement
        
        Returns:
            b_mid: np.array(n,2), the sensor placement position
    """
    with open(file_name, 'r') as f:
        num_grids = 0

        line = f.readline().split(',')

        if line[0] == 'numBound':
            num_grids = int(line[1])
        else:
            print("error format")
        b_mid = np.loadtxt(f,delimiter = ',')
    return b_mid  

#需要重写，感觉有问题
def compute_direction(people_detected, i, j, fOrb):
    """ compute the moving direction through PCA
    
        Args: 
            people_detected, list[(x1,y1),(x2,y2),(x3,y3)], ...
            i: int, which person
            j: int, which time point
            fOrb: forward or backward, char, 'f' or 'b'
        
        Returns:
            speed_vector: [dx,dy]
    """
    if fOrb == 'b':
        for idx_t in range(10):
            if people_detected[j-idx_t][i] != (1000, 1000) and people_detected[j-idx_t-1][i] == (1000, 1000):
                break
        unzero_num = idx_t
        
        position_matrix = np.zeros((unzero_num,3))
        
        if unzero_num < 2:
            speed_vector = np.array([0,0,0])
            return speed_vector, position_matrix
        
        if unzero_num == 2:
            x =  people_detected[j][i][0] - people_detected[j-1][i][0]
            y = people_detected[j][i][1] - people_detected[j-1][i][1]
            speed_vector = np.array([x,y,1])
            return speed_vector, position_matrix
            
        
        for i_t in range(unzero_num):
            position_matrix[i_t,0] = people_detected[j-i_t][i][0]
            position_matrix[i_t,1] = people_detected[j-idx_t][i][1]
            position_matrix[i_t,2] = -i_t
        
    elif fOrb == 'f':
        for idx_t in range(10):
            if people_detected[j+idx_t][i] != (1000, 1000) and people_detected[j+idx_t+1][i] == (1000, 1000):
                break
        unzero_num = idx_t
        
        position_matrix = np.zeros((unzero_num,3))
        for i_t in range(unzero_num):
            position_matrix[i_t,0] = people_detected[j+i_t][i][0]
            position_matrix[i_t,1] = people_detected[j+idx_t][i][1]
            position_matrix[i_t,2] = i_t
    
        if unzero_num < 2:
            speed_vector = np.array([0,0,0])
            return speed_vector, position_matrix
        
        if unzero_num == 2:
            x =  people_detected[j+1][i][0] - people_detected[j][i][0]
            y = people_detected[j+1][i][1] - people_detected[j][i][1]
            speed_vector = np.array([x,y,1])
            return speed_vector, position_matrix
        
    pca = PCA(n_components=1)
    principalComponents = pca.fit(position_matrix)
    v = pca.components_
    speed_vector = v[0]/v[0,2]
    
    return speed_vector, position_matrix

def distance(p1, p2):
    x = (p1[0]-p2[0])**2
    y = (p1[1]-p2[1])**2
    dist = math.sqrt(x+y)
    return dist


def Is_toward_bound(b_mid, position, direction, cOrg):
    """ Whether the person walks to the boundary or not

        Args:
            b_mid: mid point of the boundarys, narry(b_n,2)
            position: position of the person, narry(2,)
            direction: walking direction of the person, narry(2,)
            cOrg: comming in or go out, 'c'/'g'
        
        Return:
            Bool, bound_id
    """
    for i in range(b_mid.shape[0]):
        b_p = b_mid[i]
        d = distance(b_p, position)
    
        if d < 50:
            x = b_p[0] - position[0]
            y = b_p[1] - position[1]
            if cOrg == 'c':
                if x*direction[0] + y*direction[1] > 0:
                    return True, i
                else:
                    return False,-1
            if cOrg == 'g':
                if x*direction[0] + y*direction[1] < 0:
                    return True, i
                else:
                    return False,-1
    return False,-1

