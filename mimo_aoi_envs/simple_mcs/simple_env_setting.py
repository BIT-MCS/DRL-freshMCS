from mimo_aoi_envs.real_aoi.poi_distribution import *

class Setting(object):
    def __init__(self):
        self.V = {
            'MAP_X': 16,
            'MAP_Y': 16,
            'DATA': PoI_256,

            'OBSTACLE': [
                [0, 3, 1, 1],
                [2, 9, 2, 1],
                [1, 3, 1, 2],
                [2, 15, 2, 1],
                [2, 0, 1, 1],
                [4, 4, 1, 1],
                [5, 4, 1, 3],
                [5, 11, 1, 3],
                [10, 0, 3, 1],
                [10, 1, 1, 1],
                [10, 5, 1, 3],
                [8, 10, 3, 1],
                [9, 15, 1, 1],
                [13, 6, 1, 2],
                [13, 13, 1, 2],
                [12, 15, 4, 1],
                [15, 10, 1, 1]
            ],

            'NUM_UAV': 2,  # todo:试试多无人机
            'INIT_POSITION': (8, 8),
            'GRID_LENGTH': 100,  # m

            'NUM_ANTENNAS_PER_UAV': 10,  # 无人机天线根数
            'POI_STATUS_UPDATE_SIZE': 5,  # Mbit AoI重置要发送的包大小
            'VELOCITY': 25,  # m/s   # 无人机速度
            'TIME_SLOT': 20,  # second   # timeslot大小
            "ENERGY_FACTOR": 0,  # TODO:电量占reward的权重，目前非常tricky,0最好学，0.1以下值得尝试

            'WALL_PENALTY': 0.,  # -2
            "DEBUG_MODE": False,
            "TESTING_MODE": True,  # TODO：train的时候是false，test的时候是true，这玩意太占内存
        }
