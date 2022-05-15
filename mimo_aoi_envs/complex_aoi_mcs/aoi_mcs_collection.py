from mimo_aoi_envs.complex_aoi_mcs.aoi_mcs_env_setting import Setting
# from mcs_envs.env_setting.env_setting_18 import Setting
from image.mapM import MapM
import os
import copy
from os.path import join as pjoin
import numpy as np
import time
import gym
import cv2
from gym import spaces


def myint(a):
    # return int(np.ceil(a))
    return int(np.floor(a))


class Env(gym.Env):
    def __init__(self):
        # self.tr = tracker.SummaryTracker()
        self.sg = Setting()

        # basis
        self.mapx = 16
        self.mapy = 16
        self.map = MapM()  # [80,80]
        self.image_data = None
        self.image_uav = None
        self.image_aoi = None

        # [[3,80,80]]
        # Box用于实现连续数据构成的空间，其中包含两组参数：空间内数据范围（上限和下限），以及空间维度的大小
        self.channel = 3  # 3
        self.num_action = 2
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.channel, self.map.width, self.map.height))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.sg.V['NUM_UAV'] * self.num_action,))

        # num of uavs
        self.num_uav = self.sg.V['NUM_UAV']
        self.crange = self.sg.V['RANGE']  # 1.1
        self.maxdistance = self.sg.V['MAXDISTANCE']  # 1.0
        self.epsilon = self.sg.V['EPSILON']

        # mapob [16,16]
        self.map_ob = np.zeros((self.mapx, self.mapy)).astype(np.int8)

        """
        Initial Obstacles
        draw obstacles in mapob[16,16], the obstacle is 1, others is 0
        """
        # obstacles
        self.OB = 1
        obs = self.sg.V['OBSTACLE']
        for i in obs:
            for x in range(i[0], i[0] + i[2], 1):
                for y in range(i[1], i[1] + i[3], 1):
                    self.map_ob[x][y] = self.OB

        """
        Initial POI(data)
        """
        self.PoIs = np.reshape(self.sg.V['DATA'], (-1, 2)).astype(np.float32)

        # POI data Position  [256,2]
        self.data_position = self.PoIs * self.mapx

        """
        Initial image information
        """
        # [80,80]
        self.channel_value = np.zeros((self.map.width, self.map.height)).astype(np.float32)
        self.channel_uav = np.zeros((self.map.width, self.map.height)).astype(np.float32)
        self.channel_aoi = np.zeros((self.map.width, self.map.height)).astype(np.float32)

        for i, position in enumerate(self.data_position):
            self.map.draw_point(position[0], position[1], 0, self.channel_value)
            self.map.draw_point(position[0], position[1], 1 / self.sg.V["NUM_TIMESLOTS"], self.channel_aoi)

        self.map.draw_wall(self.channel_uav)
        for obstacle in self.sg.V['OBSTACLE']:
            self.map.draw_obstacle(obstacle[0], obstacle[1], obstacle[2], obstacle[3], self.channel_uav)
        for i_n in range(self.num_uav):
            self.map.draw_UAV(self.sg.V['INIT_POSITION'][0], self.sg.V['INIT_POSITION'][1], 1., self.channel_uav)

    def reset(self):
        # initialize data map
        self.map_value = np.zeros(self.PoIs.shape[0], )
        self.map_aoi = np.zeros(self.PoIs.shape[0], )

        # ----
        # initialize state(get POI) and positions of uavs
        self.uav_position = [list(self.sg.V['INIT_POSITION']) for _ in range(self.num_uav)]

        self.uav_trace = [[] for _ in range(self.num_uav)]

        # initialize indicators
        self.data_collection = np.zeros(self.num_uav).astype(np.float32)
        self.energy_consumption = np.zeros(self.num_uav).astype(np.float32)

        # walls
        self.walls = np.zeros(self.num_uav).astype(np.int16)

        # initialize images
        self.state = self.__init_image()
        return self.__get_state()

    def __get_state(self):
        return copy.deepcopy(self.state)

    def __init_image(self):
        self.image_data = copy.copy(self.channel_value)
        self.image_aoi = copy.copy(self.channel_aoi)
        self.image_uav = copy.copy(self.channel_uav)
        # ----
        image = np.zeros((self.channel, self.map.width, self.map.height)).astype(np.float32)
        for width in range(image.shape[0]):
            for height in range(image.shape[1]):
                # god view
                image[0][width][height] = self.image_data[width][height]
                image[1][width][height] = self.image_uav[width][height]
                image[2][width][height] = self.image_aoi[width][height]
        state = image
        return state

    def __draw_image(self, clear_uav, update_poi, update_aoi):
        # value
        for i, value in update_poi:
            self.map.draw_point(self.data_position[i][0], self.data_position[i][1], value / self.sg.V["MAXIMUM_VALUE"],
                                self.state[0, :, :])

        # uav
        for n in range(self.num_uav):
            self.map.clear_uav(clear_uav[n][0], clear_uav[n][1], self.state[1, :, :])
            self.map.draw_UAV(self.uav_position[n][0], self.uav_position[n][1], 1, self.state[1, :, :])
        self.map.draw_wall(self.state[1, :, :])
        for obstacle in self.sg.V['OBSTACLE']:
            self.map.draw_obstacle(obstacle[0], obstacle[1], obstacle[2], obstacle[3], self.state[1, :, :])

        # aoi
        for i, value in update_aoi:
            self.map.draw_point(self.data_position[i][0], self.data_position[i][1], value / 'NUM_TIMESLOTS',
                                self.state[2, :, :])

    def consume_energy(self, distance):
        # configs
        P0 = 99.66  # blade profile power, W
        P1 = 120.16  # derived power, W
        Vt = 25  # velocity of the UAV,m/s
        U_tips = 120  # tip speed of the rotor blade of the UAV,m/s
        v0 = 0.002  # the mean rotor induced velocity in the hovering state,m/s
        d0 = 0.48  # fuselage drag ratio
        rho = 1.225  # density of air,kg/m^3
        s0 = 0.0001  # the rotor solidity
        A = 0.5  # the area of the rotor disk,s^2

        move_time = distance / Vt
        hover_time = self.sg.V['TIME_SLOT'] - move_time

        Power_flying = P0 * (1 + 3 * Vt ** 2 / U_tips ** 2) + \
                       P1 * np.sqrt((np.sqrt(1 + Vt ** 4 / (4 * v0 ** 4)) - Vt ** 2 / (2 * v0 ** 2))) + \
                       0.5 * d0 * rho * s0 * A * Vt ** 3
        Power_hovering = P0 + P1

        return move_time * Power_flying + hover_time * Power_hovering

    def judge_obstacle(self, next_pos, cur_pos):
        if 0 <= next_pos[0] < (self.mapx - 0.1) and 0 <= next_pos[1] < (self.mapy - 0.1):
            dx = next_pos[0] - cur_pos[0]
            dy = next_pos[1] - cur_pos[1]
            acc_range = 100
            for i in range(0, acc_range + 1):
                tmp_pos_x = myint(cur_pos[0] + i * dx / acc_range)
                tmp_pos_y = myint(cur_pos[1] + i * dy / acc_range)
                if self.map_ob[tmp_pos_x][tmp_pos_y] == self.OB:
                    if self.sg.V["DEBUG_MODE"]:
                        print("!!!collision!!!")
                    return True

            return False
        else:
            if self.sg.V["DEBUG_MODE"]:
                print("!!!collision!!!")
            return True



    def data_collection_amount(self):
        return np.sum(self.map_value)

    def jain_fairness(self):
        collection = self.map_value
        for index, i in enumerate(collection):
            collection[index] = i / self.sg.V["MAXIMUM_VALUE"]
        square_of_sum = np.square(np.sum(collection))
        sum_of_square = np.sum(np.square(collection))
        if sum_of_square==0:
            return 0
        else:
            return square_of_sum / sum_of_square / float(len(collection))


    def energy_consumption_per_uav(self):
        return np.mean(self.energy_consumption)

    def age_of_information_per_poi(self):
        return np.mean(self.map_aoi)

    def step(self, actions):
        action = copy.deepcopy(actions.reshape(self.sg.V['NUM_UAV'], self.sg.V['NUM_ACTION']))

        uav_efficiency_list = [0] * self.num_uav

        update_value = []  # Updated PoI collected data
        update_aoi = []  # Updated PoI access times
        clear_uav = copy.copy(self.uav_position)
        new_uav_positions = []

        # update positions of UAVs
        for n in range(self.num_uav):
            self.uav_trace[n].append(self.uav_position[n])

            # distance is from action(x,y), which is a kind of offset,[minaction,maxaction]
            distance = np.sqrt(np.power(action[n][0], 2) + np.power(action[n][1], 2))

            uav_collected_data = 0.0
            uav_penalty = 0.0

            new_x = self.uav_position[n][1] + action[n][0]
            new_y = self.uav_position[n][2] + action[n][1]
            tmp_new_position = [new_x, new_y]

            # if reach OB or WALL, give negative reward, save original positon
            if self.judge_obstacle(tmp_new_position, self.uav_position[n]):
                new_uav_positions.append([self.uav_position[n][0], self.uav_position[n][1]])
                uav_penalty += self.sg.V['WALL_PENALTY']
                self.walls[n] += 1
            else:
                new_uav_positions.append([new_x, new_y])

            # calculate distances between UAV and data points
            _pos = np.repeat([new_uav_positions[-1][1:]], [self.data_position.shape[0]], axis=0)  # just repeat(On)  NB!
            _minus = self.data_position - _pos
            _power = np.power(_minus, 2)
            _dis = np.sum(_power, axis=1)
            for index, dis in enumerate(_dis):
                # sensing PoI(crange=1.1)
                if np.sqrt(dis) <= self.crange:
                    self.map_aoi[index] = 1  # aoi
                    update_aoi.append([index, self.map_aoi[index]])

                    # TODO
                    uav_collected_data += self.initial_value[index] * self.cspeed
                    self.map_value[index] += uav_collected_data
                    update_value.append([index, self.map_value[index]])
                else:
                    self.map_aoi[index] += 1

            # update info (collected data)
            # use energy to get POI(consume energy of UAVs, per alpha 1.0 default)
            uav_propulsion_energy = self.consume_energy(distance)
            uav_efficiency_list[n] = uav_collected_data / uav_propulsion_energy
            self.data_collection[n] += uav_collected_data
            self.energy_consumption[n] += uav_propulsion_energy

        value_fairness = self.jain_fairness()
        mean_aoi = self.age_of_information_per_poi()

        reward = value_fairness * uav_efficiency_list / mean_aoi

        # TODO:放大reward? maybe reward clipping, maybe pop-art
        # self.reward = list(np.clip(np.array(reward) / normalize, -2., 2.))
        self.reward = list(np.array(uav_efficiency_list))

        self.uav_position = new_uav_positions
        self.__draw_image(clear_uav, update_value, update_aoi)
        state = self.__get_state()
        done = False

        info = {"performance_info": {
            "data_collection": 1.0 - self.leftrewards,
            "normal_fairness": self.jain_fairness,
            "use_energy": np.sum(self.energy_consumption_per_uav),
            "collision": np.sum(self.walls),
            "efficiency": self.efficiency,
        },
            "log_info": {
                "mapx": self.mapx,
                "mapy": self.mapy,
                "num_uav": self.num_uav,
                "trace": self.uav_trace,
                "energytrace": self.energytrace,
                "crange": self.crange,
                "fills": self.fills,
                "datas": self.data_position,
                "DATAs": self.PoIs,
                "mapob": self.map_ob,
                "efficiency": self.efficiency,
                "data_collection": 1.0 - self.leftrewards,
            }
        }

        return state, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError

    # TODO:MAYBE NOT USEFUL NOW!!!
    @property
    def leftrewards(self):
        return np.sum(self.map_value) / self.totaldata

    @property
    def efficiency(self):
        return np.sum(self.data_collection / self.totaldata) * self.collection_fairness / (
            np.sum(self.energy_consumption_per_uav))

    @property
    def energy_consumption_per_uav(self):
        tmp = list(np.array(self.energy_consumption) / (self.max_energy_array))
        # for i in range(len(tmp)):
        #     if tmp[i] > 1.0:
        #         tmp[i] = 1.0

        return tmp

    @property
    def jain_fairness(self):
        collection = self.map_value
        for index, i in enumerate(collection):
            collection[index] = i / self.sg.V["MAXIMUM_VALUE"]
        square_of_sum = np.square(np.sum(collection))
        sum_of_square = np.sum(np.square(collection))
        fairness = square_of_sum / sum_of_square / float(len(collection))
        return fairness
