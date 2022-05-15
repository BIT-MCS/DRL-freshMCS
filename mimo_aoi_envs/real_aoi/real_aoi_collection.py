from mimo_aoi_envs.real_aoi.real_aoi_setting import Setting
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
        self.image_uav = None
        self.image_aoi = None
        self.update_poi_list_all = []
        self.update_uav_snr_list = []
        # [[1,80,80]]
        # Box用于实现连续数据构成的空间，其中包含两组参数：空间内数据范围（上限和下限），以及空间维度的大小
        self.channel = 2
        self.num_action = 2
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.channel, self.map.width, self.map.height))
        self.action_space = spaces.Box(low=-2, high=2, shape=(self.sg.V['NUM_UAV'] * self.num_action,))

        # num of uavs
        self.num_uav = self.sg.V['NUM_UAV']

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
        self.channel_uav = np.zeros((self.map.width, self.map.height)).astype(np.float16)
        self.channel_aoi = np.zeros((self.map.width, self.map.height)).astype(np.float16)

        for i, position in enumerate(self.data_position):
            self.map.draw_point(position[0], position[1], 1 / 255, self.channel_aoi)

        self.map.draw_wall(self.channel_uav)
        for obstacle in self.sg.V['OBSTACLE']:
            self.map.draw_obstacle(obstacle[0], obstacle[1], obstacle[2], obstacle[3], self.channel_uav)
        for i_n in range(self.num_uav):
            self.map.draw_UAV(self.sg.V['INIT_POSITION'][0], self.sg.V['INIT_POSITION'][1], 1., self.channel_uav)

    def reset(self):
        # initialize data map
        self.map_aoi = np.ones(self.PoIs.shape[0], ).astype(np.float32)
        self.energy_consumption = np.zeros(self.num_uav).astype(np.float32)
        self.uav_position = [list(self.sg.V['INIT_POSITION']) for _ in range(self.num_uav)]

        self.aoi_list = []
        self.energy_list = []
        self.map_aoi_list = []
        self.episodic_uav_snr_list = []
        self.update_poi_list_all = []
        self.update_uav_snr_list = []

        self.collision = [0] * self.num_uav
        self.uav_trace = [[] for _ in range(self.num_uav)]
        for n in range(self.num_uav):
            self.uav_trace[n].append(self.uav_position[n])

        self.aoi_list.append(np.mean(self.map_aoi))
        self.energy_list.append(np.mean(self.energy_consumption))
        self.map_aoi_list.append(self.map_aoi)

        # initialize images
        self.state = self.__init_image()
        return self.__get_state()

    def __get_state(self):
        return copy.deepcopy(self.state)

    def __init_image(self):
        self.image_aoi = copy.copy(self.channel_aoi)
        self.image_uav = copy.copy(self.channel_uav)
        # ----
        image = np.zeros((self.channel, self.map.width, self.map.height)).astype(np.float16)
        for width in range(image.shape[1]):
            for height in range(image.shape[2]):
                # god view
                image[0][width][height] = self.image_uav[width][height]
                image[1][width][height] = self.image_aoi[width][height]
        state = image
        return state

    def __draw_image(self, clear_uav, update_aoi):
        # uav
        for n in range(self.num_uav):
            self.map.clear_uav(clear_uav[n][0], clear_uav[n][1], self.state[0, :, :])
            self.map.draw_UAV(self.uav_position[n][0], self.uav_position[n][1], 1, self.state[0, :, :])
        self.map.draw_wall(self.state[0, :, :])
        for obstacle in self.sg.V['OBSTACLE']:
            self.map.draw_obstacle(obstacle[0], obstacle[1], obstacle[2], obstacle[3], self.state[0, :, :])

        # aoi
        for i, value in update_aoi:
            self.map.draw_point(self.data_position[i][0], self.data_position[i][1], value / 255,
                                self.state[1, :, :])

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

    def consume_energy(self, fly_time, hover_time, K_t):
        # configs
        Pu = 0.01  # the average transmitted power of each PoI (SN), dB (w)
        P0 = 79.8563  # blade profile power, W
        P1 = 88.6279  # derived power, W
        U_tips = 120  # tip speed of the rotor blade of the UAV,m/s
        v0 = 4.03  # the mean rotor induced velocity in the hovering state,m/s
        d0 = 0.6  # fuselage drag ratio
        rho = 1.225  # density of air,kg/m^3
        s0 = 0.05  # the rotor solidity
        A = 0.503  # the area of the rotor disk, m^2
        Vt = self.sg.V['VELOCITY']  # velocity of the UAV,m/s

        Power_flying = P0 * (1 + 3 * Vt ** 2 / U_tips ** 2) + \
                       P1 * np.sqrt((np.sqrt(1 + Vt ** 4 / (4 * v0 ** 4)) - Vt ** 2 / (2 * v0 ** 2))) + \
                       0.5 * d0 * rho * s0 * A * Vt ** 3

        Power_hovering = P0 + P1 + Pu * K_t

        return fly_time * Power_flying + hover_time * Power_hovering

    def jain_fairness(self):
        collection = self.map_aoi
        for index, i in enumerate(collection):
            collection[index] = i / self.sg.V["MAXIMUM_VALUE"]
        square_of_sum = np.square(np.sum(collection))
        sum_of_square = np.sum(np.square(collection))
        if sum_of_square == 0:
            return 0
        else:
            return square_of_sum / sum_of_square / float(len(collection))

    def judge_status_update(self, square_distance, K_t, hover_time):
        # configs
        B = 0.1  # bind width, MHz
        beta_0 = -60  # the channel power gain at the reference distance(1m), dB
        sigma2 = -104  # noise power, dBm
        alpha = 2
        recian_factor = 0.94
        height = 130  # m
        Pu = 0.01  # the average transmitted power of each PoI (SN), w, 23dbm
        M = self.sg.V["NUM_ANTENNAS_PER_UAV"]
        data_size = self.sg.V['POI_STATUS_UPDATE_SIZE']
        timeslot_size = self.sg.V['TIME_SLOT']

        channel_power_gain = 10 ** (beta_0 / 10) * (square_distance ** 2 + height ** 2) ** (-alpha / 2)
        available_k = M - K_t if K_t >= 2 else M  # ZF/MRC
        SNR = available_k * Pu * channel_power_gain / (10 ** ((sigma2 - 30) / 10))
        achievable_rate = B * np.log2(1 + SNR)

        required_time = data_size / achievable_rate
        if required_time <= hover_time:
            return True, SNR
        else:
            return False, SNR

    def update_poi_status(self, uav_position, update_poi_list, hover_time):

        # calculate distances between UAV and data points
        _pos = np.repeat([uav_position], [self.data_position.shape[0]], axis=0)  # just repeat(On)  NB!
        dis_list = np.sqrt(np.sum(np.power(self.data_position - _pos, 2), axis=1)) * self.sg.V["GRID_LENGTH"]
        sorted_index_list = np.argsort(dis_list)

        M = self.sg.V["NUM_ANTENNAS_PER_UAV"]
        success_k_list = [[] for _ in range(M + 1)]
        success_k_len_list = [0] * M
        success_k_SNR_list = [0] * M
        for tmp_k in range(M):
            if tmp_k == 0:
                continue
            else:
                tmp_SNR_list = []
                for index in sorted_index_list[:tmp_k]:
                    distance = dis_list[index]
                    reset_ok, current_SNR = self.judge_status_update(distance, tmp_k, hover_time)
                    if reset_ok is True:
                        success_k_list[tmp_k].append(index)
                        tmp_SNR_list.append(current_SNR)

            success_k_len_list[tmp_k] = len(success_k_list[tmp_k])
            if len(tmp_SNR_list) != 0:
                success_k_SNR_list[tmp_k] = np.mean(tmp_SNR_list)

        K_t = success_k_len_list.index(max(success_k_len_list))

        for index in success_k_list[K_t]:
            update_poi_list.append(index)

        return update_poi_list, K_t, success_k_SNR_list[K_t],success_k_list[K_t]

    def step(self, actions):
        action = copy.deepcopy(actions.reshape(self.sg.V['NUM_UAV'], self.num_action))

        uav_reward_list = [0] * self.num_uav
        uav_energy_list = [0] * self.num_uav
        uav_penalty_list = [0] * self.num_uav
        uav_snr_list = [0] * self.num_uav
        update_poi_list = []  # Updated PoI access times
        update_aoi_result = []
        clear_uav = copy.copy(self.uav_position)
        new_uav_positions = []
        update_poi_uav = []
        # update positions of UAVs
        for n in range(self.num_uav):
            new_x = self.uav_position[n][0] + action[n][0]
            new_y = self.uav_position[n][1] + action[n][1]
            tmp_new_position = [new_x, new_y]
            update_poi_uav_tmp = []
            # if reach OB or WALL, give negative reward, save original positon
            if self.judge_obstacle(tmp_new_position, self.uav_position[n]):
                new_uav_positions.append([self.uav_position[n][0], self.uav_position[n][1]])
                self.uav_trace[n].append([self.uav_position[n][0], self.uav_position[n][1]])
                uav_penalty_list[n] += self.sg.V['WALL_PENALTY']
                uav_consume_energy = self.consume_energy(self.sg.V["TIME_SLOT"], 0, 0) * 1e-3  # kJ
                self.collision[n] += 1
            else:
                new_uav_positions.append([new_x, new_y])
                self.uav_trace[n].append([new_x, new_y])
                distance = np.sqrt(np.power(action[n][0], 2) + np.power(action[n][1], 2)) * self.sg.V["GRID_LENGTH"]
                fly_time = distance / self.sg.V["VELOCITY"]
                hover_time = self.sg.V["TIME_SLOT"] - fly_time
                update_poi_list, K_t, poi_SNR ,update_poi_uav_tmp= self.update_poi_status(new_uav_positions[-1], update_poi_list,
                                                                       hover_time)
                uav_consume_energy = self.consume_energy(fly_time, hover_time, K_t) * 1e-3  # kJ
                uav_snr_list[n] = poi_SNR
                if self.sg.V["DEBUG_MODE"]:
                    print(K_t)

            update_poi_uav.append(update_poi_uav_tmp)
            uav_energy_list[n] = uav_consume_energy
            self.energy_consumption[n] += uav_consume_energy

        self.update_uav_snr_list.append(uav_snr_list)
        self.update_poi_list_all.append(update_poi_uav)
        for index in range(self.data_position.shape[0]):
            if index in update_poi_list:
                self.map_aoi[index] = 1
            else:
                self.map_aoi[index] += 1
            update_aoi_result.append([index, self.map_aoi[index]])

        self.aoi_list.append(np.mean(self.map_aoi))
        self.episodic_uav_snr_list.append(np.mean(uav_snr_list))
        self.energy_list.append(np.mean(self.energy_consumption))
        self.map_aoi_list.append(self.map_aoi)

        reward = self.aoi_list[-2] - self.aoi_list[-1] - np.mean(uav_energy_list) * self.sg.V["ENERGY_FACTOR"] \
                 + np.mean(uav_penalty_list)

        self.uav_position = new_uav_positions
        self.__draw_image(clear_uav, update_aoi_result)

        state = self.__get_state()
        done = False

        if self.sg.V["TESTING_MODE"]:
            info = {"performance_info": {
                "mean_aoi": np.mean(self.map_aoi),
                "sum_aoi": np.sum(self.map_aoi),
                "mean_aoi_episode": np.mean(self.aoi_list),
                "mean_use_energy": np.mean(self.energy_consumption),
                "collision": self.collision,
            },
                "log_info": {
                    "mapx": self.mapx,
                    "mapy": self.mapy,
                    "num_uav": self.num_uav,
                    "uav_trace": self.uav_trace,
                    "data_position": self.data_position,
                    "map_ob": self.map_ob,
                    "map_aoi": self.map_aoi,
                    "mean_aoi": np.mean(self.map_aoi),
                    "mean_use_energy": np.mean(self.energy_consumption),
                    "aoi_list": self.aoi_list,
                    "energy_list": self.energy_list,
                    "map_aoi_list": self.map_aoi_list,
                    "episodic_uav_snr_list": self.episodic_uav_snr_list,
                    "update_aoi_list":self.update_poi_list_all,
                    "update_snr_list":self.update_uav_snr_list
                }
            }
        else:
            info = {"performance_info": {
                "mean_aoi": np.mean(self.map_aoi),
                "sum_aoi": np.sum(self.map_aoi),
                "mean_aoi_episode": np.mean(self.aoi_list),
                "mean_use_energy": np.mean(self.energy_consumption),
                "collision": self.collision, },
            }

        return state, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError
