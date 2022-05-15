import os
import gym
import numpy
import pandas as pd
from mimo_aoi_envs.log_aoi_vec import *
import mimo_aoi_envs.real_aoi


class BranchingActionEnv(gym.Wrapper):

    def __init__(self, env_name, bins):
        self.env = gym.make(env_name)
        super().__init__(self.env)
        self.bins = bins
        self.discretized_space = np.linspace(self.env.action_space.low, self.env.action_space.high, bins).transpose(
            (1, 0))

    def step(self, a):
        action_n = np.array([self.discretized_space[index][aa] for index, aa in enumerate(a)])
        # action_n = np.zeros_like(action_n)
        return super().step(action_n)

    def random_step(self):
        discretized_action_n = np.random.randint(0, self.bins, size=self.env.action_space.shape)
        return self.step(discretized_action_n)


# TODO:adjust
exp_name = "try"

path = "./random_test/" + exp_name
if os.path.exists(path) is False:
    os.makedirs(path)

env = BranchingActionEnv("RealAoI-v0", bins=11)  # TODO：如果不想要Multi-discrete,就用env=gym.make(env_name)
start_env = env.reset()

print(env.observation_space)
print(env.action_space)
print('Starting a new TEST iterations...')

iteration = 0
episode_rewards = [0.0]  # sum of rewards for all agents
episode_step = 0
num_timeslots = 250
num_episode = 1

df = pd.DataFrame(
    columns=["Episode", "episodic average AoI per PoI", "current average AoI per PoI", "average energy consumption per UAV", "average SNR per PoI"])

while iteration < num_episode:
    new_obs_n, rew_n, done_n, info_n = env.random_step()
    obs_n = new_obs_n
    done = done_n
    episode_step += 1
    terminal = (episode_step >= num_timeslots)
    episode_rewards[-1] += rew_n  # 每一个step的总reward
    # print(rew_n)

    if done or terminal:
        print('\n%d th episode:\n' % iteration)
        print('\treward:', np.sum(episode_rewards))
        print("\ninfo:", info_n["performance_info"])


        log = Log_DEBUG(path=path, id=iteration, num_timeslots=num_timeslots, detail_and_gif=False,
                        **info_n['log_info'])
        log.draw_trajectory()
        # log.save_to_file("GA",iteration)
        del log

        df.loc[iteration] = [iteration,info_n['performance_info']['mean_aoi_episode'],
                             info_n['performance_info']['mean_aoi'],
                             info_n['performance_info']['mean_use_energy'],
                             np.mean(info_n['log_info']['episodic_uav_snr_list'])]

        obs_n = env.reset()
        episode_step = 0
        episode_rewards = [0.0]
        iteration += 1

df.sort_values("episodic average AoI per PoI", inplace=True)
df.loc[num_episode] = df.mean(axis=0)
df.iloc[num_episode][0] = "-1"
df.to_csv(path + "/testing_performance.csv", index=0)
