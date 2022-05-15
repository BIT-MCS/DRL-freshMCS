from gym.envs.registration import register

register(
    id='CrazyMCS-v0',
    entry_point='mcs_envs.crazy_env.crazy_data_collection:Env'
)