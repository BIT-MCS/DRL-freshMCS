from gym.envs.registration import register

register(
    id='RealAoI-v0',
    entry_point='mimo_aoi_envs.real_aoi.real_aoi_collection:Env'
)