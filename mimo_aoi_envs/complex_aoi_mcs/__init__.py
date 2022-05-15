from gym.envs.registration import register

register(
    id='ComplexAoI-v0',
    entry_point='mimo_aoi_envs.complex_aoi_mcs.aoi_mcs_collection:Env'
)