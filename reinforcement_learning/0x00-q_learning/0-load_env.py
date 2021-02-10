#!/usr/bin/env python3
"""function load_frozen_lake"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Function that that loads the pre-made FrozenLakeEnv evnironment
    from OpenAI’s gym

    Parameters
    ----------
    desc : list, optional
        is either None or a list of lists containing a custom description
        of the map to load for the environment, by default None
    map_name : str, optional
        is either None or a string containing the pre-made map to load,
        by default None
    is_slippery : bool, optional
        is a boolean to determine if the ice is slippery, by default False

    Note
    ----
        If both desc and map_name are None, the environment will load a
        randomly generated 8x8 map

    Returns
    -------
    env : pre-made FrozenLakeEnv evnironment from OpenAI’s gym
        the environment
    """
    env = gym.make("FrozenLake-v0",
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)

    return env
