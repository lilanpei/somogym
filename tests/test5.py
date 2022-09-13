import yaml
import gym
from pathlib import Path
import os
import sys
from stable_baselines3 import SAC
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.evaluation import evaluate_policy
import pytest

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)
from environments.utils.import_handler import import_environment

def load_env(
    env_name, render=False
):  # todo: this should probably be a utility for general use

    # get the default run config path
    run_config_file = (
        Path(os.path.dirname(__file__))
        / "../environments"
        / env_name
        / "benchmark_run_config.yaml"
    )

    with open(run_config_file, "r") as config_file:
        run_config = yaml.safe_load(config_file)

    debug = False

    # prepare env
    import_environment(env_name)
    env = gym.make(
        run_config["env_id"],
        run_config=run_config,
        run_ID=f"{env_name}-sim_test",
        # todo: may be better to check whether the run ID exists already
        render=render,
        debug=debug,
    )
    return env

if __name__ == "__main__":
    env = load_env("InHandManipulationInverted", render=True)

    model = SAC.load("sac_InHandManipulationInverted_w_es_150_bs_2e6_s_17_zs_1000_zr_0_0912")
    obs = env.reset()
    idx = 1
    total = 0
    z_r = []
    pre_zr = 0
    delta_zr = []
    rewards = []
    total_rewards = []
    while True:
        action, _states = model.predict(obs, deterministic=True)
        #print("$$$action", action)
        obs, reward, done, info = env.step(action)
        z_r.append(np.degrees(info['z_rotation_step']))
        total += reward
        total_rewards.append(total)
        rewards.append(reward)
        print(idx, total, reward, np.degrees(info['z_rotation_step']))
        idx += 1
        env.render()
        if done:
            print("total_rewards :", total_rewards)
            print("rewards :",rewards)
            print("z_r :", z_r)
            print("#####", total, reward, done, info)
            idx = 1
            total = 0
            z_r = []
            obs = env.reset()