import logging, os, sys
from gym_simplegrid.envs import SimpleGridEnv
from datetime import datetime as dt
import gymnasium as gym
from gymnasium.utils.save_video import save_video
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    # Folder name for the simulation
    FOLDER_NAME = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(f"log/{FOLDER_NAME}")

    # Logger to have feedback on the console and on a file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    logger.info("-------------START-------------")

    options = {
        "start_loc": 0,
        "goal_loc": 15,
    }

    obstacle_map = [
        "0010",
        "1000",
        "0010",
        "0000",
    ]

    decision_map = [
        [["R"], ["D"], ["X"], ["D"]],
        ["X", ["R", "D"], ["R"], ["D"]],
        [["R", "D"], ["D"], ["X"], ["D"]],
        [["R"], ["R"], ["R"], ["X"]],
    ]

    env = gym.make(
        "SimpleGrid-4x4-v0",
        obstacle_map=obstacle_map,
        render_mode="none",
        decision_map=decision_map,
    )
    obs, info = env.reset(seed=1, options=options)
    rew = env.unwrapped.reward
    done = env.unwrapped.done

    logger.info("Running action-perception loop...")
    episodes = 10000
    data ={}
    with open(f"log/{FOLDER_NAME}/history.csv", "w") as f:
        f.write(f"step,x,y,reward,done,action\n")
        for episode in tqdm(range(episodes)):
            obs, info = env.reset(seed=1, options=options)
            env.unwrapped.episode = episode
            env.unwrapped.step_count = 0
            env.unwrapped.reward_array = []
            rew = env.unwrapped.reward
            done = env.unwrapped.done           
            while not done:        
                _,_,action = env.unwrapped.get_action() 
                obs, rew, done, _, info = env.step(action)
            data[episode] = {
                "steps": env.unwrapped.step_count,
                "episode_reward": sum(env.unwrapped.reward_array),
                "reward_steps": env.unwrapped.reward_array,
                "average_reward": sum(env.unwrapped.reward_array) / len(env.unwrapped.reward_array),
            }

    Q_values = env.unwrapped.Q_values
    logger.info("...done")
    logger.info("-------------END-------------")

    # # Extract the steps and rewards from the data
    # steps = [data[episode]["steps"] for episode in range(episodes)]
    # rewards = [sum(data[episode]["reward"]) for episode in range(episodes)]

    # Extract the steps and rewards from the data
    # steps = [data[episode]["steps"] for episode in range(episodes)]
    # average_reward = [sum(data[episode]["episode_reward"]) for episode in range(episodes)]

    # steps = [data[episode]["steps"] for episode in range(episodes)]
    # average_steps = [data[episode]["average_steps"] for episode in range(episodes)]
    # avg_reward = [data[episode]["average_reward"] for episode in range(episodes)]

    # Assuming self.reward_array is accessible and contains the reward data
    # cumulative_rewards = np.cumsum(data[episode]["average_reward"])

    # plt.xlabel('Step')
    # plt.ylabel('Cumulative Reward')
    # plt.savefig('report_data/cumulative_reward_plot.png')

    # Create a figure for the steps
    # fig1, ax1 = plt.subplots()
    # ax1.set_xlabel('Episode')
    # ax1.set_ylabel('Steps')
    # ax1.plot(range(episodes), steps, color='tab:blue')
    # plt.savefig('report_data/steps_plot.png')

    # Create a figure for the rewards
    # fig2, ax2 = plt.subplots()
    # ax2.set_xlabel('Episode')
    # ax2.set_ylabel('Avg Reward')
    # ax2.plot(range(episodes), average_reward, color='tab:red')
    # plt.savefig('report_data/reward_plot.png')

    # action_names = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
    # grid_size = (4, 4)

    # # For each state, print the Q-values
    # for y in range(grid_size[1]):
    #     for x in range(grid_size[0]):
    #         state = (x, y)
    #         print(f"State: {state}")
    #         if state in Q_values:
    #             for action, q_value in Q_values[state].items():
    #                 print(f"  Action: {action_names[action]}, Q-value: {q_value}")
    #         else:
    #             print("  No Q-values")
    
    env.close()