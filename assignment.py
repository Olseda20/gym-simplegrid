if __name__ == "__main__":
    import logging, os, sys
    from gym_simplegrid.envs import SimpleGridEnv
    from datetime import datetime as dt
    import gymnasium as gym
    from gymnasium.utils.save_video import save_video

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

    rounds = 50

    env = gym.make(
        "SimpleGrid-4x4-v0",
        obstacle_map=obstacle_map,
        render_mode="human",
        decision_map=decision_map,
        human_feedback=0,
        retrospective_feedback=True,
        reward_type="",
    )
    env.human_feedback = 1
    obs, info = env.reset(seed=1, options=options)
    rew = env.unwrapped.reward
    done = env.unwrapped.done

    logger.info("Running action-perception loop...")

    with open(f"log/{FOLDER_NAME}/history.csv", "w") as f:
        f.write(f"step,x,y,reward,done,action\n")
        for t in range(50):        
            action = env.action_space.sample()
            f.write(
                f"{t},{info['agent_xy'][0]},{info['agent_xy'][1]},{rew},{done},{action}\n"
            )
        
            if done:
                logger.info(f"...agent is done at time step {t}")
                break
        
            obs, rew, done, _, info = env.step(action)

    if env.render_mode == "rgb_array_list":
        frames = env.render()
        save_video(frames, f"log/{FOLDER_NAME}", fps=env.fps)
    logger.info("...done")
    logger.info("-------------END-------------")

    env.close()
