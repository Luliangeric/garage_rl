from env.garage import Garage
from rl_brain import DoubleDQN
import matplotlib.pyplot as plt
from env.graph import Vision


env = Garage("./data/garage.json")

MEMORY_SIZE = 3000

double_DQN = DoubleDQN(
    n_actions=env.parking_number, n_features=env.parking_number+2,
    memory_size=MEMORY_SIZE)


def train():
    total_steps = 0
    times = 0
    total_reward = list()

    while times < 800:
        times += 1
        reward_sum = 0

        env.reset()
        observation_ = None
        while True:
            observation, car = env.check_state()
            if car:
                action = double_DQN.choose_action(observation)
                reward, done = env.step(action, car)
                reward_sum += reward

                if observation_:
                    double_DQN.store_transition(observation, action, reward, observation_)

                observation_ = observation

                if total_steps > MEMORY_SIZE:
                    double_DQN.learn()

                if done:
                    print('iter:', times,' - reward = ', reward_sum)
                    total_reward.append(reward_sum)
                    break

                total_steps += 1

    # env.reset()
    # garage = Vision(7000)
    # garage.initwindows("./data/garage.json", "./data/map1.txt")
    # while True:
    #     observation, car = env.check_state()
    #
    #     if car:
    #         action = double_DQN.choose_action(observation)
    #         print(action)
    #         _, done = env.step(action, car)
    #
    #         if done:
    #             break
    #
    #     garage.updatesim(env.state, 0.1)
    # print("stop")
    # garage.mainloop()
    return total_reward


reward_list = train()
plt.plot(reward_list)

plt.show()