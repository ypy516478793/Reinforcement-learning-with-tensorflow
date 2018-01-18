import numpy as np
import matplotlib.pyplot as plt
from Q_learning import QLearningTable
from MDP_env import MDP_env

def run():
    for i_episode in range(num_episode):
        state = env.reset()
        step = 0
        epi_reward = 0
        done = False
        while not done:
            action = Agent.choose_action(state)
            state_, reward, done = env.step(action)
            Agent.learn(state, action, reward, state_)

            print("Episode: " + str(i_episode+1),
                  " Steps: " + str(step+1),
                  " (State, Action): " + str((state, action)),
                  " State_: " + str(state_),
                  " e_greedy: " + str(Agent.epsilon)
                  )

            step += 1
            state = state_
            epi_reward += reward

        total_reward[i_episode] = total_reward[i_episode-1] + epi_reward
        Agent.anneal(steps=step)

def plot_figures():
    # plot average reward figure
    plt.figure(1)
    x = np.arange(1, num_episode + 1)
    y = total_reward / x
    plt.plot(x, y)
    plt.ylim(-0.01, 0.2)
    plt.xlabel("episodes")
    plt.title("average reward")
    plt.savefig("Average_reward_only_QL.png")
    plt.show()

if __name__ == "__main__":
    n_actions = 2
    num_episode = 12000
    total_reward = np.zeros(num_episode)
    env = MDP_env()
    Agent = QLearningTable(n_actions,
                           learning_rate=0.00025,
                           reward_decay=0.975,
                           e_greedy=1,
                           e_decrement=(1 - 0.1) / 2000
                           )
    run()

    Agent.q_table[0]['action'] = Agent.q_table[0].idxmax(axis=1)
    print("")
    print("q_table:")
    print(Agent.q_table[0].sort_index(axis=0, ascending=True))

    plot_figures()