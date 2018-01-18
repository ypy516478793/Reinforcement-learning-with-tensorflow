import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from DQN import DeepQNetwork
from MDP_env import MDP_env
from collections import defaultdict


def run_MDP():
    total_step = 0  # number of total transitions
    for i_episode in range(num_episodes):
        id_episode = i_episode // unit
        # initial state and goal
        state = env.reset()
        state_array = one_hot(state)
        # state_array = np.array([state])
        visits[id_episode][state] += 1
        goal = meta_controller.choose_action(state_array)
        # goal = 2          ###############
        controller.goal_attempts[goal] += 1
        done = False

        while not done:
            extrinsic_reward = 0
            start_state = state_array
            step = 0
            goal_reached = False

            while not done and not goal_reached:
                # while s is not terminal or goal is not reached
                action = controller.choose_action(state_array, goal=goal)
                # execute action and obtain next state and extrinsic reward from environment
                state_, ex_reward, done = env.step(action)
                visits[id_episode][state_] += 1
                # obtain intrinsic reward from internal critic
                in_reward, goal_reached = meta_controller.criticize(state_, goal)
                state__array = one_hot(state_)
                # store transition ({s, g}, a, r, {s_, g}) for controller
                controller.store_transition(state_array, action, in_reward, state__array, goal)
                if total_step > 200:
                    # update parameters for controller
                    controller.learn()
                    # update parameters for meta_controller
                    meta_controller.learn()

                print("Episode: " + str(i_episode + 1),
                      " Steps: " + str(step + 1),
                      " Total_step: " + str(total_step + 1),
                      " (State, Action): " + str((state, action)),
                      " State_: " + str(state_),
                      " Goal: " + str(goal),
                      " meta_e: " + str(meta_controller.epsilon[0]),
                      " con_e: " + str(controller.epsilon)
                      )

                extrinsic_reward += ex_reward
                state = state_
                state_array = one_hot(state)
                # state_array = np.array([state])
                step += 1
                total_step += 1

            if goal_reached:
                controller.goal_success[goal] += 1
                print("Goal reached!")

            # # update parameters for meta_controller
            # meta_controller.learn()
            # store transition (s_0, g, ex_r, s_) for meta_controller
            meta_controller.store_transition(start_state, goal, extrinsic_reward, state__array)
            # anneal epsilon greedy rate for controller
            controller.anneal(step, goal=goal, adaptively=False, success=goal_reached)
            # # anneal epsilon greedy rate for meta_controller
            # meta_controller.anneal()

            if not done:  # when goal is terminal, goal_reached = True & done = True
                # choose a new goal
                goal = meta_controller.choose_action(state_array)
                # goal = 2          ###############
                controller.goal_attempts[goal] += 1

            total_extrinsic_reward[i_episode] = total_extrinsic_reward[i_episode - 1] + extrinsic_reward

        # anneal epsilon greedy rate for meta_controller
        meta_controller.anneal()

        if (i_episode + 1) % unit == 0:
            if id_episode == 0:
                for k in range(n_goals):
                    goal_attempts[id_episode, k] = controller.goal_attempts[k]
                    goal_success[id_episode, k] = controller.goal_success[k]
            else:
                for k in range(n_goals):
                    goal_attempts[id_episode, k] = controller.goal_attempts[k] - \
                                                   np.sum(goal_attempts[:id_episode, k])
                    goal_success[id_episode, k] = controller.goal_success[k] - \
                                                  np.sum(goal_success[:id_episode, k])

def one_hot(state):
    state_vector = np.zeros([n_features])
    state_vector[state] = 1
    return state_vector

def output_q_tables():
    n_states = 6
    # all_state = np.arange(n_states).reshape((n_states, 1))
    all_state = np.eye(n_states)
    q_table = defaultdict(lambda: pd.DataFrame(index=list(range(n_states)), columns=list(range(n_actions)), dtype=np.float64))
    for g in range(n_goals):
        all_new_state = np.hstack((all_state, np.ones((n_states, 1)) * g))
        q_table[g].iloc[:,:] = controller.sess.run(controller.q_eval, feed_dict={controller.s: all_new_state})
        q_table[g]['action'] = q_table[g].idxmax(axis=1)
        print("")
        print("Controller's q_table (Goal = " + str(g) + "):")
        print(q_table[g].sort_index(axis=0, ascending=True))
    # g = 2
    # all_state = np.hstack((all_state, np.ones((n_states, 1)) * g))
    # q_table[g].iloc[:, :] = controller.sess.run(controller.q_eval, feed_dict={controller.s: all_state})
    # q_table[g]['action'] = q_table[g].idxmax(axis=1)
    # print("")
    # print("Controller's q_table (Goal = " + str(g) + "):")
    # print(q_table[g].sort_index(axis=0, ascending=True))

# def output_qt_tables():
#     n_states = 6
#     # all_state = np.arange(n_states).reshape((n_states, 1))
#     all_state = np.eye(n_states)
#     qt_table = defaultdict(lambda: pd.DataFrame(index=list(range(n_states)), columns=list(range(n_actions)), dtype=np.float64))
#     # for g in range(n_goals):
#     #     all_new_state = np.hstack((all_state, np.ones((n_states, 1)) * g))
#     #     q_table[g].iloc[:,:] = controller.sess.run(controller.q_eval, feed_dict={controller.s: all_new_state})
#     #     q_table[g]['action'] = q_table[g].idxmax(axis=1)
#     #     print("")
#     #     print("Controller's q_table (Goal = " + str(g) + "):")
#     #     print(q_table[g].sort_index(axis=0, ascending=True))
#     g = 2
#     all_state = np.hstack((all_state, np.ones((n_states, 1)) * g))
#     qt_table[g].iloc[:, :] = controller.sess.run(controller.q_next, feed_dict={controller.s_: all_state})
#     qt_table[g]['action'] = qt_table[g].idxmax(axis=1)
#     print("")
#     print("Controller's qt_table (Goal = " + str(g) + "):")
#     print(qt_table[g].sort_index(axis=0, ascending=True))


def plot_figures():
    # plot visits
    x = list(range(1, 13))
    plt.figure(1)
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.plot(x, visits[:, i] / unit)
        plt.xlim(1, 12)
        plt.ylim(-0.01, 2)
        plt.title("S" + str(i))
        plt.grid(True)
    plt.savefig("h_DQN_MDP.png")

    # plot goal success rates
    plt.figure(2)
    for i in range(6):
        plt.plot(x, goal_success[:, i] / goal_attempts[:, i])
    plt.xlim(1, 12)
    plt.title("Goals success rates")
    plt.savefig("Goals_success_rates_DQN.png")

    # plot average reward from the environment
    plt.figure(3)
    x_episode = np.arange(1, num_episodes + 1)
    average_reward = total_extrinsic_reward / x_episode
    plt.plot(x_episode, average_reward)
    plt.title("Average reward")
    plt.savefig("Average_reward_DQN.png")

    # plot learning cost history for controller and meta_controller
    plt.figure(4)
    plt.plot(np.arange(len(controller.cost_his)), controller.cost_his)
    plt.title("Controller learning cost")
    plt.ylabel("Cost")
    plt.xlabel("training steps")
    plt.savefig("Controller_learning_cost_DQN.png")
    plt.figure(5)
    plt.plot(np.arange(len(meta_controller.cost_his)), meta_controller.cost_his)
    plt.title("Meta_Controller learning cost")
    plt.ylabel("Cost")
    plt.xlabel("training steps")
    plt.savefig("Meta_Controller_learning_cost_DQN.png")

    plt.show()


if __name__ == "__main__":
    env = MDP_env()
    n_actions = 2
    n_features = 6
    n_goals = 6
    # n_goals = 1          ###############
    num_episodes = 12000
    unit = 1000
    visits = np.zeros((12, 6))
    goal_attempts = np.zeros((12, 6))
    goal_success = np.zeros((12, 6))
    total_extrinsic_reward = np.zeros(num_episodes)
    # controller needs to choose actions under certain goal
    controller = DeepQNetwork(n_actions,
                              n_features,
                              n_goals,
                              # learning_rate=0.00025,
                              learning_rate=0.001,
                              reward_decay=0.975,
                              e_greedy=1,
                              e_decrement=(1 - 0.1) / num_episodes,
                              output_graph=True
                              )
    # meta_controller's action sets are to choose different goals (default goal=0)
    meta_controller = DeepQNetwork(n_goals,  # n_actions = n_goals, n_goals = 1 (attributes for meta_controller)
                                   n_features,
                                   # learning_rate=0.00025,
                                   learning_rate=0.001,
                                   reward_decay=0.975,
                                   e_greedy=1,
                                   e_decrement=(1 - 0.1) / num_episodes,
                                   meta=True
                                   )
    run_MDP()
    output_q_tables()
    plot_figures()




