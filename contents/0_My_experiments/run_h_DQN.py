from DQN import DeepQNetwork
from MDP_env import MDP_env

def run_MDP(num_episodes):
    total_step = 0  #number of total transitions
    for i_episode in range(num_episodes):
        # initial state and goal
        state = env.reset()
        goal = agent.choose_goal(state, meta_e_greedy)
        agent.goal_attempts[goal] += 1
        done = False

        while not done:
            total_extrinsic_reward = 0
            start_state = state
            step = 0
            goal_reached = False

            while not done and not goal_reached:
                # while s is not terminal or goal is not reached
                action = agent.choose_action(state, goal, e_greedy)
                # execute action and obtain next state and extrinsic reward from environment
                state_, ex_reward, done, info = env.step(action)
                # obtain intrinsic reward from internal critic
                in_reward, goal_reached = agent.internal_critic(action)
                # store transition for controller
                agent.store_transition(state, goal, action, in_reward, state_)

                if total_step > 200:
                    # update parameters both for controller and meta_controller
                    agent.learn()

                total_extrinsic_reward += ex_reward
                state = state_
                step += 1
                total_step += 1

            # store transition for meta_controller
            agent.store_meta_transition(start_state, goal, total_extrinsic_reward, state_)
            # anneal epsilon greedy rate for controller
            e_greedy -= anneal_factor * step

            if not done:  # also means goal is reached
                # anneal epsilon greedy rate for meta_controller
                meta_e_greedy -= anneal_factor
                #
                agent.goal_success[goal] += 1
                # choose a new goal
                goal = agent.choose_goal(state, meta_e_greedy)


if __name__ == "__main__":
    env = MDP_env()
    n_actions = 2
    n_features = 1
    agent = DeepQNetwork(n_actions, n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run_MDP()
    agent.plot_cost()




