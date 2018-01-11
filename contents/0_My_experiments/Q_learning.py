"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
from collections import defaultdict


class QLearningTable:
    def __init__(self, n_actions, n_goals=1, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, e_decrement=0.01):
        self.actions = list(range(n_actions))  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = np.ones(n_goals) * e_greedy
        self.de_anneal = e_decrement
        self.goal_attempts = defaultdict(int)
        self.goal_success = defaultdict(int)
        self.q_table = defaultdict(lambda: pd.DataFrame(columns=self.actions, dtype=np.float64))
        # self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation, goal=0):
        self.check_state_exist(observation, goal)
        # action selection
        if np.random.uniform() > self.epsilon[goal]:
            # choose best action
            state_action = self.q_table[goal].ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, goal=0):
        self.check_state_exist(s_, goal=goal)
        q_predict = self.q_table[goal].ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table[goal].ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table[goal].ix[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state, goal=0):
        if state not in self.q_table[goal].index:
            # append new state to q table
            self.q_table[goal] = self.q_table[goal].append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table[goal].columns,
                    name=state,
                )
            )

    def criticize(self, next_state, goal):
        if next_state == goal:
            in_reward = 1
            goal_reached = True
        else:
            in_reward = 0
            goal_reached = False
        return in_reward, goal_reached

    def anneal(self, steps=1, goal=0, adaptively=False):
        if self.epsilon[goal] > 0.1:
            if adaptively:
                success_rate = self.goal_success[goal] / self.goal_attempts[goal]
                if success_rate == 0:
                    self.epsilon[goal] -= self.de_anneal * steps
                else:
                    self.epsilon[goal] = 1 - success_rate
            else:
                self.epsilon[goal] -= self.de_anneal * steps
            if self.epsilon[goal] < 0.1:
                self.epsilon[goal] = 0.1



