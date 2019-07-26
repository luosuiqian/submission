from sys import exit, exc_info, argv
import numpy as np
import pandas as pd

from netsapi.challenge import *


class CustomAgent:
    def __init__(self, environment):
        self.environment = environment
        self.action_policy = []
        self.action_list = self._get_action_list()
        self.action_numbers = [0 for _ in range(5)]
        self.adjust_list = self._get_adjust_list()
        return

    def _get_action_list(self):
        action_list = []
        for _ in range(5):
            actions = []
            for i in range(5):
                for j in range(5):
                    actions.append([(i * 2 + 1) / 10.0, (j * 2 + 1) / 10.0])
            random.shuffle(actions)  # shuffle the valid actions
            action_list.append(actions)
        return action_list

    def _get_adjust_list(self):
        random_adjust = [[1] * 5 for _ in range(6)]
        order_adjust = []
        for _ in range(3):
            for i in range(5):
                order_adjust.append([int(i == j) for j in range(5)])
        return random_adjust + order_adjust

    def _get_out_policy(self, policy_number):
        policy = {}
        for i in range(5):
            policy[str(i + 1)] = self.action_list[i][policy_number[i]][:]
        return policy

    def _get_policy(self, candidates, rewards):
        if len(rewards) == 0:
            policy_number = self.action_numbers[:]
        else:
            best_index = np.argmax(rewards)
            policy_number = self.action_policy[best_index][:]
            r = self.adjust_list[len(rewards)]
            for i in range(5):
                if r[i] == 1:
                    self.action_numbers[i] += 1
                    policy_number[i] = self.action_numbers[i]
        self.action_policy.append(policy_number)
        policy = self._get_out_policy(policy_number)
        return policy

    def generate(self):
        best_policy = None
        best_reward = -float('Inf')
        candidates = []
        rewards = []
        try:
            for i in range(21):
                policy = self._get_policy(candidates, rewards)
                while True:  # Use while-loop to get valid reward.
                    self.environment.reset()
                    reward = self.environment.evaluatePolicy(policy)
                    if np.isnan(reward):
                        print("\n\nThe reward is nan! Evaluations Remaining should add 5!\n\n")
                    else:
                        candidates.append(policy)
                        rewards.append(reward)
                        print(i + 1, reward, policy)
                        break  # Already got the valid reward. Break the while-loop.
            best_policy = candidates[np.argmax(rewards)]
            best_reward = rewards[np.argmax(rewards)]
        except (KeyboardInterrupt, SystemExit):
            print(exc_info())
        return best_policy, best_reward
