# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        for i in xrange(self.iterations):
            currValue = self.values.copy()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    self.values[state] = 0
                else:
                    value_list = []
                    for action in self.mdp.getPossibleActions(state):
                        value = 0
                        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                            value += transition[1] * (self.mdp.getReward(state, action, transition[0]) + self.discount * currValue[transition[0]])
                        value_list.append(value)
                    self.values[state] = max(value_list)


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        value = 0
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            value += transition[1] * (self.mdp.getReward(state, action, transition[0]) + self.discount * self.values[transition[0]])
        return value

    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None
        else:
            bestVal = -99999
            bestAction = 0
            all_actions = self.mdp.getPossibleActions(state)
            for action in all_actions:
                value = 0
                for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                    value += transition[1] * (self.mdp.getReward(state, action, transition[0]) + self.discount * self.values[transition[0]])
                if value > bestVal:
                    bestAction = action
                    bestVal = value
            return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
