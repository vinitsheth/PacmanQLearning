# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np

import random,util,math

class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        return self.q_values[(state, action)]


    def computeValueFromQValues(self, state):
        qVals = []
        for action in self.getLegalActions(state):
            qVals.append(self.getQValue(state, action))
        if len(self.getLegalActions(state)) == 0:
            return 0
        else:
            return max(qVals)

    def computeActionFromQValues(self, state):
        maxAction = None
        maxQVal = 0
        for action in self.getLegalActions(state):
            qVal = self.getQValue(state, action)
            if qVal > maxQVal or maxAction is None:
                maxQVal = qVal
                maxAction = action
        return maxAction

    def getAction(self, state):
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        if util.flipCoin(self.epsilon):
            self.randomCount += 1
            return random.choice(legalActions)
        else:
            self.policyCount += 1
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        q1 = (1 - self.alpha) * self.getQValue(state, action)
        if len(self.getLegalActions(nextState)) == 0:
            sample = reward
        else:
            sample = reward + (self.discount * max([self.getQValue(nextState, next_action) for next_action in self.getLegalActions(nextState)]))
        q2 = self.alpha * sample

        self.q_values[(state, action)] = q1 + q2

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class QLearningAgentMetro(ReinforcementAgent):
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.q_values = util.Counter()

    def getQValue(self, state, action):
        return self.q_values[(state, action)]


    def computeValueFromQValues(self, state):
        qVals = []
        for action in self.getLegalActions(state):
            qVals.append(self.getQValue(state, action))
        if len(self.getLegalActions(state)) == 0:
            return 0
        else:
            return max(qVals)

    def computeActionFromQValues(self, state):
        maxAction = None
        maxQVal = 0
        for action in self.getLegalActions(state):
            qVal = self.getQValue(state, action)
            if qVal > maxQVal or maxAction is None:
                maxQVal = qVal
                maxAction = action
        return maxAction

    def getAction(self, state):
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        actionR = random.choice(legalActions)
        valueR = self.getQValue(state, actionR)
        actionP = self.computeActionFromQValues(state)
        valueP = self.getQValue(state, actionP)
        e  = random.random()

        if e < np.exp(valueR - valueP)/float(self.temperature):
            self.randomCount += 1
            return actionR
        else:
            self.policyCount += 1
            return actionP

    def update(self, state, action, nextState, reward):
        q1 = (1 - self.alpha) * self.getQValue(state, action)
        if len(self.getLegalActions(nextState)) == 0:
            sample = reward
        else:
            sample = reward + (self.discount * max([self.getQValue(nextState, next_action) for next_action in self.getLegalActions(nextState)]))
        q2 = self.alpha * sample

        self.q_values[(state, action)] = q1 + q2

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class PacmanQAgentMetro(QLearningAgentMetro):
    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgentMetro.__init__(self, **args)

    def getAction(self, state):
        action = QLearningAgentMetro.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        return self.weights * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        diff = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for feature in features:
            self.weights[feature] += self.alpha * diff * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        if self.episodesSoFar == self.numTraining:
            pass
class ApproximateQAgentMetro(PacmanQAgentMetro):
    def __init__(self, extractor='SimpleExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgentMetro.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        return self.weights * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        diff = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)
        features = self.featExtractor.getFeatures(state, action)
        for feature in features:
            self.weights[feature] += self.alpha * diff * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgentMetro.final(self, state)

        if self.episodesSoFar == self.numTraining:
            pass