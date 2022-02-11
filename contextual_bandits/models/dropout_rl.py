import numpy as np
import torch

from trainer import RLReg


class DropoutRLReg(RLReg):
    """ Class for training an AgentB """

    def __init__(self, X_train, y_train, agent, buffer_size=4096, minibatch_size=64, burn_in=500,
                 criterion=torch.nn.MSELoss()):
        super(DropoutRLReg, self).__init__(X_train, y_train, agent, buffer_size, minibatch_size, burn_in)
        self.criterion = criterion

    def get_loss_(self, context_inds, actions, rewards):
        rewards_preds = self.agent.evaluate(
            torch.repeat_interleave(self.X_train[context_inds], self.agent.sample, dim=1))
        reward_preds = rewards_preds[np.arange(self.minibatch_size), actions]
        loss = self.criterion(reward_preds, rewards)
        return loss
