'''A wrapper class for optimizer '''
import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, peer_num, n_warmup_steps, init_lr, total_iterations):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = init_lr
        self.peer_num = peer_num
        # self.init_lr = np.power(d_model, -0.5)
        # self.factor = factor
        self.total_iterations = total_iterations

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_current_steps = state_dict['n_current_steps']
        self.init_lr = state_dict['init_lr']
        # self.factor = state_dict['factor']

    def state_dict(self):
        state_dict = {}
        state_dict['optimizer_state_dict'] = self._optimizer.state_dict()
        state_dict['n_warmup_steps'] = self.n_warmup_steps
        state_dict['n_current_steps'] = self.n_current_steps
        state_dict['init_lr'] = self.init_lr
        # state_dict['factor'] = self.factor
        return state_dict

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        # lr = self.factor * self.init_lr * self._get_lr_scale()
        lr = self. _get_decayed_learning_rate()

        # if self.peer_num > 2:
        #     for param_group in self._optimizer.param_groups[:2]:
        #         param_group['lr'] = lr
        #     for param_group in self._optimizer.param_groups[2:]:
        #         param_group['lr'] = lr/(self.peer_num)
        # else:
        #     for param_group in self._optimizer.param_groups:
        #         param_group['lr'] = lr

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def _get_decayed_learning_rate(self):
        learning_rate = self._decayed_learning_rate(self.init_lr, self.n_current_steps, self.total_iterations)

        if self.n_warmup_steps:
            warmup_percent_done = self.n_current_steps / self.n_warmup_steps
            warmup_learning_rate = self.init_lr * warmup_percent_done

            is_warmup = self.n_current_steps < self.n_warmup_steps

            learning_rate = ((1.0 - is_warmup) * learning_rate +
                             is_warmup * warmup_learning_rate)
        return learning_rate

    def _decayed_learning_rate(self, initial_learning_rate, step, decay_steps, end_learning_rate=0.0, power=1.0):
        step = min(step, decay_steps)
        return ((initial_learning_rate - end_learning_rate) *
                (1 - step / decay_steps) ** (power)
                ) + end_learning_rate