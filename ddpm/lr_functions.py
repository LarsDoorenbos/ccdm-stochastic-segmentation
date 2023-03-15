import numpy as np
from collections import OrderedDict


class LRFcts:
    def __init__(self, config: dict, lr_total_steps: int,  lr_restart_steps: list):
        self.base_lr = config['learning_rate']
        self.lr_total_steps = lr_total_steps
        self.lr_fct = config['lr_function']
        self.batchwise = config.get('lr_batchwise', True)

        self.lr_params = dict()
        if 'lr_params' in config:
            self.lr_params = config['lr_params']

        self.uses_restarts = True
        if len(lr_restart_steps)== 0:
            self.uses_restarts = False
        # Restart epochs, and base values
        self.lr_restarts = lr_restart_steps
        restart_vals = config.get('lr_restart_vals', 1)
        if 0 not in self.lr_restarts:
            self.lr_restarts.insert(0, 0)
        self.lr_restart_vals = [1]
        if isinstance(restart_vals, float) or isinstance(restart_vals, int):
            # Base LR value reduced to fraction every restart, end set to 0
            for i in range(1, len(self.lr_restarts)):
                self.lr_restart_vals.append(self.lr_restart_vals[i - 1] * restart_vals)
        elif isinstance(restart_vals, list):
            assert len(restart_vals) == len(config['lr_restarts']) - 1, \
                "Value Error: lr_restart_vals is list, but not the same length as lr_restarts"
            self.lr_restart_vals.extend(restart_vals)
        if lr_total_steps not in self.lr_restarts:
            self.lr_restarts.append(lr_total_steps)
            self.lr_restart_vals.append(0)
        self.lr_restarts = np.array(self.lr_restarts)
        self.lr_restart_vals = np.array(self.lr_restart_vals)

        # Length of each restart
        self.restart_lengths = np.ones_like(self.lr_restarts)
        self.restart_lengths[:-1] = self.lr_restarts[1:] - self.lr_restarts[:-1]

        # Current restart position
        self.curr_restart = len(self.lr_restarts) - np.argmax((np.arange(lr_total_steps + 1)[:, np.newaxis] >= self.lr_restarts)[:, ::-1], axis=1) - 1
        if self.lr_fct == 'piecewise_static':
            #  example entry in config['train']["piecewise_static_schedule"]: [[40,1],[50,0.1]]
            # if s<=40 ==> lr = learning_rate * 1 elif s<=50 ==> lr = learning_rate * 0.1
            assert(len(self.lr_restarts) == 2), 'with piecewise_static lr schedule lr_restarts must be empty list' \
                                              ' instead got {}'.format(self.lr_restarts)
            assert 'piecewise_static_schedule' in self.lr_params
            assert isinstance(self.lr_params['piecewise_static_schedule'], list)
            assert self.lr_params['piecewise_static_schedule'][-1][0] == config['epochs'], \
                "piecewise_static_schedule's last phase must have first element equal to number of epochs " \
                "instead got: {} and {} respectively".format(config['piecewise_static_schedule'][-1][0], config['epochs'])

            piecewise_static_schedule = self.lr_params['piecewise_static_schedule']
            self.piecewise_static_schedule = OrderedDict() # this is essential, it has to be an ordered dict
            phase_prev = 0
            for phase in piecewise_static_schedule: # get ordered dict from list
                assert phase_prev < phase[0], ' piecewise_static_schedule must have increasing first elements per phase' \
                                              ' instead got phase_prev {} and phase {}'.format(phase_prev, phase[0])
                self.piecewise_static_schedule[phase[0]] = phase[1]

    def __call__(self, step: int):
        if self.uses_restarts:
            steps_since_restart = step - self.lr_restarts[self.curr_restart[step]]
            base_val = self.lr_restart_vals[self.curr_restart[step]]
            if self.lr_fct == 'static':
                return base_val
            elif self.lr_fct == 'piecewise_static':
                return self.piecewise_static(step)
            elif self.lr_fct == 'exponential':
                return self.lr_exponential(base_val, steps_since_restart)
            elif self.lr_fct == 'polynomial':
                steps_in_restart = self.restart_lengths[self.curr_restart[step]]
                return self.lr_polynomial(base_val, steps_since_restart, steps_in_restart)
            elif self.lr_fct == 'cosine':
                steps_in_restart = self.restart_lengths[self.curr_restart[step]]
                return self.lr_cosine(base_val, steps_since_restart, steps_in_restart)
            else:
                ValueError("Learning rate schedule '{}' not recognised.".format(self.lr_fct))
        else:
            # hacky for now, remove the lr_restarts code to be used only if lr_restarts are used
            base_val = 1.0
            if (step>self.lr_total_steps):
                print(f'warning learning rate scheduler at step {step} exceeds expected lr_total_steps {self.lr_total_steps}')
            if self.lr_fct == 'exponential':
                return self.lr_exponential(base_val, step)
            elif self.lr_fct == 'polynomial':
                return self.lr_polynomial(base_val, step, self.lr_total_steps)
            elif self.lr_fct == 'linear-warmup-polynomial':
                assert  'warmup_iters' in self.lr_params \
                        and 'warmup_rate' in self.lr_params, f'lr_params must be passed via config as dict with keys ' \
                                                             f'warmup_iters and warmup_rate instead got {self.lr_params}'
                if step <= self.lr_params['warmup_iters']-1:
                    return self.linear_warmup(step)
                else:
                    return self.lr_polynomial(base_val, step, self.lr_total_steps)
            else:
                ValueError("Learning rate schedule without restarts'{}' not recognised.".format(self.lr_fct))

    def piecewise_static(self, step):
        # important this only works if self.piecewise_static_schedule is an ordered dict!
        for phase_end in self.piecewise_static_schedule.keys():
            lr = self.piecewise_static_schedule[phase_end]
            if step <= phase_end:
                return lr

    def linear_warmup(self, step: int):
        # step + 1 to account for step = 0 ... warmup_iters -1

        lr = 1 - (1 - (step+1) / self.lr_params['warmup_iters']) * (1 - self.lr_params['warmup_rate'])
        # warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
        return lr

    def lr_exponential(self, base_val: float, steps_current: int):
        gamma = .98 if self.lr_params is None else self.lr_params
        lr = base_val * gamma ** steps_current
        return lr

    def lr_polynomial(self, base_val: float, steps_current: int, max_steps: int):
        # max_steps - 1 to account for step = 0 ... max_steps -1
        # power = .9 if 'power' in self.lr_params else self.lr_params['power']
        power = self.lr_params.get('power', 1.0)
        # min_lr = self.lr_params['min_lr'] if 'min_lr' in self.lr_params else 0.0
        min_lr = self.lr_params.get('min_lr', 0.0)
        assert min_lr >= 0
        if min_lr == 0:
            min_base_val = 0
        else:
            min_base_val = min_lr / self.base_lr
        coeff = (1 - steps_current / (max_steps-1)) ** power
        lr = (base_val - min_base_val) * coeff + min_base_val
        # return lr
        return max(lr, min_base_val)

    def lr_cosine(self, base_val, steps_current, max_steps):
        lr = base_val * 0.5 * (1. + np.cos(np.pi * steps_current / max_steps))
        return lr


if __name__ == '__main__':
    def lr_exponential(base_val: float, steps_since_restart: int, steps_in_restart=None, gamma: int = .98):
        lr = base_val * gamma ** steps_since_restart
        return lr

    def lr_cosine(base_val, steps_since_restart, steps_in_restart):
        lr = base_val * 0.5 * (1. + np.cos(np.pi * steps_since_restart / steps_in_restart))
        return lr


    def linear_warmup(step: int):
        base_lr = 0.0001
        rate = 1e-6
        # step + 1 to account for step = 0 ... warmup_iters -1
        lr = 1 - (1 - (step+1) / 1500) * (1 - rate)
        # warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
        return lr * base_lr

    def lr_polynomial( base_val: float, steps_current: int, max_steps: int):
        # max_steps - 1 to account for step = 0 ... max_steps -1
        power = 1.0
        min_lr = 0.0
        coeff = (1 - steps_current / (max_steps-1)) ** power
        lr = (base_val - min_lr) * coeff + min_lr
        return lr


    def linear_warmup_then_poly(step:int, total_steps):
        if step <= 1500 - 1:
            return linear_warmup(step)
        else:
            return lr_polynomial(0.0001, step, total_steps)




    # lr_start = 0.0001
    # T = 100
    # lrs = [lr_cosine(lr_start, step, T) for step in range(T)]
    # lrs_exp = [lr_exponential(lr_start, step % (T//4), T//4) for step in range(T)]
    #
    #
    #
    import matplotlib.pyplot as plt
    # plt.plot(lrs)
    # plt.plot(lrs_exp)
    T = 160401
    lrs_exp = [linear_warmup_then_poly(step, T) for step in range(T)]
    plt.plot(lrs_exp)
    plt.show()
    a = 1