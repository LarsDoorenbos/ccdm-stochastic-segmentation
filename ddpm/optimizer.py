import torch
from torch import nn
from .lr_functions import LRFcts
from torch.utils.data.dataloader import DataLoader
import logging
LOGGER = logging.getLogger(__name__)


def build_optimizer(params: dict, model: nn.Module, feature_cond_encoder: nn.Module, train_dataloader: DataLoader, debug=False) -> dict:
    d = {'optimizer': None, 'lr_scheduler': None}  # to be returned
    parameters = model.parameters()
    
    if params["feature_cond_encoder"]["train"] and feature_cond_encoder:
        parameters = list(model.parameters()) + list(feature_cond_encoder.parameters())

    # this is the setting used throughout
    if 'optim' not in params:
        print('defaulting to adam optimiser with lr=1.0e-4')
        d['optimizer'] = torch.optim.Adam(params, lr=1.0e-4)  # default
        return d
    else:
        p_opt = params['optim']
        assert ('name' in p_opt) and ('learning_rate' in p_opt)

    if p_opt['name'] == 'SGD':
        wd = p_opt.get('weight_decay', 0.0005)  # usually only used with cnn on cityscapes
        momentum = p_opt.get('momentum', 0.9)
        d['optimizer'] = torch.optim.SGD(parameters, lr=p_opt['learning_rate'], momentum=momentum, weight_decay=wd)

    elif p_opt['name'] == 'Adam':
        d['optimizer'] = torch.optim.Adam(parameters, lr=p_opt['learning_rate'])

    elif p_opt['name'] == 'AdamW':  # usually only used with swin
        wd = p_opt.get('weight_decay', 0.01)
        betas = p_opt.get('betas', (0.9, 0.999))
        d['optimizer'] = torch.optim.AdamW(parameters, lr=p_opt['learning_rate'], betas=betas, weight_decay=wd)
    else:
        raise ValueError(f"optimizer {p_opt['name']} not recognized")

    if 'lr_function' in p_opt:
        # bs = params['batch_size']
        # polynomial learning rate requires assumes a predetermined number total steps (or equivalently epochs)
        # if no specific number of epochs was specified then use max_epochs (i.e train forever)
        epochs = p_opt.get('epochs', params['max_epochs'])
        num_batches_per_epoch = len(train_dataloader)  # fix: what happens in multigpu (wabout device-wise batch size?)
        lr_total_steps = num_batches_per_epoch * epochs

        lr_restart_steps = []

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(d['optimizer'], lr_lambda=LRFcts(p_opt, lr_total_steps,
                                                                                          lr_restart_steps))

        d['lr_scheduler'] = lr_scheduler
        LOGGER.info("lr_schedule: '{}' starting lr {} and lr_params {} over total steps {}".format(p_opt['lr_function'],
                                                                                                   p_opt['learning_rate'],
                                                                                                   p_opt['lr_params'],
                                                                                                   lr_total_steps))

        if debug:
            debug_lr_plot(lr_total_steps, lr_scheduler)
    return d


def debug_lr_plot(lr_total_steps, lr_scheduler):
    lrs = []
    for i in range(lr_total_steps):
        lrs.append(lr_scheduler.get_lr())

        if i <= lr_total_steps - 2:
            lr_scheduler.step()

    import matplotlib.pyplot as plt
    lr_funct = plt.plot(lrs)
    plt.show()
    a = 1


if __name__ == '__main__':
    par = {
        'optim': {'name': 'sgd',
                  'learning_rate': 1.0e-4,
                  'weight_decay': 0.0005,

                  }
    }