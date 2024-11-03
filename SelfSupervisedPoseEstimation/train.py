# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

import wandb_logging
import numpy as np

options = MonodepthOptions()
opts = options.parse()

if opts.wandb_sweep: 
    wanb_obj = wandb_logging.wandb_logging(opts)
    wandb_config = wanb_obj.get_config()
    
def main():
    
    frac = 0.65
    learn_rate = 10**(-float(4))
    frequency = 5
    trainer = Trainer(opts, lr = learn_rate, sampling=frequency, frac = frac)
    trainer.train()

main()
    

    
    
    
