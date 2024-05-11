from .transformer import *

hp.update(lr=1e-4, weight_decay=1e-4, total_epochs=200, dropout=0.2,
          # init_div_factor=1., final_div_factor=1.,
          )
