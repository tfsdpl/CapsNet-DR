import torch
import  numpy as np
import random

seed = np.random.randint(1)
random.seed(seed)
torch.manual_seed(seed)

T_SIZE = 0.2
#PRE_PROCESSING = 'KAGGLE'
#PRE_PROCESSING = 'GRAHMS'
PRE_PROCESSING = 'CLAHE'



LR = 0.0001
BATCH = 90
EPOCHS = 30
CL = 2.0
GRID = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')