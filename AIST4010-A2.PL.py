import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.metrics.sklearns import Accuracy

from transformers import BertTokenizer, BertModel

from torchnlp.encoders import LabelEncoder
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import collate_tensors

import pandas as pd
# from test_tube import HyperOptArgumentParser
import os
import re
import requests
from tqdm.auto import tqdm
from datetime import datetime
from collections import OrderedDict
import logging as log
import numpy as np
import glob