import os
import json

import dpdata
import numpy as np
from dpana.dpgen import DPTask


class TrainModel(object):
    def __init__(self, path):
        self.path = path

