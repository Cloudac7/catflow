import os
import json

import dpdata
import numpy as np
from miko.tesla import DPTask


class TrainModel(object):
    def __init__(self, path):
        self.path = path

