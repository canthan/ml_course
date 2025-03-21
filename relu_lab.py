import warnings
from utils.lab_utils_relu import *
from utils.autils_2 import plt_act_trio
from utils.lab_utils_common import dlc
from matplotlib.widgets import Slider
from tensorflow.keras.activations import linear, relu, sigmoid
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.style.use('./deeplearning.mplstyle')

warnings.simplefilter(action='ignore', category=UserWarning)

plt_act_trio()
