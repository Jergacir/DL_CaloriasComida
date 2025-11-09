"""
Modelo 1: ResNet50 + Food-101
"""

from .model import ResNet50Food101
from .dataset import Food101Dataset, prepare_food101_splits
from .config import *
