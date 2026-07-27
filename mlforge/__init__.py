"""An educational machine learning framework built entirely from scratch."""

from .activations import Sigmoid
from .core.parameter import Parameter
from .core.linear import Neuron
from .losses.mse import calculate_mse
from .models.linear_regression import LinearRegression
from .models.logistic_regression import LogisticRegression
from .optimizers import GradientDescent
from .training.trainer import Trainer

__all__ = [
    "Sigmoid",
    "Parameter",
    "Neuron",
    "calculate_mse",
    "LinearRegression",
    "LogisticRegression",
    "GradientDescent",
    "Trainer",
]

__version__ = "0.1.0"
