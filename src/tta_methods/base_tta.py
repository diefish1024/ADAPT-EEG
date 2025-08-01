# src/tta_methods/base_tta.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from src.utils.logger import get_logger # Assuming you have a common logger utility

logger = get_logger(__name__)

class BaseTTAMethod(ABC):
    """
    Abstract base class for Test-Time Adaptation (TTA) methods.

    All concrete TTA methods (e.g., Tent, CoTTA, SHOT) must inherit from this base class
    and implement its abstract methods. This ensures that the TTA Trainer (TTATrainer)
    can work cohesively with any TTA method that conforms to this interface.
    """

    def __init__(self, model: nn.Module, device: torch.device = None, **kwargs):
        """
        Initializes the abstract base class for TTA methods.

        Args:
            model (nn.Module): The pre-trained model to be adapted.
            device (torch.device, optional): The device (CPU/GPU) on which the model will run.
                                             If not provided, it defaults to attempting CUDA, otherwise CPU.
            **kwargs: Allows subclasses to pass additional configuration parameters.
                      The base class can process or ignore these as needed.
                      Examples include: optimizer configuration, adaptation parameters, etc.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be an instance of torch.nn.Module.")

        self.model = model
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) # Move the model to the specified device

        # Store kwargs, allowing subclasses to access them without forcing
        # all possible config parameters to be defined in the base class.
        self.kwargs = kwargs 

        logger.info(f"BaseTTAMethod initialized for model on device: {self.device}")

    @abstractmethod
    def adapt(self, target_batch: tuple):
        """
        Abstract method: Adapts the model based on a batch of target domain data.
        All concrete TTA methods inheriting from this base class must implement this method.

        This method should encapsulate the core logic of the TTA process:
        1. Obtain the target domain data from the batch.
        2. Compute the adaptation loss based on the specific TTA algorithm.
        3. Perform backpropagation and update model parameters.

        Args:
            target_batch (tuple): A batch containing target domain features (and optional labels,
                                  though TTA typically operates on unlabeled data).
                                  Example: (features, labels) or (features, _).

        Returns:
            torch.Tensor: The model's output (e.g., logits, predictions) after adaptation
                          on the current batch. This return value is often used for
                          evaluation or further processing within the TTATrainer.
        """
        pass

    def __repr__(self):
        """
        Returns a string representation of the TTA method.
        """
        return f"{self.__class__.__name__}(device={self.device})"

