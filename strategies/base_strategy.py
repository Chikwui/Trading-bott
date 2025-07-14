from abc import ABC, abstractmethod
from typing import Any

class BaseStrategy(ABC):
    def __init__(self, config: Any):
        self.name = "BaseStrategy"

    @abstractmethod
    def generate_signals(self, data):
        """Return DataFrame of signals with columns ['symbol','signal','confidence']."""
        pass
