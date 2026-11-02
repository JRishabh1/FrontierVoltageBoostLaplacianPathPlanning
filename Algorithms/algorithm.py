from abc import ABC, abstractmethod
from map_generation import Map


class BaseAlgorithm(ABC):
    def __init__(self, map: Map):
        self.map = map
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self.failure_code = None
        self.failure_reason = None
        self.failure_details = {}

    def fail(self, code, reason, **details):
        """Record an observed termination condition; preserve the None API."""
        self.failure_code = code
        self.failure_reason = reason
        self.failure_details = details
        return None

    def setup(self):
        pass

    @abstractmethod
    def solve(self, visualize=False): ...

    @abstractmethod
    def result(self, visualize=False): ...
