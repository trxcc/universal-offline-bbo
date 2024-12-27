from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class TaskMetadata(ABC):
    name: str
    input_dim: int

    @abstractmethod
    def get_variable_metadata(self) -> Dict:
        """Return metadata for each input variable"""
        pass

    @abstractmethod
    def to_string(self) -> List[str]:
        """Convert metadata to list of string descriptions in format 'x{i}:TYPE,[bounds/size]'"""
        pass


@dataclass
class ContinuousTaskMetadata(TaskMetadata):
    bounds: List[Tuple[float, float]]

    def get_variable_metadata(self) -> Dict:
        return {
            f"x{i}": {"type": "DOUBLE", "bounds": self.bounds[i]}
            for i in range(self.input_dim)
        }

    def to_string(self) -> List[str]:
        return [f"x{i}:DOUBLE, {list(bound)}" for i, bound in enumerate(self.bounds)]


@dataclass
class CategoricalTaskMetadata(TaskMetadata):
    n_categories: int

    def get_variable_metadata(self) -> Dict:
        return {
            f"x{i}": {"type": "CATEGORICAL", "size": self.n_categories}
            for i in range(self.input_dim)
        }

    def to_string(self) -> List[str]:
        return [f"x{i}:CATEGORICAL, {self.n_categories}" for i in range(self.input_dim)]
