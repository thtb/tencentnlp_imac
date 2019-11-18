import abc
import os
from typing import Any, List

from .config_util import to_real_name


class Property(abc.ABC):
    def __init__(self, value_type=None, optional: bool = False,
                 default_value: Any = None, candidates: List[Any] = None):
        self.storage_name = ""
        self.optional = optional
        self.default_value = default_value
        self.value_type = value_type
        self.candidates = candidates
        if self.value_type is not None and default_value is not None:
            if type(default_value) != self.value_type:
                raise TypeError(
                    "Wrong default value {} for value type: {}".format(
                        default_value, self.value_type))
            # TODO: check the data type in list
        if default_value is not None and self.candidates is not None:
            if default_value not in self.candidates:
                raise ValueError(
                    "Default value {} is not in candidates {}".format(
                        default_value, ",".join(self.candidates)))

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            return self
        return getattr(instance, self.storage_name)

    def _check(self, value: Any) -> Any:
        if not self.optional and value is None and self.default_value is None:
            real_name = to_real_name(self.storage_name)
            raise AttributeError(
                "{} is not OPTIONAL, should not be NONE!".format(real_name))
        if value is None:
            value = self.default_value

        if value is not None and self.candidates is not None:
            if value not in self.candidates:
                raise ValueError(
                    "Value {} is not in candidates {}".format(
                        value, ",".join(self.candidates)))

        if self.value_type is not None and value is not None:
            if type(value) != self.value_type:
                raise TypeError(
                    "Wrong default value {} for value type: {}".format(
                        value, repr(self.value_type)))

        return value

    def __set__(self, instance: Any, value: Any):
        value = self._check(value)
        setattr(instance, self.storage_name, value)


class InputPath(Property):
    def __set__(self, instance: Any, value: Any):
        value = self._check(value)
        if not os.path.exists(value):
            raise FileNotFoundError("{} is not found".format(value))
        setattr(instance, self.storage_name, value)


class OutputPath(Property):
    def __set__(self, instance: Any, value: Any):
        value = self._check(value)
        # TODO: Check if path is valid to create
        if value is not None:
            os.makedirs(value)
        setattr(instance, self.storage_name, value)
