from .config_property import Property
from .config_util import to_storage_name


class ConfigMeta(type):
    def __init__(cls, name, bases, attr_dict):
        super().__init__(name, bases, attr_dict)
        for name, attr in attr_dict.items():
            if isinstance(attr, Property):
                attr.storage_name = to_storage_name(name)


class ConfigBase(metaclass=ConfigMeta):
    pass


class ModuleConfigBase(ConfigBase):
    dropout = Property(value_type=float, default_value=0.0)


class OutputConfigBase(ConfigBase):
    pass
