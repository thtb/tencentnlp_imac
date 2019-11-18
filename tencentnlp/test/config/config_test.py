import os
import unittest
from typing import Optional

from tencentnlp.config.config import ConfigBase
from tencentnlp.config.config_property import InputPath
from tencentnlp.config.config_property import OutputPath
from tencentnlp.config.config_property import Property


class ConfigTest(ConfigBase):
    name = Property(value_type=str, optional=True, default_value="default_name")
    weight = Property(default_value="default_name")
    input_data_path = InputPath()
    output_model_path = OutputPath(optional=True)

    def __init__(self, name: Optional[str] = "name",
                 weight: Optional[float] = 0.5,
                 input_data_path: str = None, output_data_path: str = None):
        self.name = name
        self.weight = weight
        self.input_data_path = input_data_path
        self.output_model_path = output_data_path


class PropertyTest(unittest.TestCase):
    def test_value_type(self):
        with self.assertRaises(TypeError):
            class WrongDefaultValueTypeConfigTest(ConfigBase):
                name = Property(value_type=str, default_value=0.5)

        class DefaultValueTypeConfigTest(ConfigBase):
            name = Property(value_type=str, default_value="default_name")

            def __init__(self, name: Optional[str] = "name"):
                self.name = name

        with self.assertRaises(TypeError):
            _ = DefaultValueTypeConfigTest(name=0.5)

    def test_property(self):
        # test for optional
        path = str(os.getcwd())
        # name is optional with default value
        _ = ConfigTest(name=None, input_data_path=path)
        # weight is not optional with default value
        _ = ConfigTest(weight=None, input_data_path=path)
        # input_data_path is not optional without default value
        with self.assertRaises(AttributeError):
            _ = ConfigTest(input_data_path=None)
        # output_data_path is optional without default value
        _ = ConfigTest(input_data_path=path)

    def test_input_path(self):
        # test for wrong input path
        not_exists_path = os.getcwd() + "/not_exists_path"
        with self.assertRaises(FileNotFoundError):
            _ = ConfigTest(input_data_path=not_exists_path)

    def test_output_path(self):
        temp_output_path = os.getcwd() + "/temp_output_path"
        _ = ConfigTest(input_data_path=os.getcwd(),
                       output_data_path=temp_output_path)
        self.assertTrue(os.path.exists(temp_output_path))
        os.rmdir(temp_output_path)
        self.assertFalse(os.path.exists(temp_output_path))


if __name__ == '__main__':
    unittest.main()
