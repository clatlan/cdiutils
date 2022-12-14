# authors:
# Clément Atlan, c.atlan@outlook.com

import yaml
import pathlib
from typing import Dict


from cdiutils.processing.authorized_keys import AUTHORIZED_KEYS


def pretty_dict_print(dictionary):
    for k, v in dictionary.items():
        print(f"{k} = {v}")       

class ArgumentHandler:
    """
    Base class to deal with arguments that are required by scripts

    :param file_path: path of the configuration file that contains
    the arguments, str.
    :param script_type: the type of the script that the arguments will
    be parsed into, str.
    """
    def __init__(self, file_path, script_type="preprocessing", verbose=False):
        self.file_path = file_path
        if script_type not in list(AUTHORIZED_KEYS.keys()) + ["all"]:
            print("\n[ERROR] Please, provide a script_type from "
                  f"{list(AUTHORIZED_KEYS.keys())}\n")
        else:
            self.script_type = script_type

        self.raw_config = self._open_file()
        self.arguments = None
        self.verbose = verbose

    def _open_file(self):
        """open the file and return it"""
        with open(self.file_path, "r") as f:
            raw_config = f.read()
        return raw_config

    def load_arguments(self):
        extension = self._get_extension()
        if extension == ".yml" or extension == ".yaml":
            args = yaml.load(self.raw_config, Loader=yaml.FullLoader)
            if self.script_type == "all":
                self.arguments = {
                    "preprocessing": self._concatenate(
                        self._check_args(args["preprocessing"],
                                         "preprocessing"),
                        self._check_args(args["general"], "preprocessing"),
                    ),
                    "pynx": self._concatenate(
                        self._check_args(args["pynx"], "pynx"),
                        self._check_args(args["general"], "pynx")
                    ),
                    "postprocessing": self._concatenate(
                        self._check_args(args["postprocessing"],
                                           "postprocessing"),
                        self._check_args(args["general"], "postprocessing")
                    )
                }
            else:
                self.arguments = self._check_args(args, self.script_type)
            return self.arguments
        else:
            return None

    def _get_extension(self):
        """return the extension of the the file_path attribute"""
        return pathlib.Path(self.file_path).suffix

    def _check_args(self, dic, script_type):
        checked_keys = []
        for key in dic.keys():
            if key in AUTHORIZED_KEYS[script_type]:
                checked_keys.append(key)
            else:
                if self.verbose:
                    print(f"'{key}' is an unexpected key, "
                        "its value won't be considered.")
        return {key: dic[key] for key in checked_keys}
    
    def _concatenate(self, args1, args2, args3={}):
        return {**args1, **args2, **args3}

    # For now the yaml Loader already returns a dic, so not useful
    # but we may need it if we use other file format
    def to_dict(self):
        pass

    def dump(self, output_path, extension):
        pass


if __name__ == '__main__':
    config_file = "../conf/default_config.yml"
    arg_handler = ArgumentHandler(
        config_file,
        script_type="preprocessing"  # try with "postprocessing"
    )

    args = arg_handler.load_arguments()  # this can also be accessed by
    # arg_handler.arguments once load_arguments() has been computed

    print(f"The current configuration file is:\n{config_file}\n")
    print("attribute arg_handler.arguments:")
    print(arg_handler.arguments)  # or args