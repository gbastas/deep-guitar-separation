import configparser
import os

def is_string(value):
    try:
        str(value)
        return True
    except(ValueError):
        return False

def is_float(value):
    try:
        float(value)
        return True
    except(ValueError):
        return False

def is_int(value):
    try:
        int(value)
        return True
    except(ValueError):
        return False

def is_bool(value):
    if isinstance(value, bool):
        return True
    elif isinstance(value, str):
        return value.lower() in ["true", "false"]
    return False

def is_list_of_int(value):
    try:
        [int(x) for x in value.split(", ")]
        return True
    except(ValueError):
        return False

class Constants():
    def __init__(self, config_name, workspace_folder):
        config = configparser.ConfigParser()
        config.read(config_name)
        self.workspace_folder = workspace_folder
        for section_name in config.sections():
            for key, value in config.items(section_name):
                if str(key) == 'size_of_fft':
                    setattr(self, key, 2**(int(value)))
                elif is_int(value):
                    value = int(value)
                    setattr(self, key, value)
                elif is_float(value):
                    value = float(value)
                    setattr(self, key, value)
                elif is_list_of_int(value):
                    value = [int(x) for x in value.split(", ")]
                    setattr(self, key, value)
                elif is_bool(value):
                    value = {"true": True, "false": False}.get(value.lower())
                    setattr(self, key, value)
                elif is_string(value):
                    setattr(self, key, value)

                else:
                    raise ValueError("constants.ini arguement with name " + str(key) + "is of innapropriate value."+
                         "chansge value or suplement Constants class in constants_parser.py")    
        self.track_path = os.path.join(workspace_folder, self.track_path)
        # Path were training data are stored
        self.training_path = os.path.join(workspace_folder, self.training_path)
        self.result_path = workspace_folder + '/results/'
        # Path were annotations are stored
        self.annos_path = os.path.join(workspace_folder, self.annos_path)

        # self.train_frets_copy = self.train_frets.copy()
 