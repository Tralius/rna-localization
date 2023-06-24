import yaml


def read_model_file(path, padding):
    with open(path) as model_file:
        model_params = yaml.safe_load(model_file)
    model_params['param_dataLoader_train']['padding_length'] = padding
    model_params['param_dataLoader_valid']['padding_length'] = padding
    return model_params
