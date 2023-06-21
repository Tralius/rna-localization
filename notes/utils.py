import yaml
def read_model_file(path, input_shape):
    with open(path) as model_file:
        model_params = yaml.safe_load(model_file)
    model_params['model']['conv_layers'][0]['input_shape'] = input_shape
    return model_params