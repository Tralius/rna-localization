import yaml
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_model_file(path, padding, multibranched: bool = False):
    with open(path) as model_file:
        model_params = yaml.safe_load(model_file)
    model_params['param_dataLoader_train']['padding_length'] = padding
    model_params['param_dataLoader_valid']['padding_length'] = padding
    if multibranched:
        for i in range(len(model_params['params_branches'])):
            if 'attention' in list(model_params['params_branches'][i].keys()):
                for j in range(len(model_params['params_branches'][i]['attention'])):
                    model_params['params_branches'][i]['attention'][j]['seq_len'] = padding 
    else:
        if 'attention' in list(model_params['params_model'].keys()):
            for i in range(len(model_params['params_model']['attention'])):
                model_params['params_model']['attention'][i]['seq_len'] = padding 
    return model_params

def prepare_data(colab: bool = False, path=None):
    if path is None:
        Warning('Using default local path')
        path = '~/Downloads/final_data.csv'
    np.random.seed(3)
    
    if colab:
        url = 'https://www.dropbox.com/s/hv4uau8q4wwg00k/final_data.csv?dl=1'
        data_org = pd.read_csv(url)
    else:
        data_org = pd.read_csv(path)
    
    test_data = data_org.sample(frac=0.1)
    train_data = data_org.drop(test_data.index)
        
    train_split, valid_split = train_test_split(train_data, random_state=42, test_size=0.2)
    
    return train_split, valid_split, test_data

def set_variables(name: str, max_seq_len, multibranch: bool = False):
    model_architecture_path = f'model_architectures_viz/{name}_{datetime.datetime.now().date()}.png'
    model_output_path = f'model_outputs/{name}_{datetime.datetime.now().date()}.h5'
    params_dict = read_model_file(f'model_architectures/{name}.yaml', max_seq_len, multibranch)
    
    if multibranch:
        params_dataLoader_valid = params_dict['param_dataLoader_valid']
        params_dataLoader_train = params_dict['param_dataLoader_train']
        params_branches = params_dict['params_branches']
        params_consensus = params_dict['params_consensus']
        params_train = params_dict['params_train']
        
        return model_architecture_path, model_output_path, params_dataLoader_train, params_dataLoader_valid, params_branches, params_consensus, params_train
    else:
        params_dataLoader_valid = params_dict['param_dataLoader_valid']
        params_dataLoader_train = params_dict['param_dataLoader_train']
        params_model = params_dict['params_model']
        params_train = params_dict['params_train']
        
        return model_architecture_path, model_output_path, params_dataLoader_train, params_dataLoader_valid, params_model, params_train
    
    