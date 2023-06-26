from typing import Dict, Tuple, List
import keras
import pandas as pd
from models import utils, Func_Model


class CNN(Func_Model):
    """
    Architecture:
    a: Multihead-Attention layer (for singlehead set heads=1)
    c: 1D Convolution
    d: Dropout layer
    e: Dense layer
    f: Flatten layer
    p: 1D Max-Pooling layer
    """
    def __init__(self,
                 input_size: Tuple,
                 optimizer = keras.optimizers.Adam(),
                 loss = keras.losses.CategoricalCrossentropy(),
                 metrics = ['accuracy'],
                 params_model: Dict[str, List[Dict]] = None,
                 compile: Dict = None) -> None:

        super().__init__()
        
        if params_model is None:
            params_model = {}
        if compile is None:
            compile = {}
        
        input_lay = keras.Input(shape=input_size)
        #params_check = params_model

        architecture = list(params_model.get('architecture'))
        #params_check['architecture'] = architecure
        utils.check_params(params_model)

        index = {}
        for key in params_model.keys():
            index[key] = 0
            
        arch = []

        for k, j in enumerate(list(architecture)):
            if k == 0:
                arch, index = utils.add_layer(j, input_lay, index, params_model, arch)
            else:
                arch, index = utils.add_layer(j, arch[len(arch)-1], index, params_model, arch)
        
        self.model = keras.Model(inputs=input_lay, outputs=arch[len(arch)-1])
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **compile)

    def fit(self, train_data: pd.DataFrame, params_dataLoader: Dict = None, params_train: Dict = None):
        return super().fit(train_data, params_dataLoader, params_train)
    
    def evaluate(self, eval_data: pd.DataFrame, params_dataLoader: Dict = None, params_eval: Dict = None):
        return super().evaluate(eval_data, params_dataLoader, params_eval)

    def predict(self, pred_data, params_dataLoader: Dict = None, params_predict: Dict = None):
        return super().predict(pred_data, params_dataLoader, params_predict)
    
    def summary(self):
        return super().summary()    
