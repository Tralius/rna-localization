import tensorflow_probability as tfp
import tensorflow as tf
from keras.metrics import Metric
import numpy as np


class Pearson(Metric):

    def __init__(self, name='pearson', sample_axis=0, event_axis=None, keepdims=False, eps=1e-3,
                 return_dict: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self.sample_axis = sample_axis
        self.event_axis = event_axis
        self.keepdims = keepdims
        self.eps = eps
        self.return_dict = return_dict
        if self.return_dict:
            self.corr = {}
        else:
            self.corr = tf.zeros([9])

    def update_state(self, y_true, y_pred, sample_weight = None):
        y_true_std = tfp.stats.stddev(y_true, sample_axis=self.sample_axis, keepdims=True)
        y_pred_std = tfp.stats.stddev(y_pred, sample_axis=self.sample_axis, keepdims=True)
        
        y_true /= (y_true_std + self.eps)
        y_pred /= (y_pred_std + self.eps)
        
        result = tfp.stats.covariance(x=y_true,
                                      y=y_pred,
                                      event_axis=self.event_axis,
                                      sample_axis=self.sample_axis,
                                      keepdims=self.keepdims)

        if self.return_dict:
            res_dict = {}
            res_dict['ERM'] = result[0]
            res_dict['KDEL'] = result[1]
            res_dict['LMA'] = result[2]
            res_dict['MITO'] = result[3]
            res_dict['NES'] = result[4]
            res_dict['NIK'] = result[5]
            res_dict['NLS'] = result[6]
            res_dict['NUCP'] = result[7]
            res_dict['OMM'] = result[8]
            self.corr = res_dict
        else:
            self.corr = result

    def result(self):
        return self.corr

    def reset_states(self):
        if self.return_dict:
            self.corr = {}
        else:
            self.corr = tf.zeros([9])


def pearson(y_true, y_pred, sample_axis=0,
            event_axis=None,
            keepdims=False,
            eps=0.001):
    y_true /= (tfp.stats.stddev(y_true, sample_axis=sample_axis, keepdims=True) + eps)
    if y_pred is not None:
        y_pred /= (tfp.stats.stddev(y_pred, sample_axis=sample_axis, keepdims=True) + eps)

    return tfp.stats.covariance(
        x=y_true,
        y=y_pred,
        event_axis=event_axis,
        sample_axis=sample_axis,
        keepdims=keepdims)
