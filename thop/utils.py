import logging

import torch
import torch.nn as nn
from count_hooks import *
import resnet_cifar
import model_adapters

from enum import Enum
import re

class ParameterType(Enum):
    CNN_WEIGHTS = 1
    CNN_BIAS = 2
    FC_WEIGHTS = 3
    FC_BIAS = 4
    BN_WEIGHT = 5
    BN_BIAS = 6
    DOWNSAMPLE_WEIGHTS = 7
    DOWNSAMPLE_BIAS = 8
    DOWNSAMPLE_BN_W = 9
    DOWNSAMPLE_BN_B = 10

def int_from_str(str):
    return list(map(int, re.findall(r'\d+', str)))

register_hooks = {
    nn.Conv2d: count_conv2d,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: count_relu,
    nn.ReLU6: count_relu,

    nn.MaxPool1d: count_maxpool,
    nn.MaxPool2d: count_maxpool,
    nn.MaxPool3d: count_maxpool,
    nn.AdaptiveMaxPool1d: count_adap_maxpool,
    nn.AdaptiveMaxPool2d: count_adap_maxpool,
    nn.AdaptiveMaxPool3d: count_adap_maxpool,
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: None,
}


def profile(model, model_adapter, input_size, custom_ops={}, device="cpu"):
    original_device = model.parameters().__next__().device
    training = model.training
    total_params = 0.

    model.eval().to(device)
    handler_collection = []

    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

        m_type = type(tensor)
        fn = None
        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        else:
            print("Not implemented for ", m_type)

        if fn is not None:
            #print("Register FLOP counter for module %s" % str(m))
            handler = tensor.register_forward_hook(fn)
            handler_collection.append(handler)
            total_params += parameters.shape.numel()


    x = torch.zeros(input_size).to(device)
    with torch.no_grad():
        model(x)

    total_ops = 0

    for name, parameters in model.named_parameters():
        param_type, tensor_index, layer_index, block_index = model_adapter.get_param_type_and_layer_index(name)
        tensor = model_adapter.get_layer(model, param_type, tensor_index, layer_index, block_index)

        if hasattr(tensor,  'total_ops'):
            total_ops += tensor.total_ops


    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    return total_ops, total_params

if __name__ == "__main__":

    model = resnet_cifar.resnet56_cifar()
    flops, p = profile(model, model_adapters.ResNetAdapter(), input_size=(1,3,32,32))