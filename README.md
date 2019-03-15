# THOP: PyTorch-OpCounter

This is a fork of https://github.com/Lyken17/pytorch-OpCounter but with handling of custom architecture.

## How to install 
    
* Using GitHub (always latest)
    
    `pip install --upgrade git+https://github.com/Natlem/pytorch-OpCounter.git`
    
## How to use 
* Basic usage 
    ```python
    from torchvision.models import resnet50
    from thop import profile
    model = resnet50()
    flops, params = profile(model, input_size=(1, 3, 224,224))
    ```    

* Define the rule for 3rd party module.
    
    ```python
    class YourModule(nn.Module):
        # your definition
    def count_your_model(model, x, y):
        # your rule here
    flops, params = profile(model, input_size=(1, 3, 224,224), 
                            custom_ops={YourModule: count_your_model})
    ```
* Define models or adapter.
    You can define your own model in custom_models and your adapter for that model in model_adapters
    
    
