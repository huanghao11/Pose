from .mlcrnet import MLCRNet

__factory = {
    'MLCRNet': MLCRNet,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)