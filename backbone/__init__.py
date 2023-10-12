from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100
from .iresnet_FPPQ import iresnet18_FPPQ, iresnet34_FPPQ, iresnet50_FPPQ, iresnet100_FPPQ

def get_model_FPPQ(name, **kwargs):
    # resnet
    if name == "r18":
        return iresnet18_FPPQ(False, **kwargs)
    elif name == "r34":
        return iresnet34_FPPQ(False, **kwargs)
    elif name == "r50":
        return iresnet50_FPPQ(False, **kwargs)
    elif name == "r100":
        return iresnet100_FPPQ(False, **kwargs)


def get_model(name, **kwargs):
    if name == "r18":
        return iresnet18(False, **kwargs)
    elif name == "r34":
        return iresnet34(False, **kwargs)
    elif name == "r50":
        return iresnet50(False, **kwargs)
    elif name == "r100":
        return iresnet100(False, **kwargs)

