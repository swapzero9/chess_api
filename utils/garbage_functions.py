
from inspect import ismethod

def method_exists(instance, method):
    return hasattr(instance, method) and ismethod(getattr(instance, method))