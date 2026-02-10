from fetch.models import Source, DiscoveredDoc


REGISTRY = {}

def register(name):
    def decorator(func):
        REGISTRY[name] = func
        return func
    return decorator


def get_handler(name):
    if name not in REGISTRY:
        raise KeyError('unknown handler:', name)
    return REGISTRY[name]


from . import nalog_about_nalog 
from . import nalog_docs       
from . import pravo_ips  

    