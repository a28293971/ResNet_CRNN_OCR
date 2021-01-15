from ._own import _OWN

def get_dataset(config):
    if config.DATASET.DATASET == "OWN":
        return _OWN
    else:
        raise NotImplemented()