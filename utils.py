import numpy as np

def relL2Error(reco, source):
    return np.linalg.norm(reco - source) / np.linalg.norm(source)

def relL1Error(reco, source):
    return np.linalg.norm(reco - source, 1) / np.linalg.norm(source, 1)