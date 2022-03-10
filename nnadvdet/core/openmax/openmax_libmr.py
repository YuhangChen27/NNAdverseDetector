import libmr
import numpy as np
import scipy.spatial.distance as spd
from ..detector import AdverseDetector


class OpenmaxLibmr(AdverseDetector):
    def __init__(self, config):
        AdverseDetector.__init__(self, config)
