import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from .stokes import *
from .mueller import *
from .demosaicing import *
from .visualization import *
from .container import PolarizationContainer
from .io import *
from . import random
from .spectrum import spectrum_to_color
from .pbrdf import *

# Alias for old style
gammaCorrection = gamma
