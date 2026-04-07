# Compatibility aliases for old style names
from .pbrdf import load as load_pbsdf
from .pbrdf import save as save_pbsdf
from .stokes import stokes_to_aolp as cvtStokesToAoLP
from .stokes import stokes_to_docp as cvtStokesToDoCP
from .stokes import stokes_to_dolp as cvtStokesToDoLP
from .stokes import stokes_to_dop as cvtStokesToDoP
from .stokes import stokes_to_eang as cvtStokesToEllipticityAngle
from .vis import colorize as applyColorMap
from .vis import colorize_aolp as applyColorToAoLP
from .vis import colorize_cop as applyColorToCoP
from .vis import colorize_dop as applyColorToDoP
from .vis import colorize_top as applyColorToToP
