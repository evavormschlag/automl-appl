import numpy as np
from scipy.stats import qmc
from torch.quasirandom import SobolEngine

def sobol_lrs(n_points, bound, seed):
    """
    Erzeuge n_points Lernraten mit Sobol in [bound[0], bound[1]].
    bound ist hier im linearen LR-Raum (z.B. [1e-5, 1.0]).
    """
    sobol = SobolEngine(dimension=1, scramble=True, seed=seed)

    # Sobol in [0,1]
    u = sobol.draw(n_points).numpy().reshape(-1, 1)

    # in log10(LR)-Raum skalieren
    log_min, log_max = np.log10(bound[0]), np.log10(bound[1])
    log_lrs = log_min + (log_max - log_min) * u

    # zurück in LR-Raum
    lrs = 10 ** log_lrs
    return lrs.reshape(-1)

