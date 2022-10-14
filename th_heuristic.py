
import numpy as np

def get_beta(pA):
    return 0.08 * (-np.log(1 - pA) - 5) + 0.6 if pA >= 0.5 else 0.5


# experimental
def get_beta2(pA):
    return max(pA - 0.1, 0.4)





# translator from beta (p) here to concrete l2 ball norm
# mainly for debug and tuning use
def T(d, k, sigma, p):
    from scipy.stats import gamma
    import numpy as np
    return np.sqrt(d / (d - 2.0 * k)) * sigma * np.sqrt(2.0 * gamma(d / 2.0 - k).ppf(p))