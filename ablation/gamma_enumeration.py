"""
    prove Proposition B.3 by enumeration
"""

from scipy.stats import gamma

if __name__ == '__main__':
    for half_k in range(1, 100):
        k = half_k / 2.
        dist = gamma(k)
        # print('k =', k)
        # print('GammaCDF(k)(0.98 k) =')
        # print('  ', dist.cdf(0.98 * k))
        print(k, '&', f'{dist.cdf(0.98 * k):.4f}', '\\\\')