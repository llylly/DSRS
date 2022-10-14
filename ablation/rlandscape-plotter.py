import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

sigma = 0.5

in_file = 'r-landscape'

def meshization(X, Y, Z):
    rows = len(set(X))
    print(rows)
    cols = max([len([None for j in X if j == i]) for i in X])
    print(cols)
    DX, DY, DZ = np.zeros((rows, cols)), np.zeros((rows, cols)), np.zeros((rows, cols))
    prev_i = None
    px = -1
    for i, item in enumerate(X):
        if item != prev_i:
            if px >= 0  and py < cols - 1:
                for j in range(py + 1, cols):
                    DX[px, j] = DX[px, j-1]
                    DY[px, j] = DY[px, j-1]
                    DZ[px, j] = DZ[px, j-1]
            px += 1
            py = 0
        else:
            py += 1
        DX[px, py] = X[i]
        DY[px, py] = Y[i]
        DZ[px, py] = Z[i]
        prev_i = X[i]
    if py < cols - 1:
        for j in range(py + 1, cols):
            DX[px, j] = DX[px, j-1]
            DY[px, j] = DY[px, j-1]
            DZ[px, j] = DZ[px, j-1]
    return DX, DY, DZ

if __name__ == '__main__':
    Xs, Ys, Zs = list(), list(), list()
    with open(in_file + '.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            x, y, z = line.split(' ')
            x, y, z = x.strip(), y.strip(), z.strip()
            if z != 'None':
                x, y, z = float(x), float(y), float(z)
                Xs.append(x)
                Ys.append(y)
                Zs.append(z)

    Xs, Yz, Zs = np.array(Xs), np.array(Ys), np.array(Zs)

    DXs, DYs, DZs = meshization(Xs, Ys, Zs)
    BZs = np.zeros_like(DZs)
    for i,item in enumerate(DXs):
        BZs[i,:] = norm.ppf(item) * sigma

    fig = plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.0, bottom=0.1, right=0.97, top=1.05)
    # plt.subplots_adjust(left=0.135, bottom=0.15, right=1.0, top=0.95)


    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(DXs[:,1:-1], DYs[:,1:-1], DZs[:,1:-1], cmap=cm.coolwarm, vmin=0., vmax=1.5,
                           linewidth=0, antialiased=True, alpha=1.0, label='Neyman-Pearson')
    ax.set_xlabel('$P_A$')
    ax.set_ylabel('$Q_A$')
    ax.set_zlabel('$r$')
    ax.set_xticks(np.linspace(0.5, 1.0, num=3))
    ax.set_yticks(np.linspace(0.0, 1.0, num=3)[1:])
    ax.set_zticks(np.linspace(0.0, 1.5, num=4))
    ax.set_ylim(0.25, 1.0)
    ax.set_zlim(0, 1.5)
    ax.view_init(azim=115, elev=30)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax.plot_surface(DXs[:,1:-1], DYs[:,1:-1], BZs[:,1:-1], cmap=cm.coolwarm, vmin=0., vmax=1.5,
                            linewidth=0, antialiased=True, alpha=1.0, label='DSRS')
    ax.set_xlabel('$P_A$')
    ax.set_ylabel('$Q_A$')
    ax.set_zlabel('$r$')
    ax.set_xticks(np.linspace(0.5, 1.0, num=3))
    ax.set_yticks(np.linspace(0.0, 1.0, num=3)[1:])
    ax.set_zticks(np.linspace(0.0, 1.5, num=4))
    ax.set_ylim(0.25, 1.0)
    ax.set_zlim(0, 1.5)
    ax.view_init(azim=115, elev=30)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # fig.colorbar(surf2, shrink=0.5, aspect=5)
    # fig.legend()

    # c = plt.pcolormesh(DXs, DYs, DZs, cmap='RdBu', vmin=0., vmax=1.)
    # fig.colorbar(c)



    # plt.show()
    plt.savefig('rlandscape.pdf')
