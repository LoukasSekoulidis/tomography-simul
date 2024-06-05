import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import tomograph


def show_phantom(size):
    Im = tomograph.phantom(size)
    plt.imshow(Im, cmap='gist_yarg',
               extent=[-1.0, 1.0, -1.0, 1.0], interpolation='nearest')
    plt.show()

# show_phantom(128)


def create_sinogram(nAngles, nSamples, angle_range=(0, np.pi)):
    """
    Funktion erzeugt Sinogramm

    :param angle_range: Winkel über die die Strahlen laufen in rad. default=(0-180 Grad)
    :param nAngles: Anzahl der Winkelschritte innerhalb der angle_range (Anzahl der Strahlenfronten)
    :param nSamples: Anzahl der Strahlen pro Winkel (Anzahl der Strahlen pro Strahlenfront)

    :return: Tuple sinogram matrix, Strahlstartpunkte, Strahlrichtungen
    """
    # rp three dimension:  Anzahl der Winkel | Anzhal der Strahlen je Winkel | x-y-Position je Strahl
    rp = np.zeros((nAngles, nSamples, 2))
    rd = np.zeros((nAngles, 2))
    steps = angle_range[1] / nAngles
    spaces = np.linspace(-1, 1, nSamples)
    sinogram = np.zeros((nAngles, nSamples))

    for i in range(0, nAngles):
        currentAngle = steps * i
        x = np.cos(currentAngle)
        y = np.sin(currentAngle)
        rd[i] = np.array([-x, -y])

        for j in range(0, nSamples):
            rp[i, j] = np.array([x, y]) + spaces[j] * np.array([-y, x])
            sinogram[i, j] = tomograph.trace(rp[i, j], rd[i])

    return sinogram, rp, rd


# ---------------------------------------------
# Main Programablauf:
# ---------------------------------------------
gridsizes = [32],  # 128, 256]
# plot mit unterfigures
fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=len(gridsizes))
# Für alle Gridsizes:

for i, ng in enumerate(gridsizes):
    print("GRID: ", ng)
    nGrid = ng

    nSamples = 2 * nGrid[i]
    nAngles = 2 * nGrid[i]
    Matrix = create_sinogram(nAngles, nSamples)

    b = np.log(np.ravel(Matrix[0]))
    rp = Matrix[1]
    rd = Matrix[2]

    ax0.imshow(Matrix[0], cmap="gist_yarg")

    A = np.zeros((len(b), ng[i]**2))
    for j in range(0, nAngles):
        rays = tomograph.grid_intersect(nGrid[i], rp[j], rd[j])
        I = rays[0]
        G = rays[1]
        dt = rays[2]

        A[j * nSamples + I, G] = dt

    AtA = A.T @ A
    Atb = A.T @ b

    x = np.linalg.solve(AtA, Atb)

    xShaped = x.reshape(ng[i], ng[i])

    ax1.imshow(xShaped, cmap="gist_yarg")

    plt.savefig('tg_fig32.png', bbox_inches='tight')
    plt.show()
