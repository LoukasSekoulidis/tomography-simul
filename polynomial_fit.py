import numpy as np
import matplotlib.pyplot as plt

# Laden der gegebenen Daten d0 - d4
d0 = np.load('data/d0.npy')
d1 = np.load('data/d1.npy')
d2 = np.load('data/d2.npy')
d3 = np.load('data/d3.npy')
d4 = np.load('data/d4.npy')

x = np.linspace(-2, 2, 200)


fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(
    ncols=5, figsize=(20, 5))

ax0.scatter(x, d0, marker='.')
ax1.scatter(x, d1, marker='.')
ax2.scatter(x, d2, marker='.')
ax3.scatter(x, d3, marker='.')
ax4.scatter(x, d4, marker='.')

# Implementieren Sie ein Funktion, die gegeben den x-Werten und dem Funktiongrad
# die Matrix A aufstellt.


def create_matrix(x, degree):
    retMatrix = np.zeros(len(x) * (degree + 1)).reshape(len(x), degree + 1)
    for i in range(0, len(x)):
        temp = degree
        for j in range(0, degree + 1):
            retMatrix[i, j] = x[i] ** temp  # change i to x[i]
            temp -= 1
        temp = degree
    return retMatrix


def calc_bestPoly(data):
    savedPolys = np.zeros(20 * 200).reshape(20, 200)
    distances = np.zeros(20)

    for i in range(1, 21):
        # create Matrix for various Degrees
        X = create_matrix(x, i)
        XtX = X.T @ X
        Xtb = X.T @ data
        # calculate function Variables
        calc = np.linalg.solve(XtX, Xtb)
        # define function
        poly = np.poly1d(calc)

        sumDistance = 1

        for j in range(0, len(x)):
            resJ = poly(x[j])

            savedPolys[i - 1, j] = resJ

            sumDistance += np.abs(resJ - data[j])
        # genauigkeit veringern um bei kleinen Differenzen das Polynom mit dem kleineren Index zu bekommen! (?) schlechte Lösung
        distances[i - 1] = round(sumDistance, -2)

    smallestE = np.argmin(distances)
    print('Polynom: ', smallestE + 1)
    print('Distances: \n', distances)
    return savedPolys[smallestE]


res0 = calc_bestPoly(d0)
res1 = calc_bestPoly(d1)
res2 = calc_bestPoly(d2)
res3 = calc_bestPoly(d3)
res4 = calc_bestPoly(d4)

ax0.plot(x, res0, '-r')
ax1.plot(x, res1, '-r')
ax2.plot(x, res2, '-r')
ax3.plot(x, res3, '-r')
ax4.plot(x, res4, '-r')

plt.savefig('poly', bbox_inches='tight')
plt.show()
