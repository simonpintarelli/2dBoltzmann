
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_coeffs2d(basis, C, tre=1e-10):
    """
    Keyword Arguments:
    basis --
    C     --
    """

    # minC = min(abs(C))
    # maxC = mac(abs(C))
    # vnorm = mpl.Colors.normalize(vmin=minC, vmax=maxC)
    # polynomial degree
    def deg(elem):
        k = elem.fr.k
        j = elem.fr.j
        return 2 * (np.floor(k / 2) - j) + 2 * j + (k % 2)

    X = np.array([(elem.fa.l, deg(elem)) for elem in basis.elements])
    lmax = basis.get_max_l()
    rmax = basis.get_max_r_deg()

    C = C.reshape(
        -1, )

    ind = np.abs(C) > tre
    fig = plt.scatter(
        X[ind, 0],
        X[ind, 1],
        c=np.log10(np.abs(C[ind])),
        cmap=mpl.cm.jet,
        lw=0,
        marker='s',
        s=30)
    plt.xlim((-1, lmax + 1))
    plt.ylim((-1, rmax + 1))
    plt.xlabel('$l$')
    plt.ylabel('$n$')
    return fig
