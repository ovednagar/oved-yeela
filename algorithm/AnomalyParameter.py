from scipy import spatial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import decomposition as sk
from CanculationB import main as calculateB


def plotBk(B):  # pca!!
    """
    plt.plot(x_arr,reg_y, color='salmon', linewidth=1.5)
    patchC = mpatches.Patch(color='salmon', label='y = '+str(round(c,4))+'x + e')
    patchR = mpatches.Patch(color='salmon', label='r^2 = ' + str(round(r,4)))
    patchP = mpatches.Patch(color='salmon', label='p = ' + str(p_value))
    plt.legend(handles=[patchC ,patchR,patchP],fontsize = 'small',loc=2)
    plt.title("bk vectors after pca")
    plt.xlabel(x,fontsize=10)
    plt.ylabel(y,fontsize=10)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()
    """


def calculateParam(B, closeNeighbors=5):
    """
    we recieve a matrix of bk by rows (row1=b1)
    :return:list =[[param,vertix]]
    """
    dMat = spatial.distance_matrix(B, B)
    dMat = dMat.astype(float)
    np.fill_diagonal(dMat, np.inf)  # diagonal is zeros
    dim, dim = dMat.shape
    paramList = []
    for graphk in range(dim):
        sum = 0
        dMat_row=np.asarray(dMat[graphk,:])
        sorted_row=np.sort(dMat_row)
        for col in range(closeNeighbors):
            sum += sorted_row[col]
        param = 1 / sum
        paramList.append((param, graphk))
    return paramList
