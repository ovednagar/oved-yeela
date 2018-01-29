from FeturesMatrix import get_graph_dictonary
from CanculationB import main as calculateB
from AnomalyParameter import calculateParam
from addGroupsFeature import get_graph_top_groups_appearance
import matplotlib.pyplot as plt
import numpy as np


# main function to run correctly
def main():

    # initialize:
    # number of chosen features
    numPairFtrWanted = 20
    # calculate C for all graphs together/ for each graph separately
    singleC = False
    # include most popular groups vector?
    group_info = False
    # destination folder for regression pictures
    folder_name = str(numPairFtrWanted) + "ftr,singleC:" + str(singleC) + ",group info:" + str(group_info)
    # name of graph directory
    graph_type = "whatsAppByTime"
    # path of directory
    path_graph_dir = "/home/oved/Documents/yeela final project/python code git/Keren's/data/undirected/"

    ftrDict = get_graph_dictonary(path_graph_dir, graph_type)
    # import fabricated data-set for testing
    # ftrDict = TrainData.main()
    B = calculateB(ftrDict, graph_type, folder_name, numPairFtrWanted, singleC)

    # get group info
    if group_info:
        B = np.asarray(B)
        add_to_B = get_graph_top_groups_appearance()
        B = np.matrix(np.hstack((B, add_to_B)))

    # calculate distance between each gra[h to its neighbors to determine anomalies
    #  returns list of tuples (Param, graph_name)
    ParamList = calculateParam(B)
    # split tuples to x,y axis for plotting
    graph = []
    param = []
    for pair in ParamList:
        graph.append(pair[1])
        param.append(pair[0])
    plot_dots(graph, param)


def plot_dots(x_arr, y_arr):
    path = "distribution"
    plt.scatter(x_arr, y_arr, color='mediumaquamarine', marker="d", s=10)
    plt.title("parameter distribution")
    plt.xlabel("Time", fontsize=10)
    plt.ylabel("parameter", fontsize=10)
    for label, x, y in zip(x_arr, x_arr, y_arr, ):
        plt.annotate(label, xy=(x, y))
    plt.savefig(path)
    plt.clf()
    plt.close()


main()
