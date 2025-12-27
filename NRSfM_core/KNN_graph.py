from scipy.spatial import distance
import numpy as np

def Graph_distance(points,k):
    closest = np.argsort(distance.squareform(distance.pdist(points.transpose())), axis=1)
    ID_graph=closest[:, 1:k + 1]

    return ID_graph