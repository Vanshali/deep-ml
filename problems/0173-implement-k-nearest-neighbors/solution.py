import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    pass
    neighbors = []
    for i in range(len(points)):
        dist = np.sqrt(sum(np.square(a-b) for a,b in zip(points[i], query_point)))
        neighbors.append([points[i], dist])
    neighbors = sorted(neighbors, key = lambda x: (x[1], points.index))
    res = [neighbors[i][0] for i in range(k)]

    return res
       