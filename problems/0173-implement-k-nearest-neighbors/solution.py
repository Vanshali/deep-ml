import numpy as np
import heapq

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
    if k < 0 or not points:
        return []

    pts = np.asarray(points)
    q = np.asarray(query_point)
    k = min(k, len(pts))

    def key(item):
        i, p = item
        p = np.asarray(p)
        dist = ((p-q)**2).sum()
        return (dist,i)
    neighbor = heapq.nsmallest(k, enumerate(pts), key=key)
    return [tuple(p) for _,p in neighbor]
       