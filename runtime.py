import time
import create_clusters
from generate_datasets import Generate, DATA_TYPE
from enum import Enum

class CLUSTER_ALGORITHM(Enum):
    KMEANS = 0
    WARD = 1
    DBSCAN = 2
    GAUSSIAN_MIXTURE = 3

class Runtime:
    def __init__(self,k_means_clusters : int, ward_clusters : int, dbscan_eps : float, gauss_components : int):
        self.k_means_clusters = k_means_clusters
        self.ward_clusters = ward_clusters
        self.dbscan_eps = dbscan_eps
        self.gauss_components = gauss_components

    def messure(self, gen : Generate, dtype : DATA_TYPE, ctype : CLUSTER_ALGORITHM) -> float:
        start = time.time()
        if ctype == CLUSTER_ALGORITHM.KMEANS:
            create_clusters.dta_Cluster.kmeans(gen, dtype, self.k_means_clusters)
        elif ctype==CLUSTER_ALGORITHM.WARD:
            create_clusters.dta_Cluster.ward(gen, dtype, self.ward_clusters)
        elif ctype == CLUSTER_ALGORITHM.DBSCAN:
            create_clusters.dta_Cluster.dbscan(gen, dtype, self.dbscan_eps)
        elif ctype == CLUSTER_ALGORITHM.GAUSSIAN_MIXTURE:
            create_clusters.dta_Cluster.gaussian_mixture(gen, dtype, self.gauss_components)
        end = time.time()
        return end - start

