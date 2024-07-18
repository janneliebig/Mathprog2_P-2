from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from generate_datasets import Generate, DATA_TYPE

"""
Includes functions kmeans, ward, dbscan and gaussian_mixture to cluster a dataset from DATA_TYPE 'type'.
"""
class dta_Cluster:
    # implemented by Janne Liebig
    def circles(gen: Generate):
        X=gen.circlesX
        kmeans = KMeans(n_clusters=2, random_state=42)
        y_kmeans = kmeans.fit_predict(X)
        gen.circlesY = y_kmeans

    # implemented by Janne Liebig
    """
    Clusters dataset from type 'type' with K-Means-algorithm.
    """
    def kmeans(gen: Generate, type:DATA_TYPE, n_clusters: int):
        X = gen.get_dataset(type)[0]
        kmeans_obj = KMeans(n_clusters=n_clusters, random_state=42)
        y_kmeans = kmeans_obj.fit_predict(X)
        gen.set_y(type=type, y=y_kmeans)

    # implemented by Janne Liebig
    """
    Clusters dataset from type 'type' with Ward's method'.
    """
    def ward(gen: Generate, type:DATA_TYPE, n_clusters : int):
        X = gen.get_dataset(type)[0]
        ward_obj = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        y_ward = ward_obj.fit_predict(X)
        gen.set_y(type=type, y=y_ward)

    # implemented by Janne Liebig
    """
    Clusters dataset from type 'type' with DBSCAN-algorithm.
    """
    def dbscan(gen : Generate, type : DATA_TYPE, eps : float):
        X = gen.get_dataset(type)[0]
        dbscan_obj = DBSCAN(eps=eps)
        y_dbscan = dbscan_obj.fit_predict(X)
        gen.set_y(type=type, y=y_dbscan)

    # implemented by Janne Liebig
    """
    Clusters dataset from type 'type' with Gaussian-Mixture-algorithm.
    """
    def gaussian_mixture(gen : Generate, type : DATA_TYPE, n_components : int):
        X = gen.get_dataset(type)[0]
        gauss_obj = GaussianMixture(n_components=n_components, covariance_type="full")
        y_gauss = gauss_obj.fit_predict(X)
        gen.set_y(type=type, y=y_gauss)

        
    