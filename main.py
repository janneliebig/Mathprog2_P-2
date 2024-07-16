import pandas as pd
import numpy as np
from generate_datasets import Generate, DATA_TYPE
import plot_datasets
import create_clusters
from runtime import Runtime, CLUSTER_ALGORITHM

def std_datasets(plot : bool, ctype : CLUSTER_ALGORITHM, runtime: Runtime):
    gen = Generate(n_samples=1000, factor=0.5, m_noise=0.05, c_noise=0.05, random_state=1000)
    
    if plot: plot_datasets.plot_std_datasets(gen)
    
    gen.scale_datasets()
    gen_plot = gen # copy for plot

    t= [[runtime.messure(gen, d, c) for d in DATA_TYPE] for c in CLUSTER_ALGORITHM]

    for dtype in DATA_TYPE:
        runtime.messure(gen_plot, dtype, ctype) 

    print("%-20s%-20s  %s" % ('DATASET', 'ALGORITHM', 'TIME'))
    for d in DATA_TYPE:
        for c in CLUSTER_ALGORITHM:
            print("%-20s%-20s: %.10f sec" % (d.name, c.name, t[c.value][d.value]))

    if plot: plot_datasets.plot_std_datasets(gen_plot)


def check_difference(type: DATA_TYPE, algorithm: int, gen: Generate):
    y= gen.get_dataset(type=type)[1]
    unique, frequenzy_base=np.unique(Generate.get_dataset(gen, type=type)[1], return_counts=True)
    if algorithm==0:
        create_clusters.dta_Cluster.kmeans(gen,type,np.size(unique))
    elif algorithm==1:
        create_clusters.dta_Cluster.ward(gen,type,np.size(unique))
    elif algorithm==2:
        create_clusters.dta_Cluster.dbscan(gen,type,np.size(unique))
    elif algorithm==3:
        create_clusters.dta_Cluster.gaussian_mixture(gen,type,np.size(unique))
    score=np.zeros(np.size(unique))
    y_new= gen.get_dataset(type=type)[1]
    final_score=np.zeros(np.size(unique))
    for i, un in enumerate(unique):
        score=np.zeros(np.size(unique))
        for x in range(np.size(y)):
            for cluster in range(np.size(unique)):    
                if y[x]==un and y_new[x]==cluster:
                    score[cluster]+=1
        final_score[i]=np.max(score)
    
    print((100*np.sum(final_score)/np.size(y)),"percent were correct for set ",type," and function ",algorithm)
    print("The correct maximum cluster size was " , np.max(frequenzy_base), " and the algorithm got " , np.max(np.unique(y_new,return_counts=True)[1]))


# main:
ctype = CLUSTER_ALGORITHM.GAUSSIAN_MIXTURE # Cluster-Algorithmus, dessen Cluster geplottet werden
runtime = Runtime(4, 4, 0.3, 4)
std_datasets(plot=True, ctype=ctype, runtime=runtime)
print(" 0 : KMEANS, 1 : WARD, 2 : DBSCAN, 3 : GAUSSIAN_MIXTURE")
for x in DATA_TYPE:
    print(x)
    for i in range(4):
        check_difference(type=x,algorithm=i,gen=Generate(1000,0.5,0.05,0.05,1000))

