import pandas as pd
import numpy as np
import math
from generate_datasets import Generate, DATA_TYPE
import plot_datasets
import create_clusters
from runtime import Runtime, CLUSTER_ALGORITHM


# implemented by Nils Jakobs and Janne Liebig
"""
    Plots all datasets unscaled. Then it plots datasets scaled and clustered with algorithm ctype.
    Messures runtime and prints table.
"""
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

# implemented by Nils Jakobs
def check_difference(type: DATA_TYPE, algorithm: int, gen: Generate):
    y= gen.get_dataset(type=type)[1]
    unique, frequenzy_base=np.unique(Generate.get_dataset(gen, type=type)[1], return_counts=True)
    if algorithm==0:
        create_clusters.dta_Cluster.kmeans(gen,type,np.size(unique))
    elif algorithm==1:
        create_clusters.dta_Cluster.ward(gen,type,np.size(unique))
    elif algorithm==3:
        eps=1
        gen=dbscan_cluster_to_eps(gen=gen,type=type,clusters=np.size(unique),current_eps=eps,current_step=eps)
    elif algorithm==2:
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
    return np.sum(final_score)/np.size(y)

# implemented by Nils Jakobs
def dbscan_cluster_to_eps(gen: Generate, type:DATA_TYPE,clusters: int, current_eps: float, current_step: float):
    create_clusters.dta_Cluster.dbscan(gen=gen,type=type,eps=current_eps)
    current_clusters=np.size(np.unique(gen.get_dataset(type=type)[1]))
    if int(math.log10(current_eps))%100==0 and current_eps<0.01:
        print(current_clusters, current_eps, current_step)
    if current_clusters==clusters:
        return gen
    elif current_clusters<clusters:
        return dbscan_cluster_to_eps(gen=gen, type=type, clusters=clusters, current_eps=(current_eps-current_step/2), current_step=current_step/2)
    elif current_clusters>clusters:
        return dbscan_cluster_to_eps(gen=gen, type=type, clusters=clusters, current_eps=(current_eps+current_step/2), current_step=current_step/2)

# main:
# implemented by Nils Jakobs and Janne Liebig
ctype = CLUSTER_ALGORITHM.GAUSSIAN_MIXTURE # Cluster-Algorithmus, dessen Cluster geplottet werden
runtime = Runtime(4, 4, 0.3, 4)
std_datasets(plot=False, ctype=ctype, runtime=runtime)
print(" 0 : KMEANS, 1 : WARD, 3 : DBSCAN, 2 : GAUSSIAN_MIXTURE")
options=np.zeros((4,4))
for idx,x in enumerate(DATA_TYPE):
    print(x)
    for i in range(4):
        if x != DATA_TYPE.LEAVES or i!=3:
            options[idx][i]=100*check_difference(type=x,algorithm=i,gen=Generate(1000,0.5,0.05,0.05,1000))
print(options)
