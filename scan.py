import pandas as pd
import numpy as np
def ScanFiles():
    df=pd.read_csv('datasets_clustering/data_Mar_64.csv', index_col='Art')
    arr_Mar=df.to_numpy()
    arr_Mar=np.delete(arr_Mar,(0),0)
    arr_Mar=arr_Mar.flatten()
    arr_Mar=np.transpose(arr_Mar)
    df=pd.read_csv('datasets_clustering/data_Sha_64.csv', index_col='Art')
    arr_Sha=df.to_numpy()
    arr_Sha=np.delete(arr_Sha,(0),0)
    arr_Sha=arr_Sha.flatten()
    arr_Sha=np.transpose(arr_Sha)
    df=pd.read_csv('datasets_clustering/data_Tex_64.csv', index_col='Art')
    arr_Tex=df.to_numpy()
    arr_Tex=arr_Tex.flatten()
    arr_Tex=np.transpose(arr_Tex)  
    dta=np.stack((arr_Mar,arr_Sha,arr_Tex),axis=1)
    y=np.zeros([64,15])
    for i in range(1,100):
        state=np.ones([64,16])
        state*=i
        y=np.concatenate((y,state),axis=1)
    y=np.transpose(y)
    y=y.flatten()
    return dta,y