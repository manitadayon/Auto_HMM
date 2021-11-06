# I will first implement this in a form of Jupyter Notebook.
from Hidden_Markov_Model import *
from Hidden_Markov_Model.HMM import *
import time

import time
Start=time.time()
Train_ratio=0.1
Cov_Type='diag'
Max_state=3
Max_mixture=4
Iter=1000
Feat=1
N=2000
T=50
flag=1
Path='Path to CSV file'
Data=pd.read_csv(Path)
Exam_HMM=Supervised_HMM(Train_ratio,Cov_Type,Max_state,Max_mixture,Iter,Feat,N,T,Data,flag)
Exam_HMM.Best_States()
END=time.time()
print('Total Time Takes in seconds',END-Start)