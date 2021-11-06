from Hidden_Markov_Model import *
from Hidden_Markov_Model.DHMM import *

import time
Start=time.time()
Train_ratio=0.8
Max_state=3
Iter=1000
Feat=1
N=2000
T=50
flag=0
N_symb=3
Path= 'Path to CSV file'
Data=pd.read_csv(Path)
Data=Data.astype(int)
First_DHMM=Supervised_DHMM(Train_ratio,Max_state,Iter,Feat,N,T,Data,N_symb)
First_DHMM.Best_States()
END=time.time()
print('Total Time Takes in seconds',END-Start)