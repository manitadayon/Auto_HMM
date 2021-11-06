
from Hidden_Markov_Model import *

class DHMM_Model:
    def __init__(self,Train_ratio,Max_state,Iter,Feat,N,T,Data,N_symb):
        self.Train_ratio=Train_ratio
        self.Test_ratio=1-Train_ratio
        self.Max_state=Max_state
        self.Iter=Iter
        self.Model=[]
        self.Feat=Feat
        self.N=N
        self.BEST=[]
        self.T=T
        self.Data=Data
        self.N_symb=N_symb
   
    def __repr__(self):
        return f'''The Model has the following configuration: 
              Model = {self.Model}
              Number of Feature = {self.Feat}
              Training Ratio = {self.Train_ratio}
              Number of Cases = {self.N}
              Number of Iterations = {self.Iter}
              Maximum Number of Hidden States = {self.Max_state}
              Number of Free Parameters = {self.num_params}
              Best Parameters are = {self.BEST}
              Length of Each Time Series is = {self.T}
        '''
    
    def Dstate_sorting(self):
        last_col=self.Model.emissionprob_[:,-1] # Choose the last column
        last_col=last_col.ravel().tolist()  
        last_col_sorted=last_col[:]   # Another copy of the list to compare the sorted and normal list
        last_col_sorted.sort(reverse=True)   #since the higher probability the better(Remember the last column corresponds to the best observations.) 
        result = [last_col.index(ii) for ii in last_col_sorted] 
        return result
    
    def AIC_BIC(self):
        AIC=[]
        BIC=[]
        Record2=[]
        self.Component2=[]
        Record_aic=np.zeros((1,2))   # The state and the score.
        Record_bic=np.zeros((1,2))
        for ii in range(2,self.Max_state+1):
            print(f'Iteration Corresponding to State {ii}')
     
            self.num_params = ii*(ii-1)+ ii*(self.N_symb-1)+(ii-1) # these parameters can be automatically selected
            # based on transmat_ and emission_prob_. In this case the self.num_params will be used after Model.

            Model=MultinomialHMM(n_components=ii,tol=pow(10,-5)).fit(self.Train_Data,self.Len)
            AIC.append(-2 * Model.score(self.Train_Data) + 2 * self.num_params)
            BIC.append(-2 * Model.score(self.Train_Data) +  self.num_params * np.log(self.Train_Data.shape[0]))

        Temp1=np.argmin(AIC)  
        opt_state=Temp1+2     # This is because 1 state is not possible and if np.argmin is equal to 0 this means we have 2 states.
        Record_aic[0,:]=np.array([opt_state,min(AIC)])
        Temp2=np.argmin(BIC)
        opt_state=Temp2+2     # This is because 1 state is not possible
        Record_bic[0,:]=np.array([opt_state,min(BIC)])                  
        Record2.append(Record_aic)
        Record2.append(Record_bic)

        self.Component2.append(Record2)
        Record2=[]
    
    def Best_BIC(self):
        Hold_bic=[]
        Hold_state=[]
        Hold_bic.append(self.Component2[0][1][0][1])#      
        Hold_state.append(self.Component2[0][1][0][0])
        
        self.BEST.append(Hold_state[Hold_bic.index(min(Hold_bic))]) # Corresponds to the best state.

    def Viterbi_list(self):   # Test_data is list of numpy arrays.
        self.traj=[]
    
        mapping=self.Dstate_sorting()   # what this one does is the order of elements in descending order

        Temp=mapping[:]
        for ii in range(self.Test_data.shape[0]):
            count10=1000
            seq1=self.Test_data[ii,:].reshape((-1,self.Feat))
            States_Viterbi=self.Model.predict(seq1)
            L=len(mapping)   # This is number of states
                    # we want the highest mastery level is mapped to the highest number.
            for kk in range(L):
                for jj in range(len(States_Viterbi)):
                    if (States_Viterbi[jj]==mapping[0]):
                        States_Viterbi[jj]=count10
                
                del mapping[0]
                count10=count10-1
            mapping=Temp[:]
        
            self.traj.append(States_Viterbi)
        count10=1000
        for mm in range(len(self.traj)):
            ML_old=list(range(count10,count10-L,-1))
            for zz in range(L):
                for vv in range(len(self.traj[mm])):
                    if(self.traj[mm][vv]==ML_old[0]):
                        self.traj[mm][vv]=ML_old[0]-(count10-L)
                del ML_old[0]


        
        # return self.traj
    
    def Best_States(self):
        self.Data_train=(self.Data.iloc[0:int(self.N*self.Train_ratio),:])
        self.Test_data=(np.array(self.Data.iloc[int(self.N*self.Train_ratio):self.N,:]))
       
        self.Len=[self.T for ii in range(0,self.Data_train.shape[0])]   # Lengths must be list
        self.Train_Data = np.array(self.Data_train).reshape((-1,1)) # Convert to numpy array with one column
        self.AIC_BIC()  # Return list of mixture components, states and BIC and AIC values
        self.Best_BIC()   # Find the best state and mixture component
        self.Model=MultinomialHMM(n_components=int(self.BEST[0]),tol=pow(10,-5)).fit(self.Train_Data,self.Len)
        self.score=self.Model.score(self.Train_Data,self.Len)
        self.Viterbi_list()
        return self.traj

class Supervised_DHMM(DHMM_Model):
    def AIC_BIC(self):
        AIC=[]
        BIC=[]
        Record2=[]
        self.Component2=[]
        Record_aic=np.zeros((1,2))   
        Record_bic=np.zeros((1,2))
        print(f'The Number of States is {self.Max_state}')   

        self.num_params = self.Max_state*(self.Max_state-1)+ self.Max_state*(self.N_symb-1)+(self.Max_state-1) # these parameters can be automatically selected
        Model=MultinomialHMM(n_components=self.Max_state,tol=pow(10,-5)).fit(self.Train_Data,self.Len)
        AIC.append(-2 * Model.score(self.Train_Data) + 2 * self.num_params)
        BIC.append(-2 * Model.score(self.Train_Data) +  self.num_params * np.log(self.Train_Data.shape[0]))
    
        Temp1=np.argmin(AIC)
        opt_state=self.Max_state     
  
        Record_aic[0,:]=np.array([opt_state,min(AIC)])
       
        Temp2=np.argmin(BIC)
        opt_state=self.Max_state    # This is because 1 state is not possible
        Record_bic[0,:]=np.array([opt_state,min(BIC)])                  
        Record2.append(Record_aic)
        Record2.append(Record_bic)
     

        self.Component2.append(Record2)
        Record2=[]
        
    def __repr__(self):
        return f'''The Model has the following configuration: 
              Model = {self.Model}
              Number of Feature = {self.Feat}
              Training  Ratio = {self.Train_ratio}
              Number of Cases = {self.N}
              Number of Iterations = {self.Iter}
              Number of Hidden States = {self.Max_state}
              Length of Each Time Series is = {self.T}
        '''
    
   
