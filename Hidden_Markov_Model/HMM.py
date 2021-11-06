from Hidden_Markov_Model import *

class HMM_Model():
    def __init__(self,Train_ratio,Cov_Type,Max_state,Max_mixture,Iter,Feat,N,T,Data,flag):
        self.Train_ratio=Train_ratio
        self.Test_ratio=1-Train_ratio
        self.Cov_Type=Cov_Type
        self.Max_state=Max_state
        self.Max_mixture=Max_mixture
        self.Iter=Iter
        self.Model=[]
        self.Feat=Feat
        self.N=N
        self.BEST=[]
        self.T=T
        self.Data=Data
        self.flag=flag  # Whether good state is sorted normally or in reverse
    # @staticmethod
    def State_sorting(self):
        list_mean=np.mean(self.Model.means_,axis=1)   
        list_mean=list_mean.ravel().tolist()  
        list_mean_sorted=list_mean[:]   
        if(self.flag==1):   
            list_mean_sorted.sort(reverse=True)   # Sort the list 
        elif(self.flag==0):
            list_mean_sorted.sort()   # Sort the list
        result = [list_mean.index(i) for i in list_mean_sorted] 
        return result

    def __repr__(self):
        return f'''The Model has the following configuration: 
              Model = {self.Model}
              Number of Feature = {self.Feat}
              Train to Test Ratio = {self.Train_ratio}
              Number of Cases = {self.N}
              Covariance Type = {self.Cov_Type}
              Number of Iterations = {self.Iter}
              Maximum Number of Hidden States = {self.Max_state}
              Maximum Number of Mixture Components = {self.Max_mixture}
              Number of Free Parameters = {self.num_params}
              Best Parameters are = {self.BEST}
              Length of Each Time Series is = {self.T}
        '''
    
    def Viterbi_list(self):   # Test_data is list of numpy arrays.
        self.traj=[]
        mapping=self.State_sorting()   # what this one does is the order of elements in descending order

        Temp=mapping[:]
        for ii in range(self.Test_data.shape[0]):
            count10=1000
            seq1=self.Test_data[ii,:].reshape((-1,self.Feat))
            States_Viterbi=self.Model.predict(seq1)
            L=len(mapping)   # This is number of states
                   
            for kk in range(L):
                for jj in range(len(States_Viterbi)):
        #             for kk in range(L):
                    if (States_Viterbi[jj]==mapping[0]):
                        States_Viterbi[jj]=count10
                
                del mapping[0]
                count10=count10-1
            mapping=Temp[:]
        
            self.traj.append(States_Viterbi)
        count10=1000
    #     ML_old=list(range(count10,count10-L,-1))   # from 1000 to 996 for example with 5 states 
        for mm in range(len(self.traj)):
    #         import pdb 
    #         pdb.set_trace()
            ML_old=list(range(count10,count10-L,-1))
            for zz in range(L):
                for vv in range(len(self.traj[mm])):
                    if(self.traj[mm][vv]==ML_old[0]):
                        self.traj[mm][vv]=ML_old[0]-(count10-L)
                del ML_old[0]


        
        # return self.traj
        
    def AIC_BIC(self):
        # Cov_Type=['diag','spherical','full','tied']
        # We compute the log likelihood and from that we calculate the aic and bic from that and choose the one with the 
        # lowest aic and bic
        # Len1=[n_section[0] for i in range((Chapter1_Train.shape[0]//n_section[0]))]
    #     Len1=(section_completed[0:Train_length,1].tolist())    # Lengths must be list
    #     Len2=[int(i) for i in Len1]
        ### Len is the length of each individual time series 
        ### Train_data is the training data
        ### N_state is maximum number of state to sweep
        ### N_mixture is the maximum of mixture components to sweep (used for GMM)
        ### feature is dimension of the time series. Univariate time series, feature is 1.
        AIC=[]
        BIC=[]
        Record2=[]
        self.Component2=[]
        Init=[26,64,75,100]
        Record_aic=np.zeros((len(Init),2))   
        Record_bic=np.zeros((len(Init),2))   
       
        for ii in range(1,self.Max_mixture+1):
        
            Record_aic=np.zeros((1,2))   
            Record_bic=np.zeros((1,2))
            print('One mixture component is over',ii)
      
            AIC.clear()
            BIC.clear()
            for jj in range(2,self.Max_state+1):
 
                self.num_params = jj*(jj-1)+ jj*(ii-1)+(ii*jj)*self.Feat+(jj*ii*self.Feat)
        #                
                Model=GMMHMM(n_components=jj,n_mix=ii,params='stmcw', init_params='stmcw',tol=pow(10,-5),n_iter=self.Iter).fit(self.Train_Data,self.Len)
                AIC.append(-2 * Model.score(self.Train_Data) + 2 * self.num_params)
                BIC.append(-2 * Model.score(self.Train_Data) +  self.num_params * np.log(self.Train_Data.shape[0]))
    
            Temp1=np.argmin(AIC)
            opt_state=Temp1+2     
        
            Record_aic[0,:]=np.array([opt_state,min(AIC)])
            Temp2=np.argmin(BIC)
            opt_state=Temp2+2     
            Record_bic[0,:]=np.array([opt_state,min(BIC)])                  
            Record2.append(Record_aic)
            Record2.append(Record_bic)
    
            self.Component2.append(Record2)
            Record2=[]
        
    def Best_BIC(self):
        Hold_bic=[]
        Hold_state=[]
        for ii in range(len(self.Component2)):
            Hold_bic.append(self.Component2[ii][1][0][1])#     print(count13)   # This derermines the mixture component 
            Hold_state.append(self.Component2[ii][1][0][0])
        self.BEST.append(Hold_bic.index(min(Hold_bic))+1)
       
        self.BEST.append(Hold_state[Hold_bic.index(min(Hold_bic))])
    
    def Best_States(self):
        self.Data_train=(self.Data.iloc[0:int(self.N*self.Train_ratio),:])
        self.Test_data=(np.array(self.Data.iloc[int(self.N*self.Train_ratio):self.N,:]))
       
        self.Len=[self.T for ii in range(0,self.Data_train.shape[0])]   # Lengths must be list
        self.Train_Data = np.array(self.Data_train).reshape((-1,1)) # Convert to numpy array with one column
        self.AIC_BIC()  
        self.Best_BIC()  
      
        self.Model=GMMHMM(n_components=int(self.BEST[1]),n_mix=int(self.BEST[0]),covariance_type=self.Cov_Type,params='stmcw', init_params='stmcw',tol=pow(10,-5),n_iter=self.Iter).fit(self.Train_Data,self.Len)
        self.score=self.Model.score(self.Train_Data,self.Len)
        self.Viterbi_list()
        return self.traj

class Supervised_HMM(HMM_Model):
    def AIC_BIC(self):
        AIC=[]
        BIC=[]
        Record2=[]
        self.Component2=[]
        Init=[26,64,75,100]
        Record_aic=np.zeros((len(Init),2))   
        Record_bic=np.zeros((len(Init),2))   
       
        
        for ii in range(1,self.Max_mixture+1):
        # for ii in range(len(n_component)):
            Record_aic=np.zeros((1,2))   
            Record_bic=np.zeros((1,2))
            print('One mixture component is over',ii)
        #     for mm in range(len(Init)):
            AIC.clear()
            BIC.clear()
            # for jj in range(2,self.Max_state+1):
            # self.num_params=self.Max_state*(self.Max_state-1)+ self.Max_state*(ii-1)+(ii*self.Max_state)*self.Feat+((self.Feat**2+self.Feat)/2)*ii*self.Max_state  # Full Covariance
            self.num_params = self.Max_state*(self.Max_state-1)+ self.Max_state*(ii-1)+(ii*self.Max_state)*self.Feat+(self.Max_state*ii*self.Feat)  # Diagonal
            Model=GMMHMM(n_components=self.Max_state,n_mix=ii,covariance_type=self.Cov_Type,params='stmcw', init_params='stmcw',tol=pow(10,-5),n_iter=self.Iter).fit(self.Train_Data,self.Len)
            AIC.append(-2 * Model.score(self.Train_Data) + 2 * self.num_params)
            BIC.append(-2 * Model.score(self.Train_Data) +  self.num_params * np.log(self.Train_Data.shape[0]))
    
            Temp1=np.argmin(AIC)
            opt_state=self.Max_state     
            Record_aic[0,:]=np.array([opt_state,min(AIC)])
            Temp2=np.argmin(BIC)
            opt_state=self.Max_state    
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
              Covariance Type = {self.Cov_Type}
              Number of Iterations = {self.Iter}
              Number of Hidden States = {self.Max_state}
              Maximum Number of Mixture Components = {self.Max_mixture}
              Length of Each Time Series is = {self.T}
              Number of Free Parameters = {self.num_params}
              Best Parameters are = {self.BEST}
        '''
    
   
