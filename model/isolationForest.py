'''
    Outlier detection using Isolation forest

'''

#imports 
import pandas as pd 
import seaborn as sns 
from sklearn.ensemble import IsolationForest


#load data
df = pd.read_csv('../Data/Xeek_Well_15-9-15.csv')

#data analysis 
df = df.dropna()


#Build the model 
 #choose our features 
anomaly_inputs = ['NPHI', 'RHOB']

'''
    Model params : 
        contamination -> how much of the overall data can be considered as an outlier 
        random_state -> control the random splitting of trees 
'''
IFModel = IsolationForest(contamination=0.1, random_state=42) 

#fit the model 
IFModel.fit(df[anomaly_inputs])


#Compute anomaly scores (using decision function) 
df['anomaly_scores'] = IFModel.decision_function(df[anomaly_inputs])

#predict anomalies using model
df['anomaly'] = IFModel.predict(df[anomaly_inputs])



#get only the columns we need 
    #we ll get 1 for normal and -1 for anomalies 
df.loc[:, ['NPHI', 'RHOB', 'anomaly_scores', 'anomaly']]


print(df.head)





