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


#use visualizations for better understanding 

#create a generic function
def anom_plot(data, anomaly_detection_method, x_var, y_var, 
              xaxis_limits=[0,1], yaxis_limits=[0,1]):
    print(f'Anomaly detection method: {anomaly_detection_method}')

    method = f'{anomaly_detection_method}_anomaly'

    print(f"Number of anomalous values {len(data[data['anomaly'] == -1])}")
    print(f"Number of non anomalous values {len(data[data['anomaly'] == 1])}")
    print(f"Total number of values {len(data)}")

    #create the plot 
    #initialize grid of subplots based on the anomaly column 
        #col = 'anomaly' specifies that the grid will have columns based on the unique values in the anomaly column
        #hue = 'anomaly' specifies that the different hues (colors) will be used to differentiate between diff values in the anomaly column  
    g = sns.FacetGrid(data, col='anomaly', height=4, hue='anomaly', hue_order=[1,-1])
    
    '''
        Mapping Scatterplot onto Grid : 
            creates scatterplot following Grid 
    '''
    g.map(sns.scatterplot, x_var, y_var)
    
    g.fig.suptitle(f'Anomaly detection method : {anomaly_detection_method}', y=1.10, fontweight='bold')
    
    #setting axes limit 
    g.set(xlim=xaxis_limits, ylim= yaxis_limits)
    
    #Setting titles 
    axes = g.axes.flatten()
    axes[0].set_title(f"Outliers\n{len(data[data['anomaly'] == -1])} points")
    axes[1].set_title(f"Inliers\n{len(data[data['anomaly'] == 1])} points")

    return g 



#test : 
anom_plot(df, "Isolation forest", "NPHI", "RHOB", [0, 0.8], [3, 1.5])




    







