import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from plotly import __version__
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
import cufflinks as cf
import tensorflow as tf
init_notebook_mode(connected=True)
cf.go_offline()

# loading the data set
df=pd.read_csv("input.csv")
df.head()

#cheking null values in the dataset 
print(df.columns)
df.isnull().sum()

# Analysis to see distribution of classes
def plot_distribution(class_array, title):
    plt.figure(title)
    pd.DataFrame(class_array, columns = ['Class']).Class.value_counts().plot(kind='pie',autopct='%.3f %%',)
    plt.axis('equal')
    plt.title(title)
plot_distribution(df.Class,"Class Distribution")
pd.value_counts(df['Class'])

# Only amount is not normal, we normalize it to make data consistent
df["normAmount"]=(StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1)))
df.head()

print("Regular")
print(df.Amount[df['Class']==0].describe())
print()
print("Fraudulent")
print(df.Amount[df['Class']==1].describe())

# Fraudulent Data vs Time 
sns.kdeplot(df[df['Class']==1]['Time'],shade=True,color="green")
# Non Fraudulent Data vs Time
sns.kdeplot(df[df['Class']==0]['Time'],shade=True,color="red")

# Checking the important features and distribution of 2 classes among them
import matplotlib.gridspec as gridspec

vf = df.columns[1:29]
plt.figure(figsize=(18,30*4))
gs =gridspec.GridSpec(28, 1)
for i, features in enumerate(df[vf]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[features][df.Class ==1],color="b",bins=40)
    sns.distplot(df[features][df.Class ==0],color="r",bins=40)
    ax.set_xlabel('')
    ax.set_title('Distplot for : '+ str(features))
plt.show()

# Making additional features to help classification
df['V1_']=df.V1.map(lambda x:1 if x<-3 else 0)
df['V2_']=df.V2.map(lambda x:1 if x>2.5 else 0)
df['V3_']=df.V3.map(lambda x:1 if x<-4 else 0)
df['V4_']=df.V4.map(lambda x:1 if x>2.5 else 0)
df['V5_']=df.V5.map(lambda x:1 if x<-4.5 else 0)
df['V6_']=df.V6.map(lambda x:1 if x<-2.5 else 0)
df['V7_']=df.V7.map(lambda x:1 if x<-3 else 0)
df['V9_']=df.V9.map(lambda x:1 if x<-2 else 0)
df['V10_']=df.V10.map(lambda x:1 if x<-2.5 else 0)
df['V11_']=df.V11.map(lambda x:1 if x>2 else 0)
df['V12_']=df.V12.map(lambda x:1 if x<-2 else 0)
df['V14_']=df.V14.map(lambda x:1 if x<-2.5 else 0)
df['V16_']=df.V16.map(lambda x:1 if x<-2 else 0)
df['V17_']=df.V17.map(lambda x:1 if x<-2 else 0)
df['V18_']=df.V18.map(lambda x:1 if x<-2 else 0)
df['V19_']=df.V19.map(lambda x:1 if x>1.5 else 0)
df['V21_']=df.V21.map(lambda x:1 if x>0.6 else 0)

df=df.drop(['Amount'],axis=1)
df.head()

# making different Feature for each class 
df.loc[df.Class==0,'NonFraudulent']=1
df.loc[df.Class==1,'NonFraudulent']=0
df=df.rename(columns={'Class': 'Fraudulent'})
Fraudulent=df[df.Fraudulent==1]
NonFraudulent = df[df.NonFraudulent==1]

# Making the train and test dataset by keeping the ratio of each class constant in the training set
X_train=Fraudulent.sample(frac=0.8)
num_frauds=len(X_train)
X_train=pd.concat([X_train,NonFraudulent.sample(frac=0.8)])
X_test = df.loc[~df.index.isin(X_train.index)]

#shuffling to make the model adaptive
import sklearn
X_train=sklearn.utils.shuffle(X_train)
X_test=sklearn.utils.shuffle(X_test)

# Seperating the class features into y_train and y_test
y_train=X_train['Fraudulent']
y_train=pd.concat([y_train,X_train.NonFraudulent],axis=1)
y_test=X_test['Fraudulent']
y_test=pd.concat([y_test,X_test.NonFraudulent],axis=1)

X_train=X_train.drop(['Fraudulent','NonFraudulent','Time'],axis=1)
X_test=X_test.drop(['Fraudulent','NonFraudulent','Time'],axis=1)

# to give more weight to the positive class 
Weights_for_class=len(X_train)/num_frauds
y_train.Fraudulent=y_train*Weights_for_class
y_test.Fraudulent=y_test*Weights_for_class

X_train.head()
input_X=X_train.as_matrix()
input_y=y_train.as_matrix()
test_X=X_test.as_matrix()
test_y=y_test.as_matrix()

# Creating the graph



# Number of nodes in each layer
N_hidden_1=15
N_hidden_2=40
N_hidden_3=20

# Amount of units to keep during dropout
dropout_units = tf.placeholder(tf.float32)

X=tf.placeholder(tf.float32,shape=[None,35])

# Layer1
w1=tf.Variable(tf.truncated_normal([35,N_hidden_1], stddev=0.1))
b1=tf.Variable(tf.constant(0.1, shape=[N_hidden_1]))
layer1=tf.nn.sigmoid(tf.matmul(X,w1)+b1)

# Layer 2
w2=tf.Variable(tf.truncated_normal([N_hidden_1,N_hidden_2], stddev=0.1))
b2=tf.Variable(tf.constant(0.1, shape=[N_hidden_2]))
layer2=tf.nn.sigmoid(tf.matmul(layer1,w2)+b2)

# Layer 3
w3=tf.Variable(tf.truncated_normal([N_hidden_2,N_hidden_3], stddev=0.1))
b3=tf.Variable(tf.constant(0.1, shape=[N_hidden_3]))
layer3=tf.nn.sigmoid(tf.matmul(layer2,w3)+b3)

# Output layer
w4=tf.Variable(tf.truncated_normal([N_hidden_3,2], stddev=0.1))
b4=tf.Variable(tf.constant(0.1, shape=[2]))

# Predicted output
y_pred=tf.nn.softmax(tf.matmul(layer3,w4)+b4)

# True output
y_true=tf.placeholder(tf.float32,shape=[None,2])

# Setting up variables for the 
total_samples=y_train.shape[0]
batch_size=4000
n_epoch=5000
dropout=0.7
learning_rate=0.005

# Cost Function
cross_entropy =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))

# Creating an optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# True prediction to find the accuracy
correct_predicted=tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
accuracy=tf.reduce_mean(tf.cast(correct_predicted,tf.float32))

init=tf.global_variables_initializer()

# starting the session

stopping_early=0
validation_acc_arr=[]
batch_size=3000
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epoch):
        for i in range(int(total_samples/batch_size)):
            batch_x=input_X[i*batch_size:(i+1)*batch_size]
            batch_y=input_y[i*batch_size:(i+1)*batch_size]
            
            sess.run([optimizer],feed_dict={X:batch_x,y_true:batch_y,dropout_units:dropout})
            
        
        if(epoch%50==0):
            train_acc=sess.run([accuracy],feed_dict={X:input_X,y_true:input_y,dropout_units:dropout})
            test_acc=sess.run([accuracy],feed_dict={X:test_X,y_true:test_y,dropout_units:1.0})

            print("EPOCH :",epoch)
            print("batch", i)
            print("TRAIN_ACCURACY : ", train_acc)
            print("VALIDATION_ACCURACY :",test_acc)
           
                
            validation_acc_arr.append(test_acc)
            if validation_acc_arr<max(validation_acc_arr) and epoch>200:
                stopping_early=+1
                if stopping_early==15:
                    break
                else:
                    stopping_early=0
    print()
    print("done")
