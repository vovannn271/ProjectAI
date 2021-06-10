import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ds=pd.read_csv("heart.csv")
ds.head()
ds.info()




# age to count plot
plt.figure(figsize=(20,15))
sns.countplot(x=ds["age"])
plt.xlabel("Age",fontsize=25)
plt.ylabel("Count",fontsize=25)
plt.style.use("ggplot")
plt.show()


#Countplot of Sex
plt.figure(figsize=(20,10))
sns.countplot(x=ds["sex"])
plt.title("Countplot of Sex")
plt.style.use("ggplot")
plt.show()


# correlation among the variables
plt.figure(figsize=(20,10))
sns.heatmap(ds.corr())
plt.style.use("ggplot")
plt.show()





#plot of Blood presure of patients
plt.figure(figsize=(20,10))
sns.displot(ds["trtbps"])
plt.title("Patients' Blood presure")
plt.xlabel("Blood presure",fontsize=19)
plt.ylabel("Count",fontsize=19)
plt.style.use("ggplot")
plt.show()


#plot of CHOLESTROL LEVEL of patients
plt.figure(figsize=(20,20))
sns.set_color_codes()
sns.displot(ds["chol"],color="y")
plt.title("Patients' Cholestrol level")
plt.xlabel("Cholestrol level",fontsize=19)
plt.ylabel("Count",fontsize=19)
plt.style.use("ggplot")
plt.show()


#plot of Heart rate of patients
plt.figure(figsize=(20,10))
plt.style.use("ggplot")
sns.displot(ds["thalachh"],color="blue")
plt.title("Patients' HEART RATE",fontsize=19)
plt.xlabel("Heart rate",fontsize=19)
plt.ylabel("Count",fontsize=19)
plt.show()




#Plots of heart attack related to some characteristics
fig, ax = plt.subplots(2,2, figsize=(25,15))
plt.style.use("ggplot")
sns.kdeplot(x="age", data=ds, hue="output", ax=ax[0][0], fill="True").set(title="Heart attack to Age")
sns.kdeplot(x="thalachh", data=ds, hue="output", ax=ax[0][1], fill="True").set(title="Heart attack to Heart rate")
sns.kdeplot(x="chol", data=ds, hue="output", ax=ax[1][0], fill="True").set(title="Heart attack to Cholestrol")
sns.kdeplot(x="trtbps", data=ds, hue="output", ax=ax[1][1], fill="True").set(title="Heart attack to Blood pressure")




#for predictable random
seed(0)
tf.random.set_seed(0)


x_train, x_test, y_train, y_test = train_test_split(ds.drop('output', axis=1), ds['output'], test_size = 0.2, random_state = 0)


x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

ann = tf.keras.models.Sequential()

#adding 2 hidden layers that have 25 neurons
ann.add(tf.keras.layers.Dense(25,activation = 'relu'))
ann.add(tf.keras.layers.Dense(25,activation = 'relu'))


#adding hidden layer that has 10 neurons
ann.add(tf.keras.layers.Dense(10,activation = 'relu'))


#adding output layer that has 1 neuron (y/n)
ann.add(tf.keras.layers.Dense(1,activation = 'sigmoid'))



#compiling our model using adam optimization algorithm
ann.compile(loss = 'binary_crossentropy',optimizer = "Adam", metrics = ['accuracy'])



#training our model
ann.fit(x_train, y_train,batch_size = 30, epochs = 100)




#Comparing loss and accuracy values plot
losses = pd.DataFrame(ann.history.history)
losses.plot()

plt.title("Comparing loss and accuracy values",fontsize=20)
plt.xlabel("Epoch count",fontsize=20)
plt.ylabel("Value",fontsize=20)
plt.style.use("ggplot")
plt.show()





print('Accuracy - {0:.2f}%'.format(metrics.accuracy_score(y_true= y_test, y_pred= ann.predict_classes(x_test)) * 100))
