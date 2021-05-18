#!/usr/bin/env python
# coding: utf-8

# # Projet : Ensembles classifiers in Class Imbalance Learning
#     
# ## Nemanja Kostadinovic et Rebecca Leygonie

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab

from subprocess import call
from IPython.display import Image

import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, recall_score, classification_report, f1_score, make_scorer, precision_score


# # 1 - Import et analyse exploratoire des données 

# In[2]:


data = pd.read_csv("CarteBancaire.csv")
pd.set_option('display.max_columns', 35)


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


data.describe()


# In[6]:


print('Count of duplicates:', data.duplicated().sum())


# In[7]:


data = data.drop_duplicates()


# In[8]:


print('Presence of null values:', data.isnull().sum().any())


# In[9]:


target, freq = np.unique(data['Class'], return_counts=True)

plt.figure(figsize=(5,5))
plt.ylabel('Fréquence', fontsize=10)
plt.xlabel('Classe', fontsize=10)
sns.barplot(target, freq)
plt.title('Distribution des classes du dataset')
plt.savefig('distribution_origine.png', bbox_inches='tight')


# In[10]:


freq


# Les classes (0: pas de fraude , 1: fraude) sont mal distribués si nous utilisons ces données avec notre modèle Random Forest nous allons certainement nous retrouver face à un problème de sur-apprentissage car le classifieur va supposer que dans la majorité des cas il n'y a pas de fraudes donc ne va pas pouvoir reconnaitre une opération frauduleuse et va prédire seulement la classe 0.

# In[11]:


train = data.drop(['Class','Time','Amount'], axis='columns').copy()
train.sample(3)


# In[12]:


y = data[['Class']].copy()
y.sample(3)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)


# # 2 - Application des modèles sur les données d'origine

# ## 1. Modèles de Bagging 

# ### 1.1 BaggingClassifier
# 
# Un classificateur Bagging est un méta-estimateur d'ensemble qui ajuste les classificateurs de base chacun sur des sous-ensembles aléatoires de l'ensemble de données d'origine, puis agrège leurs prédictions individuelles (par vote ou par calcul de la moyenne) pour former une prédiction finale. 

# In[14]:


clf = BaggingClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[15]:


print(score)


# In[16]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc1.png', bbox_inches='tight')

# In[17]:


Bgcla_FN=conf_mat[1][0]/sum(conf_mat[1])*100
Bgcla_FP=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 1.2 Random Forest 
# 
# Une forêt aléatoire est un méta estimateur qui ajuste un certain nombre de classificateurs d'arbres de décision sur divers sous-échantillons de l'ensemble de données et utilise la moyenne pour améliorer la précision prédictive et contrôler le surajustement.

# In[18]:


clf=RandomForestClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[19]:


print(score)


# In[20]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc2.png', bbox_inches='tight')


# In[21]:


RF_FN=conf_mat[1][0]/sum(conf_mat[1])*100
RF_FP=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 1.3  ExtraTreesClassifier
# 
# L'extra tree classifier met en œuvre un méta estimateur qui ajuste un certain nombre d'arbres de décision randomisés (alias extra-arbres) sur divers sous-échantillons de l'ensemble de données et utilise la moyenne pour améliorer la précision prédictive et contrôler le surajustement.

# In[22]:


clf = ExtraTreesClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[23]:


print(score)


# In[24]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc3.png', bbox_inches='tight')


# In[25]:


ETC_FN=conf_mat[1][0]/sum(conf_mat[1])*100
ETC_FP=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 1.4 Logistic Regression

# In[26]:


logbagClf = BaggingClassifier(LogisticRegression(random_state=0, solver='lbfgs'), n_estimators = 400, oob_score = True, random_state = 90)
logbagClf.fit(X_train, y_train.values.ravel())


# In[27]:


predicted = logbagClf.predict(X_test)
score = accuracy_score(y_test, predicted)
print(score)


# In[28]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc4.png', bbox_inches='tight')


# In[29]:


RL_FN=conf_mat[1][0]/sum(conf_mat[1])*100
RL_FP=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 1.5 Gaussian Naïve Bayes

# In[30]:


GNBbagClf = BaggingClassifier(GaussianNB(), n_estimators = 400, oob_score = True, random_state = 90)
GNBbagClf.fit(X_train, y_train.values.ravel())


# In[31]:


predicted = logbagClf.predict(X_test)
score = accuracy_score(y_test, predicted)
print(score)


# In[32]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc5.png', bbox_inches='tight')


# In[33]:


GNB_FN=conf_mat[1][0]/sum(conf_mat[1])*100
GNB_FP=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 1.6 One class SVM

# In[34]:


data_sampled = data.sample(100000)
target, freq = np.unique(data_sampled['Class'], return_counts=True)
plt.figure(figsize=(5,5))
plt.ylabel('Fréquence', fontsize=10)
plt.xlabel('Classe', fontsize=10)
sns.barplot(target, freq)
plt.title('Distribution des classes du dataset')
plt.savefig('dist2.png', bbox_inches='tight')


# In[35]:


data_sampled.loc[data_sampled['Class'] == 1, "Class"] = -1
data_sampled.loc[data_sampled['Class'] == 0, "Class"] = 1


# In[36]:


train_sampled = data_sampled.drop(['Class','Time','Amount'], axis='columns').copy()
train_sampled.sample(3)


# In[37]:


y_sampled = data_sampled[['Class']].copy()
y_sampled.sample(3)


# In[38]:


X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(train_sampled, y_sampled, test_size=0.2, random_state=0)


# In[39]:


X_train_sampled.shape


# In[40]:


OSVM = BaggingClassifier(OneClassSVM(verbose=True), n_estimators = 5, oob_score = True, random_state = 90)
OSVM.fit(X_train_sampled, y_train_sampled.values.ravel())


# In[41]:


predicted = OSVM.predict(X_test_sampled)


# In[42]:


score = accuracy_score(y_test_sampled, predicted)
print(score)
conf_mat = confusion_matrix(y_test_sampled, predicted)
labels = ['-1','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc7.png', bbox_inches='tight')


# In[43]:


OSVM_FN=conf_mat[1][0]/sum(conf_mat[1])*100
OSVM_FP=conf_mat[0][1]/sum(conf_mat[0])*100


# ## 2. Modèles de Boosting

# ### 2.1 AdaBoostClassifier
# 
# 
# Un classificateur AdaBoost est un méta-estimateur qui commence par adapter un classificateur à l'ensemble de données d'origine, puis adapte des copies supplémentaires du classificateur au même ensemble de données, mais où les poids des cas mal classés sont ajustés de telle sorte que les classificateurs suivants se concentrent davantage sur les cas difficiles.

# In[44]:


clf = AdaBoostClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[45]:


print(score)


# In[46]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc8.png', bbox_inches='tight')


# In[47]:


ABC_FN=conf_mat[1][0]/sum(conf_mat[1])*100
ABC_FP=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 2.2 GradientBoostingClassifier
# 
# Le modèle Gradient Boosting construit un modèle additif de manière progressive, il permet d'optimiser des fonctions de perte arbitrairement différenciables. À chaque étape, des arbres de régression de n_classes_ sont ajustés sur le gradient négatif de la fonction de perte de déviance binomiale ou multinomiale.

# In[48]:


clf = GradientBoostingClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[49]:


print(score)


# In[50]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc9.png', bbox_inches='tight')


# In[51]:


GBC_FN=conf_mat[1][0]/sum(conf_mat[1])*100
GBC_FP=conf_mat[0][1]/sum(conf_mat[0])*100


# ## Comparaison

# In[52]:


modèles = ['Bagging','RandomForest  ','Extra Trees','AdaBoost  ','GradientBoosting','LogisticRegression', 'Gaussian NaïveBayes', 'OneClass SVM']
FN = [Bgcla_FN,RF_FN,ETC_FN,ABC_FN,GBC_FN,RL_FN,GNB_FN,OSVM_FN]
FP = [Bgcla_FP,RF_FP,ETC_FP,ABC_FP,GBC_FP,RL_FP,GNB_FP,OSVM_FP]
# Position sur l'axe des x pour chaque étiquette
position = np.arange(len(modèles))
# Largeur des barres
largeur = .35

# Création de la figure et d'un set de sous-graphiques
fig, ax = plt.subplots(figsize=(20, 5))
r1 = ax.bar(position - largeur/2, FN, largeur)
r2 = ax.bar(position + largeur/2, FP, largeur)

# Modification des marques sur l'axe des x et de leurs étiquettes
ax.set_xticks(position)
ax.set_xticklabels(modèles)


# In[53]:


fig = plt.figure()

x = [1,2,3,4,5,6,7,8]
height1 = [Bgcla_FN,RF_FN,ETC_FN,ABC_FN,GBC_FN,RL_FN,GNB_FN,OSVM_FN]
height2 = [Bgcla_FP,Bgcla_FP,ETC_FP,ABC_FP,GBC_FP,RL_FP,GNB_FP,OSVM_FP]
width = 0.5
BarName = ['Bagging Classifier','RandomForest','Extra Trees','AdaBoost','GradientBoosting','LogisticRegression', 'Gaussian NaïveBayes','OneClass SVM']

plt.bar(x, height1, width,color='b' )
plt.bar(x, height2, width,color='b' )
#plt.scatter([i+width/2.0 for i in x],height,s=40)

#plt.xlim(0,6)
#plt.ylim(0,100)
#plt.grid()

plt.ylabel('Pourcentage de faux négatifs %')
plt.title('Pourcentage de faux négatifs par modèle')

pylab.xticks(x, BarName, rotation=40)

plt.savefig('SimpleBar.png')
plt.show()


# # 2 - Application des modèles sur des données sous-échantilloné (under-sampling)

# In[54]:


rus = RandomUnderSampler()
X_under_sampled, y_under_sampled = rus.fit_resample(train, y)



sns.countplot('Class', data= y_under_sampled)
plt.title('Distribution des classes après sous-échantillonage', fontsize=14)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X_under_sampled, y_under_sampled, test_size=0.2, random_state=0)



# ## 1. Modèles de Bagging

# ### 1.1 Logistic Regression

# In[64]:


logbagClf = BaggingClassifier(LogisticRegression(random_state=0, solver='lbfgs'), n_estimators = 400, oob_score = True, random_state = 90)
logbagClf.fit(X_train, y_train.values.ravel())
predicted = logbagClf.predict(X_test)
score = accuracy_score(y_test, predicted)
print(score)


# In[65]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc10.png', bbox_inches='tight')


# In[66]:


RL_FN2=conf_mat[1][0]/sum(conf_mat[1])*100
RL_FP2=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 1.2 Gaussian Naives Bayes

# In[67]:


GNBbagClf = BaggingClassifier(GaussianNB(), n_estimators = 400, oob_score = True, random_state = 90)
GNBbagClf.fit(X_train, y_train.values.ravel())
predicted = GNBbagClf.predict(X_test)
score = accuracy_score(y_test, predicted)
print(score)


# In[68]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc11.png', bbox_inches='tight')


# In[69]:


GNB_FN2=conf_mat[1][0]/sum(conf_mat[1])*100
GNB_FP2=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 1.3  One class SVM

# In[70]:


y_under_train_OCSVM= y_train.copy()
y_under_test_OCSVM= y_test.copy()


# In[71]:


y_under_train_OCSVM.loc[y_under_train_OCSVM['Class'] == 1, "Class"] = -1
y_under_train_OCSVM.loc[y_under_train_OCSVM['Class'] == 0, "Class"] = 1

y_under_test_OCSVM.loc[y_under_test_OCSVM['Class'] == 1, "Class"] = -1
y_under_test_OCSVM.loc[y_under_test_OCSVM['Class'] == 0, "Class"] = 1


# In[72]:


OSVM = BaggingClassifier(OneClassSVM(verbose=True), n_estimators = 5, oob_score = True, random_state = 90)
OSVM.fit(X_train, y_under_train_OCSVM.values.ravel())


# In[73]:


predicted = OSVM.predict(X_test)


# In[74]:


score = accuracy_score(y_under_test_OCSVM, predicted)
print(score)
conf_mat = confusion_matrix(y_under_test_OCSVM, predicted)
labels = ['-1','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc12.png', bbox_inches='tight')


# In[75]:


OSVM_FN2=conf_mat[1][0]/sum(conf_mat[1])*100
OSVM_FP2=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 1.4 BaggingClassifier

# In[76]:


clf = BaggingClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[77]:


print(score)


# In[78]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc13.png', bbox_inches='tight')


# In[79]:


Bgcla_FN2=conf_mat[1][0]/sum(conf_mat[1])*100
Bgcla_FP2=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 1.5 Random Forest

# In[80]:


clf=RandomForestClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[81]:


print(score)


# In[82]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc14.png', bbox_inches='tight')


# In[83]:


RF_FN2=conf_mat[1][0]/sum(conf_mat[1])*100
RF_FP2=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 1.6 ExtraTreesClassifier

# In[84]:


clf = ExtraTreesClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[85]:


print(score)


# In[86]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc15.png', bbox_inches='tight')


# In[87]:


ET_FN2=conf_mat[1][0]/sum(conf_mat[1])*100
ET_FP2=conf_mat[0][1]/sum(conf_mat[0])*100


# ## 2 Modèles de Boosting

# ### 2.1 AdaBoostClassifier

# In[88]:


clf = AdaBoostClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[89]:


print(score)


# In[90]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc16.png', bbox_inches='tight')


# In[91]:


AB_FN2=conf_mat[1][0]/sum(conf_mat[1])*100
AB_FP2=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 2.2 GradientBoostingClassifier

# In[92]:


clf = GradientBoostingClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[93]:


print(score)


# In[94]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc17.png', bbox_inches='tight')


# In[95]:


GB_FN2=conf_mat[1][0]/sum(conf_mat[1])*100
GB_FP2=conf_mat[0][1]/sum(conf_mat[0])*100


# In[96]:


modèles = ['Bagging','RandomForest  ','Extra Trees','AdaBoost  ','GradientBoosting','RegressionLogistic','Gaussian NaiveBayes','OneClass SVM']
FN = [Bgcla_FN2,RF_FN2,ET_FN2,AB_FN2,GB_FN2,RL_FN2,GNB_FN2,OSVM_FN2]
FP = [Bgcla_FP2,RF_FP2,ET_FP2,AB_FP2,GB_FP2,RL_FP2,GNB_FP2,OSVM_FP2]
# Position sur l'axe des x pour chaque étiquette
position = np.arange(len(modèles))
# Largeur des barres
largeur = .35

# Création de la figure et d'un set de sous-graphiques
fig, ax = plt.subplots(figsize=(20, 5))
r1 = ax.bar(position - largeur/2, FN, largeur)
r2 = ax.bar(position + largeur/2, FP, largeur)

# Modification des marques sur l'axe des x et de leurs étiquettes
ax.set_xticks(position)
ax.set_xticklabels(modèles)


# # 3 - Application des modèles sur des données sur-échantillonées (over-sampling)

# In[97]:


rus = RandomOverSampler()
X_over_sampled, y_over_sampled = rus.fit_resample(train, y)


sns.countplot('Class', data= y_over_sampled)
plt.title('Distribution des classes après sur-échantillonage', fontsize=14)
plt.show()


# In[100]:


X_train, X_test, y_train, y_test = train_test_split(X_over_sampled, y_over_sampled, test_size=0.2, random_state=0)


# ## 1 Modèles de Bagging

# ### 1.1 Logistic Regression

# In[101]:


logbagClf = BaggingClassifier(LogisticRegression(random_state=0, solver='lbfgs'), n_estimators = 400, oob_score = True, random_state = 90)
logbagClf.fit(X_train, y_train.values.ravel())
predicted = logbagClf.predict(X_test)
score = accuracy_score(y_test, predicted)
print(score)


# In[102]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc18.png', bbox_inches='tight')


# In[103]:


RL_FN3=conf_mat[1][0]/sum(conf_mat[1])*100
RL_FP3=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 1.2 Gaussian Naives Bayes

# In[104]:


GNBbagClf = BaggingClassifier(GaussianNB(), n_estimators = 400, oob_score = True, random_state = 90)
GNBbagClf.fit(X_train, y_train.values.ravel())
predicted = GNBbagClf.predict(X_test)
score = accuracy_score(y_test, predicted)
print(score)


# In[105]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc19.png', bbox_inches='tight')


# In[106]:


GNB_FN3=conf_mat[1][0]/sum(conf_mat[1])*100
GNB_FP3=conf_mat[0][1]/sum(conf_mat[0])*100


# ### 1.3 One class SVM

# In[107]:


y_train_under_OCSVM = y_train.head(100000).copy()
y_test_under_OCSVM = y_test.copy()
X_train_under_OCSVM = X_train.head(100000).copy()


# In[108]:


target, freq = np.unique(y_test_under_OCSVM['Class'], return_counts=True)
plt.figure(figsize=(5,5))
plt.ylabel('Fréquence', fontsize=10)
plt.xlabel('Classe', fontsize=10)
sns.barplot(target, freq)
plt.title('Distribution des classes du dataset d\'apprentissage')
plt.savefig('dist_under_train.png', bbox_inches='tight')


# In[109]:


target, freq = np.unique(y_train_under_OCSVM['Class'], return_counts=True)
plt.figure(figsize=(5,5))
plt.ylabel('Fréquence', fontsize=10)
plt.xlabel('Classe', fontsize=10)
sns.barplot(target, freq)
plt.title('Distribution des classes du dataset de test')
plt.savefig('dist_under_test.png', bbox_inches='tight')


# In[110]:


y_train_under_OCSVM.loc[y_train_under_OCSVM['Class'] == 1, "Class"] = -1
y_train_under_OCSVM.loc[y_train_under_OCSVM['Class'] == 0, "Class"] = 1

y_test_under_OCSVM.loc[y_test_under_OCSVM['Class'] == 1, "Class"] = -1
y_test_under_OCSVM.loc[y_test_under_OCSVM['Class'] == 0, "Class"] = 1


# In[111]:


OSVM = BaggingClassifier(OneClassSVM(verbose=True), n_estimators = 5, oob_score = True, random_state = 90)
OSVM.fit(X_train_under_OCSVM, y_train_under_OCSVM.values.ravel())


# In[112]:


predicted = OSVM.predict(X_test)


# In[113]:


score = accuracy_score(y_test_under_OCSVM, predicted)
print(score)
conf_mat = confusion_matrix(y_test_under_OCSVM, predicted)
labels = ['-1','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.show()


# In[114]:


OSVM_FN3=conf_mat[1][0]/sum(conf_mat[1])*100
OSVM_FP3=conf_mat[0][1]/sum(conf_mat[0])*100


# ## 1.4 BaggingClassifier

# In[115]:


clf = BaggingClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[116]:


print(score)


# In[117]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc21.png', bbox_inches='tight')


# In[118]:


Bgcla_FN3=conf_mat[1][0]/sum(conf_mat[1])*100
Bgcla_FP3=conf_mat[0][1]/sum(conf_mat[0])*100


# ## 1.5 Random Forest

# In[119]:


clf=RandomForestClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[120]:


print(score)


# In[121]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc22.png', bbox_inches='tight')


# In[122]:


RF_FN3=conf_mat[1][0]/sum(conf_mat[1])*100
RF_FP3=conf_mat[0][1]/sum(conf_mat[0])*100


# ## 1.6 ExtraTrees Classifier

# In[123]:


clf = ExtraTreesClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[124]:


print(score)


# In[125]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc23.png', bbox_inches='tight')


# In[126]:


ET_FN3=conf_mat[1][0]/sum(conf_mat[1])*100
ET_FP3=conf_mat[0][1]/sum(conf_mat[0])*100


# # 2 Boosting

# ## 2.1 AdaBoostClassifier

# In[127]:


clf = AdaBoostClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[128]:


print(score)


# In[129]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc24.png', bbox_inches='tight')


# In[130]:


AB_FN3=conf_mat[1][0]/sum(conf_mat[1])*100
AB_FP3=conf_mat[0][1]/sum(conf_mat[0])*100


# ## 2.2 GradientBoostingClassifier

# In[131]:


clf = GradientBoostingClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)


# In[132]:


print(score)


# In[133]:


conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc25.png', bbox_inches='tight')


# In[134]:


GB_FN3=conf_mat[1][0]/sum(conf_mat[1])*100
GB_FP3=conf_mat[0][1]/sum(conf_mat[0])*100


# In[135]:


np.arange(10,2)


# ## Comparatif

# In[136]:


modèles = ['Bagging','RandomForest  ','Extra Trees','AdaBoost  ','GradientBoosting','RegressionLogistic', "Gaussian Naïve Bayes",'OneClass SVM']
FN = [Bgcla_FN3,RF_FN3,ET_FN3,AB_FN3,GB_FN3,RL_FN3,GNB_FN3,OSVM_FN3]
FP = [Bgcla_FP3,RF_FP3,ET_FP3,AB_FP3,GB_FP3,RL_FP3,GNB_FP3,OSVM_FP3]
# Position sur l'axe des x pour chaque étiquette
position = np.arange(len(modèles))
# Largeur des barres
largeur = .35

# Création de la figure et d'un set de sous-graphiques
fig, ax = plt.subplots(figsize=(20, 5))
r1 = ax.bar(position - largeur/2, FN, largeur)
r2 = ax.bar(position + largeur/2, FP, largeur)



# Modification des marques sur l'axe des x et de leurs étiquettes
ax.set_xticks(position)
ax.set_xticklabels(modèles)


# In[137]:


modèles = ['Bagging','RandomForest ','Extra Trees','AdaBoost  ','GradientBoosting','RegressionLogistic', "Gaussian Naïve Bayes",'OneClass SVM']
FN = [Bgcla_FN3,RF_FN3,ET_FN3,AB_FN3,GB_FN3,RL_FN3,GNB_FN3,OSVM_FN3]
FP = [Bgcla_FP3,RF_FP3,ET_FP3,AB_FP3,GB_FP3,RL_FP3,GNB_FP3,OSVM_FN3]
# Position sur l'axe des x pour chaque étiquette
position = np.arange(len(modèles))
# Largeur des barres
largeur = .35

# Création de la figure et d'un set de sous-graphiques
fig, ax = plt.subplots(figsize=(15,8))
r1 = ax.bar(position - largeur/2, FN, largeur)
r2 = ax.bar(position + largeur/2, FP, largeur)

# Modification des marques sur l'axe des x et de leurs étiquettes
ax.set_xticks(position)
ax.set_xticklabels(modèles)


# ## Test de modèles adaptés pour les données déséquilibrées

# In[138]:


X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=0)


# In[139]:


from imblearn.ensemble import BalancedRandomForestClassifier
clf = BalancedRandomForestClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)
print(score)
conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc25.png', bbox_inches='tight')


# In[140]:


from imblearn.ensemble import BalancedBaggingClassifier 
clf = BalancedBaggingClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)
print(score)
conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc26.png', bbox_inches='tight')

# In[141]:


from imblearn.ensemble import RUSBoostClassifier
clf = RUSBoostClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)
print(score)
conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc27.png', bbox_inches='tight')


# In[142]:


from imblearn.ensemble import EasyEnsembleClassifier
clf = EasyEnsembleClassifier()
clf.fit(X_train, y_train.values.ravel())
predicted = clf.predict(X_test)
score = accuracy_score(y_test, predicted)
print(score)
conf_mat = confusion_matrix(y_test, predicted)
labels = ['0','1']
fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(conf_mat, annot=True, cmap="Set3", fmt="d", xticklabels = labels, yticklabels = labels)
plt.ylabel('Vraie valeur')
plt.xlabel('Predite')
plt.title('Matrice de confusion')
plt.savefig('mc28.png', bbox_inches='tight')

