import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pylab as pl
from sklearn.metrics import silhouette_score

df=pd.DataFrame(data=np.random.uniform(low=5.5,high=7,size=400) ,columns=['PH'])
df['EC']=np.random.uniform(low=0.8,high=1.2,size=400)
df['Temp']=np.random.uniform(low=17,high=22,size=400)
df['Humidity']=np.random.uniform(low=45,high=70,size=400)
df['Co2']=np.random.uniform(low=410,high=800,size=400)
df['o2']=np.random.uniform(low=20,high=30,size=400)

print(df)

#df.plot()
#plt.show()

km=KMeans(n_clusters=6)
y_km=km.fit_predict(df)
print(silhouette_score(df, y_km))

df['target']=y_km
pca = PCA(n_components=2).fit(df)
pca_2d = pca.transform(df)

for i in range(0, pca_2d.shape[0]):
	if km.labels_[i] == 0:
		c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='red',
    marker='+')
	elif km.labels_[i] == 1:
		c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='green',
    marker='o')
	elif km.labels_[i] == 2:
		c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='blue',
    marker='*')
	elif km.labels_[i] ==3:
		c4 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='pink',
    marker='^')
	elif km.labels_[i]==4:
		c5 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='black',
    marker='*')
	elif km.labels_[i]==5:
		c6 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='yellow',
    marker='*')

pl.legend([c1, c2, c3,c4,c5,c6], ['cluster_1', 'cluster_2',
    'cluster_3','cluster_4','cluster_5','cluster_6'])
plt.show()


df=df.drop(['target'],axis=1)

df.plot()
plt.show()


