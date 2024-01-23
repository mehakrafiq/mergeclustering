# %%
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import copy
from itertools import cycle, islice
from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.style.use('default')

# %% [markdown]
# # Begin here to import PCA and UMAP and Leiden Cluster Labels

# %%
pca_analysis = pd.read_csv('pca_results.csv', header = 0)
pca_analysis

# %%
fig = plt.figure()
ax = fig.add_subplot()
pcx = '0'
pcy = '1'
ax.scatter(pca_analysis[pcx], pca_analysis[pcy])
ax.set_xlabel(pcx)
ax.set_ylabel(pcy)
plt.show()

# %%
# Creating the plot
pcx = '3'
plt.scatter(range(2638), pca_analysis[pcx]) # 'o' is for circle markers
plt.title("One-Dimensional Vector Plot")
plt.xlabel("Index")
plt.ylabel("Value")

# Display the plot
plt.show()

# %%
n_pcs = 7
values_matrix = pca_analysis.values[:, 0:n_pcs]

# %%
values_matrix

# %%
import numpy as np

# Assuming values_matrix is your data matrix

mean = np.mean(values_matrix, axis=0)
std_dev = np.std(values_matrix, axis=0)

# Calculate the bounds for 3 standard deviations
lower_bound = mean - 3 * std_dev
upper_bound = mean + 3 * std_dev

# Create 23 bins within the bounds and add 2 bins for the outliers
bins = np.empty((10, values_matrix.shape[1]))
for col in range(values_matrix.shape[1]):
    bins[1:-1, col] = np.linspace(lower_bound[col], upper_bound[col], num=8)
    bins[0, col] = -np.inf  # Bin for values below lower bound
    bins[-1, col] = np.inf  # Bin for values above upper bound

# Digitize the values
digitized = np.empty_like(values_matrix)
for col in range(values_matrix.shape[1]):
    digitized[:, col] = np.digitize(values_matrix[:, col], bins=bins[:, col])

# digitized now contains the indices of the bins to which each value belongs


# %%
print(np.shape(bins))
print(np.shape(digitized))

# %%
#count the number of points in each bin combo. We have 25 bins for each x,y
dict25 = {}
for row in digitized:
    str_row = ' '.join(map(str, row.astype(int)))
    if  str_row not in dict25.keys():
        dict25[str_row] = 1
    else:
        dict25[str_row] = dict25[str_row] + 1

# %%
len(dict25)

# %%
#convert to probabilities
total_counts = sum(dict25.values())
dict25_sp = {}
for k, v in dict25.items():
    dict25_sp[k] = v / total_counts

# %%
#We have to sort it from highest probability to lowest in the txt output

# Sort the dictionary by value in descending order
sorted_dict = dict(sorted(dict25_sp.items(), key=lambda x: x[1], reverse=True))

# %%
ph=np.loadtxt('scanpy_pcs.txt', dtype=float, usecols=(0,1), skiprows=2, delimiter='|')
for i in range(len(ph)):
    if ph[i,1]==-1.0:
        ph[i,1]=0.0
plt.scatter(ph[:,1],ph[:,0], color='blue', lw=0, s=40)
plt.plot([-1,1],[-1,1],'--k')
plt.xlim([-0.0002,0.03])
plt.ylim([-0.0002,0.03])

print(len(ph))

# %%
microstates = pd.read_csv( "scanpy_pcs.txt.negmap" , sep="|" , skiprows= [1])
microstates.columns = [col.strip() for col in microstates.columns]
microstates["Vector"] = microstates["Vector"].apply(lambda v: np.array(v.strip().strip("[|]").split(), dtype= int))
microstates.head()

# %%
peaks = pd.read_csv("scanpy_pcs.txt" , sep="|" , skiprows= [1])
peaks.columns = [col.strip() for col in peaks.columns]
peaks.head()

# %%
clusters_ids = peaks["Birth State Index"].unique()

cluster_centers = np.array(microstates['Vector'][clusters_ids])

# %%
#Now we want to relabel the Pk value of the Pk centers because we had to kill them during persistent homology algorithm

for peak in clusters_ids:
    microstates['Pk'][peak] = peak
microstates[0:10]

# %%
# group dataframe by "Pk" and sum the "Prob" column for each group
cluster_probs = microstates.groupby('Pk')['Prob'].sum().to_dict()
cluster_probs = dict(sorted(cluster_probs.items(), key=lambda x: x[1])) #, reverse = True

#print(cluster_probs)

# %%
cluster_probs

# %%
# Function to find Pk value for a given vector element
def find_pk_for_vector_element(dataframe, vector_element):
    for index, row in dataframe.iterrows():
        if np.array_equal(row['Vector'], vector_element):
            return row['Pk']
    return None

pks_for_data = []
for i in range(len(digitized)):
    pks_for_data.append(find_pk_for_vector_element(microstates, digitized[i].astype(int)))

# %%
from collections import Counter

frequency = Counter(pks_for_data)

# Extract keys where values are less than 5
keys_less_than = [key for key, value in frequency.items() if value < 5] #ari = 0.89 for <5

# %%
print(len(frequency))
print(len(keys_less_than))

# %%
#This is just an example of how I compute the distance between all the clusters (islands)

import numpy as np
from scipy.spatial.distance import cdist
#this computes the single linkage distance between every island (cluster) based on the euclidean metric
#output is a symmetric matrix

size = len(clusters_ids)
NEW_single_linkage_distances = np.zeros((size, size))
labels = list(cluster_probs.keys())

for i, label1 in enumerate(labels):
    island1 = list(microstates.loc[microstates['Pk'] == label1, 'Vector'])
    for j in range(i+1, len(labels)):
        label2 = labels[j]
        island2 = list(microstates.loc[microstates['Pk'] == label2, 'Vector'])
        #distance_matrix = cdist(island1, island2, metric='euclidean')
        distance_matrix = cdist(island1, island2, metric='cityblock')
        min_distance = np.min(distance_matrix)
        NEW_single_linkage_distances[i, j] = min_distance
        NEW_single_linkage_distances[j, i] = min_distance

# %%
microstates_pks = microstates['Pk']
microstates_vectors = microstates['Vector']

# %%
# count how many islands (i.e. clusters) have a probability less than 5/2638
count = sum(1 for value in cluster_probs.values() if value < 5/2638)
count

# %%



