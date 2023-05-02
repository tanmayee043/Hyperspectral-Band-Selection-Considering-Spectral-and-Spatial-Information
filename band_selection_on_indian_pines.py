#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import scipy.io as sio
import spectral
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier

def sam_sc(data, k):
    """
    Select top k bands using Spectral Angle Mapper with Spatial Coherence.
    Input: 
        data: a 3D array (height x width x number of bands)
        k: the number of top bands to select
    Output:
        topk: a 1D array of length k containing the indices of the top k bands.
    """
    # Reshape data to a 2D array (number of pixels x number of bands)
    X = data.reshape((-1, data.shape[-1]))
    
    # Compute pairwise cosine similarity between spectral bands
    similarity = cosine_similarity(X.T)
    
    # Compute minimum spanning tree of the similarity graph
    mst = minimum_spanning_tree(csr_matrix(1 - similarity)).toarray()
    
    # Compute spatial coherence matrix
    height, width, nbands = data.shape
    coher = np.zeros((nbands, nbands))
    for i in range(height):
        for j in range(width):
            for b1 in range(nbands):
                for b2 in range(nbands):
                    if mst[b1, b2] == 1:
                        coher[b1, b2] += np.dot(data[i, j, b1], data[i, j, b2])
    
    # Compute SAM scores with spatial coherence
    sam = np.zeros(nbands)
    for i in range(nbands):
        for j in range(nbands):
            if i < j and mst[i, j] == 1:
                num = coher[i, j]
                den = np.sqrt(np.sum(X[:, i] ** 2) * np.sum(X[:, j] ** 2))
                sam[i] += num / den
    
    # Select top k bands with highest SAM scores
    topk = np.argsort(sam)[-k:]
    return topk


# In[2]:


import numpy as np
import scipy.io
import spectral
from skimage import filters, exposure
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import signal

# Load the Indian Pines dataset from the .mat file
indian_pines_mat= scipy.io.loadmat('Indian_pines_corrected.mat')

# Extract the data cube from the loaded dataset
data = indian_pines_mat['indian_pines_corrected']

# Print the shape of the data cube
print(f"Data cube shape of indian_pines_corrected: {data.shape}")

# Print the number of bands in the dataset
print(f"Number of bands of indian_pines_corrected: {data.shape[-1]}")


# In[3]:


# Get the shape of the original data
height, width, num_bands = data.shape

# Reshape the data to a 2D matrix
data_2d = data.reshape(height*width, num_bands)

# Define the pooling filter
pool_filter = np.ones((2, 2))

# Perform max pooling on each spectral band
data_pooled = np.zeros((height//2, width//2, num_bands))
for b in range(num_bands):
    band_data = data_2d[:, b].reshape(height, width)
    band_pooled = signal.convolve2d(band_data, pool_filter, mode='valid')[::2, ::2]
    data_pooled[:, :, b] = band_pooled

# Print the shape of the original and pooled data
print('Original shape:', data.shape)
print('Pooled shape:', data_pooled.shape)


# In[4]:


import scipy.io
import numpy as np

# Load the Indian Pines ground truth dataset from the .mat file
indian_pines_gt_mat = scipy.io.loadmat('Indian_pines_gt.mat')

# Extract the ground truth labels from the loaded dataset
data_gt = indian_pines_gt_mat['indian_pines_gt']

# Print the shape and unique values of the ground truth labels
print(f"Ground truth shape of indian_pines_gt: {data_gt.shape}")
print(f"Unique labels of indian_pines_gt: {set(data_gt.flatten())}")


# In[6]:


# Define a color map for the different classes
colors = np.array([
    [0, 0, 0],      # Class 0 (background)
    [255, 255, 0],  # Class 1 (Alfalfa)
    [0, 255, 0],    # Class 2 (Corn-notill)
    [0, 128, 0],    # Class 3 (Corn-mintill)
    [255, 0, 0],    # Class 4 (Corn)
    [128, 0, 0],    # Class 5 (Grass-pasture)
    [255, 255, 255],# Class 6 (Grass-trees)
    [0, 255, 255],  # Class 7 (Grass-pasture-mowed)
    [0, 255, 128],  # Class 8 (Hay-windrowed)
    [0, 128, 128],  # Class 9 (Oats)
    [255, 128, 0],  # Class 10 (Soybean-notill)
    [128, 255, 0],  # Class 11 (Soybean-mintill)
    [128, 128, 0],  # Class 12 (Soybean-clean)
    [255, 255, 128],# Class 13 (Wheat)
    [255, 255, 255],# Class 14 (Woods)
    [128, 128, 128],# Class 15 (Buildings-Grass-Trees-Drives)
    [0, 0, 255],    # Class 16 (New Class 1)
    [0, 255, 128],  # Class 17 (New Class 2)
    [255, 0, 128],  # Class 18 (New Class 3)
    [128, 0, 128]   # Class 19 (New Class 4)
])


# Create an RGB image from the ground truth labels using the color map
indian_pines_gt_rgb = colors[data_gt]

# Display the RGB image using matplotlib
plt.imshow(indian_pines_gt_rgb)
plt.axis('off')
plt.show()


# In[8]:


# Define the number of pixels to keep
num_pixels = 5184

# Get the indices of the pixels to keep
idx = np.random.choice(range(data_gt.shape[0] * data_gt.shape[1]), num_pixels, replace=False)

# Convert the indices to row and column coordinates
rows = idx // data_gt.shape[1]
cols = idx % data_gt.shape[1]

# Get the subset of the data
subset = data_gt[rows, cols]

# Save the subset to a new file
scipy.io.savemat('Indian_pines_gt_subset.mat', {'indian_pines_gt': subset})
# Load the of new dataset 
labels = scipy.io.loadmat('Indian_pines_gt_subset.mat')['indian_pines_gt']
print("new shape",labels.shape)


# In[34]:


X = data_pooled.reshape(-1, data_pooled.shape[-1])
y = labels.ravel()


# In[9]:


# Vary the value of k from 1 to 30 and compute the corresponding accuracy
accuracies1 = []
precisions1=[]
recalls1=[]
accuracies1_dt = []
precisions1_dt=[]
recalls1_dt=[]
accuracies1_knn = []
precisions1_knn=[]
recalls1_knn=[]
for k in range (1,31):
    topk = sam_sc(data_pooled,k)
    # Print indices of top k band
    print(topk) 
    X_sel = X[:, topk]
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)
    #classifier: random forest
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train[:, :k], y_train)
    y_pred = rfc. predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred, average='macro',zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro',zero_division=1)
    print("Classifier: Random Forest")
    print("overall accuracy: ",accuracy," macro-precision: ",precision," macro-recall: ",recall)
    accuracies1.append(accuracy)
    precisions1.append(precision)
    recalls1.append(recall)
    #classifier: decision tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy_dt = np.mean(y_pred == y_test)
    precision_dt = precision_score(y_test, y_pred, average='macro',zero_division=1)
    recall_dt = recall_score(y_test, y_pred, average='macro',zero_division=1)
    print("Classifier: Descision Tree")
    print("overall accuracy: ",accuracy_dt," macro-precision: ",precision_dt," macro-recall: ",recall_dt)
    accuracies1_dt.append(accuracy_dt)
    precisions1_dt.append(precision_dt)
    recalls1_dt.append(recall_dt)
    #classifier: knn
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_knn = np.mean(y_pred == y_test)
    precision_knn = precision_score(y_test, y_pred, average='macro',zero_division=1)
    recall_knn = recall_score(y_test, y_pred, average='macro',zero_division=1)
    print("Classifier: KNN")
    print("overall accuracy: ",accuracy_knn," macro-precision: ",precision_knn," macro-recall: ",recall_knn)
    accuracies1_knn.append(accuracy_knn)
    precisions1_knn.append(precision_knn)
    recalls1_knn.append(recall_knn)


# In[10]:


import scipy.io as sio
import numpy as np
from skimage.morphology import opening, closing, disk
from sklearn.cluster import KMeans
import warnings
def mpss(data_pooled,k):
    # Define the structuring element for the morphological operations
    selem=disk(3)
    # Compute the morphological profiles
    profiles = np.zeros_like(data_pooled)

    for i in range(data_pooled.shape[2]):
        band = data_pooled[:, :, i]
        opened = opening(band, selem)
        closed = closing(band, selem)
        profiles[:, :, i] = opened - closed
    # Cluster the pixels using k-means
    # Compute the spatial similarity matrix
    n_pixels = data_pooled.shape[0] * data_pooled.shape[1]
    locations = np.zeros((n_pixels, 2))

    for i in range(data_pooled.shape[0]):
        for j in range(data_pooled.shape[1]):
            locations[i * data_pooled.shape[1] + j] = [i, j]

    distances = np.zeros((n_pixels, n_pixels))

    for i in range(n_pixels):
        for j in range(i, n_pixels):
            distances[i, j] = np.linalg.norm(locations[i] - locations[j])
            distances[j, i] = distances[i, j]
        
    # Cluster the pixels using k-means
    kmeans = KMeans(n_clusters=k,n_init='auto',random_state=0).fit(np.concatenate((profiles.reshape(n_pixels, -1), distances), axis=1))
    labels = kmeans.labels_

    # Compute the variance of each band within each cluster
    variances = np.zeros((k, data_pooled.shape[2]))

    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        cluster_profiles = profiles.reshape(n_pixels, -1)[cluster_indices]
        for j in range(data_pooled.shape[2]):
            variances[i, j] = np.var(cluster_profiles[:, j])

    # Select the top k bands with the highest variance within each cluster
    top_bands = []
    for i in range(k):
        top_bands = np.argsort(variances[i])[-k:]

    return top_bands    
    


# In[11]:


# Vary the value of k from 1 to 30 and compute the corresponding accuracy
accuracies2 = []
precisions2 = []
recalls2 = []
accuracies2_dt = []
precisions2_dt=[]
recalls2_dt=[]
accuracies2_knn = []
precisions2_knn=[]
recalls2_knn=[]
for k in range (1,31):
    top_bands = mpss(data_pooled,k)
    # Print indices of top k band
    print(top_bands) 
    X_sel = X[:, top_bands]
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)
    #classifier: random forest
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train[:, :k], y_train)
    y_pred = rfc. predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred, average='macro',zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro',zero_division=1)
    print("Classifier: Random Forest")
    print("overall accuracy: ",accuracy," macro-precision: ",precision," macro-recall: ",recall)
    accuracies2.append(accuracy)
    precisions2.append(precision)
    recalls2.append(recall)
     #classifier: decision tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy_dt = np.mean(y_pred == y_test)
    precision_dt = precision_score(y_test, y_pred, average='macro',zero_division=1)
    recall_dt = recall_score(y_test, y_pred, average='macro',zero_division=1)
    print("Classifier: Descision Tree")
    print("overall accuracy: ",accuracy_dt," macro-precision: ",precision_dt," macro-recall: ",recall_dt)
    accuracies2_dt.append(accuracy_dt)
    precisions2_dt.append(precision_dt)
    recalls2_dt.append(recall_dt)
    #classifier: knn
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_knn = np.mean(y_pred == y_test)
    precision_knn = precision_score(y_test, y_pred, average='macro',zero_division=1)
    recall_knn = recall_score(y_test, y_pred, average='macro',zero_division=1)
    print("Classifier: KNN")
    print("overall accuracy: ",accuracy_knn," macro-precision: ",precision_knn," macro-recall: ",recall_knn)
    accuracies2_knn.append(accuracy_knn)
    precisions2_knn.append(precision_knn)
    recalls2_knn.append(recall_knn)


# In[84]:


import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors
def mfsm(X,y,k):
    # Reshape the data into a 2D array
    
    X_2d = np.reshape(X, (-1, X.shape[-1]))

    # Perform feature selection using mutual information
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X_2d, y.ravel())

    # Get the indices of the selected bands
    selected_bands=[]
    selected_bands = selector.get_support(indices=True)
    return selected_bands


# In[85]:


# Vary the value of k from 1 to 30 and compute the corresponding accuracy
accuracies3 = []
precisions3 = []
recalls3 = []
accuracies3_dt = []
precisions3_dt = []
recalls3_dt=[]
accuracies3_knn = []
precisions3_knn = []
recalls3_knn = []
for k in range (1,31):
    selected_bands = mfsm(data_pooled,labels,k)
    # Print indices of top k band
    print(selected_bands) 
    X_sel = X[:, selected_bands]
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)
    #classifier: random forest
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train[:, :k], y_train)
    y_pred = rfc. predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred, average='macro',zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro',zero_division=1)
    print("Classifier: random forest")
    print("overall accuracy: ",accuracy," macro-precision: ",precision," macro-recall: ",recall)
    accuracies3.append(accuracy)
    precisions3.append(precision)
    recalls3.append(recall)
    #classifier: descision tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy_dt = np.mean(y_pred == y_test)
    precision_dt = precision_score(y_test, y_pred, average='macro',zero_division=1)
    recall_dt = recall_score(y_test, y_pred, average='macro',zero_division=1)
    print("Classifier: Descision Tree")
    print("overall accuracy: ",accuracy_dt," macro-precision: ",precision_dt," macro-recall: ",recall_dt)
    accuracies3_dt.append(accuracy_dt)
    precisions3_dt.append(precision_dt)
    recalls3_dt.append(recall_dt)
    #classifier: knn
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy_knn = np.mean(y_pred == y_test)
    precision_knn = precision_score(y_test, y_pred, average='macro',zero_division=1)
    recall_knn = recall_score(y_test, y_pred, average='macro',zero_division=1)
    print("Classifier: KNN")
    print("overall accuracy: ",accuracy_knn," macro-precision: ",precision_knn," macro-recall: ",recall_knn)
    accuracies3_knn.append(accuracy_knn)
    precisions3_knn.append(precision_knn)
    recalls3_knn.append(recall_knn)


# In[86]:


fig, axs = plt.subplots(1, 3, figsize=(15,4))
axs[0].plot(range(1,31), accuracies1_knn, 'g', label='SAM_SC')
axs[0].plot(range(1,31), accuracies2_knn, 'b', label='MPSS')
axs[0].plot(range(1,31), accuracies3_knn, 'r', label='MFSM')
axs[0].set_xlabel('Number of Selected Bands (k)')
axs[0].set_ylabel('Overall Accuracy')
axs[0].set_title('Classifier:KNN')
axs[1].plot(range(1,31),precisions1_knn, 'g', label='SAM_SC')
axs[1].plot(range(1,31),precisions2_knn, 'b', label='MPSS')
axs[1].plot(range(1,31),precisions3_knn, 'r', label='MFSM')
axs[1].set_xlabel('Number of Selected Bands (k)')
axs[1].set_ylabel('Macro precision')
axs[1].set_title('Classifier:KNN')
axs[2].plot(range(1,31),recalls1_knn, 'g', label='SAM_SC')
axs[2].plot(range(1,31),recalls2_knn, 'b', label='MPSS')
axs[2].plot(range(1,31),recalls3_knn, 'r', label='MFSM')
axs[2].set_xlabel('Number of Selected Bands (k)')
axs[2].set_ylabel('Macro recall')
axs[2].set_title('Classifier:KNN')
plt.legend()
plt.show()


# In[87]:


fig, axs = plt.subplots(1, 3, figsize=(15,4))
axs[0].plot(range(1,31), accuracies1, 'g', label='SAM_SC')
axs[0].plot(range(1,31), accuracies2, 'b', label='MPSS')
axs[0].plot(range(1,31), accuracies3, 'r', label='MFSM')
axs[0].set_xlabel('Number of Selected Bands (k)')
axs[0].set_ylabel('Overall Accuracy')
axs[0].set_title('Classifier:RF')
axs[1].plot(range(1,31),precisions1, 'g', label='SAM_SC')
axs[1].plot(range(1,31),precisions2, 'b', label='MPSS')
axs[1].plot(range(1,31),precisions3, 'r', label='MFSM')
axs[1].set_xlabel('Number of Selected Bands (k)')
axs[1].set_ylabel('Macro precision')
axs[1].set_title('Classifier:RF')
axs[2].plot(range(1,31),recalls1, 'g', label='SAM_SC')
axs[2].plot(range(1,31),recalls2, 'b', label='MPSS')
axs[2].plot(range(1,31),recalls3, 'r', label='MFSM')
axs[2].set_xlabel('Number of Selected Bands (k)')
axs[2].set_ylabel('Macro recall')
axs[2].set_title('Classifier:RF')
plt.legend()
plt.show()


# In[88]:


fig, axs = plt.subplots(1, 3, figsize=(15,4))
axs[0].plot(range(1,31), accuracies1_dt, 'g', label='SAM_SC')
axs[0].plot(range(1,31), accuracies2_dt, 'b', label='MPSS')
axs[0].plot(range(1,31), accuracies3_dt, 'r', label='MFSM')
axs[0].set_xlabel('Number of Selected Bands (k)')
axs[0].set_ylabel('Overall Accuracy')
axs[0].set_title('Classifier:DT')  
axs[1].plot(range(1,31),precisions1_dt, 'g', label='SAM_SC')
axs[1].plot(range(1,31),precisions2_dt, 'b', label='MPSS')
axs[1].plot(range(1,31),precisions3_dt, 'r', label='MFSM')
axs[1].set_xlabel('Number of Selected Bands (k)')
axs[1].set_ylabel('Macro precision')
axs[1].set_title('Classifier:DT')
axs[2].plot(range(1,31),recalls1_dt, 'g', label='SAM_SC')
axs[2].plot(range(1,31),recalls2_dt, 'b', label='MPSS')
axs[2].plot(range(1,31),recalls3_dt, 'r', label='MFSM')
axs[2].set_xlabel('Number of Selected Bands (k)')
axs[2].set_ylabel('Macro recall')
axs[2].set_title('Classifier:DT')
#axs[1].imshow(labels)
#axs[1].set_title('Ground truth image')
plt.legend()
plt.show()


# In[ ]:




