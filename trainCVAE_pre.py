from keras.layers import Input, Dense, Lambda, Dropout, Flatten, concatenate
from keras.models import Model, Sequential
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam, SGD
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.utils import np_utils
import cv2
import os
import keras.backend.tensorflow_backend as KTF
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
import pandas as pd

# ================== LAB RESOURCES ARE LIMITED=================== #

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
def get_session(gpu_fraction=0.5):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

#============================Define constants=======================================#
# Some Constants

# Batch size and maximum number of epochs
MAX_EPOCH = 25
BATCH_SIZE = 128

# Input Dimension i.e. image pretrained VGG-net features
n_x = 4096

# Conditional Variable size i.e. sketch extracted features
n_y = 4096

# Z size : random variable
n_z = 1024
internalSize = 2048
# path = '../../Datasets/SUN/'

sketch_features = Input(shape=[n_y], name='sketch_features')
image_features = Input(shape=[n_x] , name='image_features')
input_combined = concatenate([image_features, sketch_features])

#Construct Encoder
temp_h_q = Dense(internalSize*2, activation='relu')(input_combined)
temp_h_q_bn = BatchNormalization()(temp_h_q)
h_q_zd = Dropout(rate=0.3)(temp_h_q_bn)
h_q = Dense(internalSize, activation='relu')(h_q_zd)
h_q_bn = BatchNormalization()(h_q)

#parameters of hidden variable
mu = Dense(n_z, activation='tanh')(h_q_bn)
log_sigma = Dense(n_z, activation='tanh')(h_q_bn)

#Sampling layer - defined
def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=[n_z], mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps

#concatenate sampled z and conditional input i.e. sketch
z = Lambda(sample_z)([mu, log_sigma])
z_cond = concatenate([z, sketch_features])

#Define layers
decoder_hidden = Dense(internalSize, activation='relu')
decoder_out = Dense(n_x, activation='relu', name='decoder_out')

#construct Decoder
h_p = decoder_hidden(z_cond)
reconstr = decoder_out(h_p)

#Form models
encoder = Model(inputs=[sketch_features , image_features], outputs=[mu])

#Changed Decoder
# d_in = Input(shape=[n_z+n_y])

input_z = Input(shape=[n_z], name='input_z')
d_in = concatenate([input_z, sketch_features])
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(inputs=[sketch_features, input_z], outputs=[d_out])

# Predict the attribute again to enforce its usage
attr_int = Dense(internalSize, activation='relu')(reconstr)
attr_recons = Dense(n_y, activation='relu', name='recons_output')(attr_int)

# Form the VAE model
vae = Model(inputs=[sketch_features , image_features], outputs=[reconstr, attr_recons])


def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.mean(K.square(y_pred - y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    return recon + kl

encoder.summary()
decoder.summary()
adam = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
# sgd = SGD(lr=0.0005, momentum=0.9, nesterov=True)
vae.compile(optimizer=adam, loss={'decoder_out':vae_loss, 'recons_output':'mean_squared_error'}, loss_weights={'decoder_out':1.0, 'recons_output':10.0})

# ============================Load the required data================================== #

#first load the image features and imagePaths
image_VGG_features = np.load('image_vgg_features.npy')            #(12500, 4096)
image_paths = np.load('image_paths.npy')                    #(12500, )

#next load the sketch_paths
sketch_paths = np.load('sketch_paths.npy')                  #(,)
sketch_VGG_features = np.load('sketch_vgg_features.npy')    #(,)

#next load the image extension dataset
image_VGG_features_ext = np.load('image_ext_vgg_features.npy')    #(73002, 4096)
image_paths_ext = np.load('image_ext_paths.npy')            #(73002, )


train_sketch_paths = sketch_paths.tolist()
#Do a train and test split
sketch_paths_per_class = {}
for sketchPath in sketch_paths:
    className = sketchPath.split('/')[-2]
    if className not in sketch_paths_per_class:
        sketch_paths_per_class[className] = []
    sketch_paths_per_class[className].append(sketchPath)
    
#-------------------------------------------------Non-zero shot split-----------------------------------------------#
test_sketch_paths = np.array([])
# for className in sketch_paths_per_class:
#     sample_paths = np.random.choice(sketch_paths_per_class[className], 50, replace=False)
#     test_sketch_paths = np.append(test_sketch_paths, sample_paths)
#     for test_path in sample_paths:
#         train_sketch_paths.remove(test_path)
        
# train_sketch_paths = np.array(train_sketch_paths)

# print len(train_sketch_paths)
# print len(test_sketch_paths)

#---------------------------------------------------Zero shot split-------------------------------------------------#

test_ref_classes = np.load('test_split_ref.npy')

trainClasses = []
for className in sketch_paths_per_class:
    if className not in test_ref_classes:
        trainClasses.append(className)
        continue
    else:
        test_sketch_paths = np.append(test_sketch_paths, sketch_paths_per_class[className])
        for test_path in sketch_paths_per_class[className]:
            train_sketch_paths.remove(test_path)
        
print len(image_paths)
print len(train_sketch_paths)
print len(test_sketch_paths)


#-------------------------------------------------------------------------------------------------------------------#

#form an inverted index for sketch paths
sketch_path_index_tracker = {}
for idx in range(len(sketch_paths)):
    sketch_path_index_tracker[sketch_paths[idx]] = idx

#form an inverted index for image paths
image_path_index_tracker = {}
for idx in range(len(image_paths)):
    image_path_index_tracker[image_paths[idx]] = idx



# Now seperate features for train and test
train_sketch_X = np.zeros((len(train_sketch_paths), n_y))
test_sketch_X = np.zeros((len(test_sketch_paths), n_y))

for ii in range(len(train_sketch_paths)):
    index = sketch_path_index_tracker[train_sketch_paths[ii]]
    train_sketch_X[ii,:] = sketch_VGG_features[index, :]

for ii in range(len(test_sketch_paths)):
    index = sketch_path_index_tracker[test_sketch_paths[ii]]
    test_sketch_X[ii,:] = sketch_VGG_features[index, :]


def getImagePath(sketchPath):
    tempArr = sketchPath.replace('sketch', 'photo').split('-')
    imagePath = ''
    for idx in range(len(tempArr)-1):
        imagePath = imagePath + tempArr[idx] + '-'
    imagePath = imagePath[:-1]
    imagePath = imagePath + '.jpg'
    return imagePath

# Combine parallel images and sketches
train_X_img = np.zeros((len(train_sketch_paths), n_x))
for idx in range(len(train_sketch_paths)):
    imagePath = getImagePath(train_sketch_paths[idx]) 
    imageIdx = image_path_index_tracker[imagePath]
    train_X_img[idx,:] = image_VGG_features[imageIdx]
    

test_X_img = np.zeros((len(test_sketch_paths), n_x ))
for idx in range(len(test_sketch_paths)):
    imagePath = getImagePath(test_sketch_paths[idx]) 
    imageIdx = image_path_index_tracker[imagePath]
    test_X_img[idx,:] = image_VGG_features[imageIdx]



# =========================== TEST RETREIVAL ======================================#

#Build a nearest neighbour classifier
NEIGH_NUM = 200

from sklearn.neighbors import NearestNeighbors,LSHForest
# combined_img_features = np.concatenate((image_VGG_features, image_VGG_features_ext))

#-----------------------------------Generalized zero shot setting-------------------------------#

# nbrs = NearestNeighbors(n_neighbors=NEIGH_NUM, metric='cosine', algorithm='brute').fit(image_VGG_features_ext)

#--------------------------------Non-generalized zero shot setting-------------------------------#

#remove all the training class images
image_paths_ext_index_tracker = {}
for idx in range(len(image_paths_ext)):
    image_paths_ext_index_tracker[image_paths_ext[idx]] = idx

con_image_paths_ext = []
for path in image_paths_ext:
    className = path.split('/')[-2]
    if className not in trainClasses:
        con_image_paths_ext.append(path)

con_img_VGG_features_ext = np.zeros((len(con_image_paths_ext), 4096))
for idx in range(len(con_image_paths_ext)):
    originalIndex = image_paths_ext_index_tracker[con_image_paths_ext[idx]]
    con_img_VGG_features_ext[idx, :] = image_VGG_features_ext[originalIndex, :] 

nbrs = NearestNeighbors(n_neighbors=NEIGH_NUM, metric='cosine', algorithm='brute').fit(con_img_VGG_features_ext)

#------------------------------------------------------------------------------------------------#

#testing on test queries

# image_classes = np.array([path.split('/')[-2] for path in image_paths_ext])
image_classes = np.array([path.split('/')[-2] for path in con_image_paths_ext])

test_sketch_classes = np.array([path.split('/')[-2] for path in test_sketch_paths])


def mapChange(inputArr):
    dup = np.copy(inputArr)
    for idx in range(inputArr.shape[1]):
        if (idx != 0):
            dup[:,idx] = dup[:,idx-1] + dup[:,idx]
    return np.multiply(dup, inputArr)

#Use multiple z while prediction
#Write a function for prediction of precision
#Uses average of all predicted features for retrieval
def find_precision():
    RANDOM_Z_PER_SKETCH = 100
    noiseIP = np.random.normal(size=[RANDOM_Z_PER_SKETCH*len(test_sketch_paths) , n_z])
    sketchIP = np.zeros([RANDOM_Z_PER_SKETCH*len(test_sketch_paths) , n_y])
    for ii in range(0,len(test_sketch_paths)):
        for jj in range(0 , RANDOM_Z_PER_SKETCH):
            sketchIP[ii*RANDOM_Z_PER_SKETCH + jj] = test_sketch_X[ii]
    print 'Predicting...'
    predImageFeatures = decoder.predict({'sketch_features' : sketchIP , 'input_z' : noiseIP} , verbose=1)
    avgPredImgFeatures = np.zeros([len(test_sketch_paths) , n_x])
    for ii in range(0 , len(test_sketch_paths)):
        avgPredImgFeatures[ii] = np.mean(predImageFeatures[ii*RANDOM_Z_PER_SKETCH:(ii+1)*RANDOM_Z_PER_SKETCH] , axis=0)
    
    # From here...
    distances, indices = nbrs.kneighbors(avgPredImgFeatures)
    retrieved_classes = image_classes[indices]
    results = np.zeros(retrieved_classes.shape)
    for idx in range(results.shape[0]):
        results[idx] = (retrieved_classes[idx] == test_sketch_classes[idx])
    precision_200 = np.mean(results, axis=1)
    temp = [np.arange(200) for ii in range(results.shape[0])]
    mAP_term = 1.0/(np.stack(temp, axis=0) + 1)
    mAP = np.mean(np.multiply(mapChange(results), mAP_term), axis=1)
    print ''
    print 'The mean precision@200 for test sketches is ' + str(np.mean(precision_200))
    print 'The mAP for test_sketches is ' + str(np.mean(mAP))
    return np.mean(precision_200)

#===============================Training the model============================================#


prec = 0

vae.fit({'sketch_features': train_sketch_X , 'image_features': train_X_img }, [train_X_img, train_sketch_X] , batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCH)
find_precision()

# Use min as metric to retrieve closest
# find_precision_min()


def find_precision_kmeans():
    RANDOM_Z_PER_SKETCH = 100
    noiseIP = np.random.normal(size=[RANDOM_Z_PER_SKETCH*len(test_sketch_paths) , n_z])
    sketchIP = np.zeros([RANDOM_Z_PER_SKETCH*len(test_sketch_paths) , n_y])
    for ii in range(0,len(test_sketch_paths)):
        for jj in range(0 , RANDOM_Z_PER_SKETCH):
            sketchIP[ii*RANDOM_Z_PER_SKETCH + jj] = test_sketch_X[ii]
    print 'Predicting...'
    predImageFeatures = decoder.predict({'sketch_features' : sketchIP , 'input_z' : noiseIP} , verbose=1)
        

#use min(d,{Z}) as the distance measure
def find_precision_min():
    RANDOM_Z_PER_SKETCH = 10
    noiseIP = np.random.normal(size=[RANDOM_Z_PER_SKETCH*len(test_sketch_paths) , n_z])
    sketchIP = np.zeros([RANDOM_Z_PER_SKETCH*len(test_sketch_paths) , n_y])
    for ii in range(0,len(test_sketch_paths)):
        for jj in range(0 , RANDOM_Z_PER_SKETCH):
            sketchIP[ii*RANDOM_Z_PER_SKETCH + jj] = test_sketch_X[ii]
    print 'Predicting...'
    predImageFeatures = decoder.predict({'sketch_features' : sketchIP , 'input_z' : noiseIP} , verbose=1)
    #find the closest for each prediction
    distances = np.zeros((len(test_sketch_paths)*RANDOM_Z_PER_SKETCH, NEIGH_NUM))
    indices = np.zeros((len(test_sketch_paths)*RANDOM_Z_PER_SKETCH, NEIGH_NUM))
    PRED_BATCH_SIZE = 625
    for ii in range(PRED_BATCH_SIZE):
        distances[ii*PRED_BATCH_SIZE:(ii+1)*PRED_BATCH_SIZE], indices[ii*PRED_BATCH_SIZE:(ii+1)*PRED_BATCH_SIZE] = nbrs.kneighbors(predImageFeatures[ii*PRED_BATCH_SIZE:(ii+1)*PRED_BATCH_SIZE])
    comb_distances = np.zeros((len(test_sketch_paths), RANDOM_Z_PER_SKETCH*NEIGH_NUM))
    comb_indices = np.zeros((len(test_sketch_paths), RANDOM_Z_PER_SKETCH*NEIGH_NUM))
    for ii in range(len(test_sketch_paths)):
        for jj in range(RANDOM_Z_PER_SKETCH):
            comb_distances[ii,jj*NEIGH_NUM:(jj+1)*NEIGH_NUM] = distances[ii*RANDOM_Z_PER_SKETCH + jj, :]
            comb_indices[ii,jj*NEIGH_NUM:(jj+1)*NEIGH_NUM] = indices[ii*RANDOM_Z_PER_SKETCH + jj, :]            
    #next reduce these indices to top 200
    #first sort the array
    for ii in range(len(test_sketch_paths)):
        arrIdx = comb_distances[ii].argsort()
        comb_distances[ii] = comb_distances[ii][arrIdx]
        comb_indices[ii] = comb_indices[ii][arrIdx]
    #then get top 200 without dupliactes
    top_indices = np.zeros((len(test_sketch_paths), NEIGH_NUM)).astype(int)
    for ii in range(len(test_sketch_paths)):
        top_indices[ii,:] = pd.unique(comb_indices[ii,:])[:NEIGH_NUM]
    retrieved_classes = image_classes[top_indices]
    results = np.zeros(retrieved_classes.shape)
    for idx in range(results.shape[0]):
        results[idx] = (retrieved_classes[idx] == test_sketch_classes[idx])
    precision_200 = np.mean(results, axis=1)
    temp = [np.arange(200) for ii in range(results.shape[0])]
    mAP_term = 1.0/(np.stack(temp, axis=0) + 1)
    mAP = np.mean(np.multiply(mapChange(results), mAP_term), axis=1)
    print ''
    print 'The mean precision@200 using min metric for test sketches is ' + str(np.mean(precision_200))
    print 'The mAP for test_sketches using min metric is ' + str(np.mean(mAP))


