#Load VGG-net and save the image features in a dictionary
# source /media/data/ashish/tensorflow/tensorflow//bin/activate
import numpy as np
import keras
import pickle
import cv2

from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
import os
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.preprocessing import image


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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



#Load VGG-16 as image model due to lack of proper image data
vgg_model = vgg16.VGG16(weights='imagenet', include_top=True)
vgg_model.layers.pop()
vgg_model.layers[-1].outbound_nodes = []
vgg_model.outputs = [vgg_model.layers[-1].output]
vgg_model.summary()


#Store all sketch file names

#------------------------Change Sketchy Path here---------------------------------#
sketch_root = "/home/interns/sasi/Sketchy/sketch/tx_000100000000"
#---------------------------------------------------------------------------------#
#Store all the sketch paths
sketch_paths = []
for path, subdirs, files in os.walk(sketch_root):
    for fileName in files:
        sketch_paths.append(path + '/' + fileName)


np.save('sketch_paths', np.array(sketch_paths))


# for sketches
sketch_paths = np.load('sketch_paths.npy')

BATCH_SIZE = 41
X_out = np.zeros((len(sketch_paths), 4096))
X_in = np.zeros((BATCH_SIZE, 224, 224, 3))
for ii in range(len(sketch_paths)/BATCH_SIZE):
    print ('Batch ' + str(ii) + ' in progress...')
    for jj in range(BATCH_SIZE):
        X_in[jj,:,:,:] = image.img_to_array( image.load_img(sketch_paths[ii*BATCH_SIZE + jj], target_size=(224, 224)) )
    X_in = preprocess_input(X_in)
    X_out[ii*BATCH_SIZE:(ii+1)*BATCH_SIZE, :] = vgg_model.predict_on_batch(X_in)


#store the image paths and vgg_features
np.save('vgg_sketch_features_mod', X_out)
np.save('sketch_paths', np.array(sketch_paths))

