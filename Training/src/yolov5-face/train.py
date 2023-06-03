import tensorflow.compat.v1 as tf
import facenet
import math
import numpy as np
from sklearn.svm import SVC
import pickle

tf.disable_v2_behavior()


def face2database(picture_path, model_path, database_path, batch_size=90, image_size=160):

    with tf.Graph().as_default():
        with tf.Session() as sess:
            dataset = facenet.get_dataset(picture_path)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model_path)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            np.savez(database_path, emb=emb_array, lab=labels)

def ClassifyTrainSVC(database_path, SVCpath):
    Database = np.load(database_path)
    name_lables = Database['lab']
    embeddings = Database['emb']
    name_unique = np.unique(name_lables)
    labels = []
    for i in range(len(name_lables)):
        for j in range(len(name_unique)):
            if name_lables[i] == name_unique[j]:
                labels.append(j)
    print('Training classifier')
    model = SVC(kernel='linear', probability=True)
    model.fit(embeddings, labels)
    with open(SVCpath, 'wb') as outfile:
        pickle.dump((model, name_unique), outfile)
        print('Saved classifier model to file "%s"' % SVCpath)
