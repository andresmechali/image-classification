
# Image Classification
In this project, you'll classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).  The dataset consists of airplanes, dogs, cats, and other objects. You'll preprocess the images, then train a convolutional neural network on all the samples. The images need to be normalized and the labels need to be one-hot encoded.  You'll get to apply what you learned and build a convolutional, max pooling, dropout, and fully connected layers.  At the end, you'll get to see your neural network's predictions on the sample images.
## Get the Data
Run the following cell to download the [CIFAR-10 dataset for python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import problem_unittests as tests
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('cifar-10-python.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()


tests.test_folder_path(cifar10_dataset_folder_path)
```

    All files found!
    

## Explore the Data
The dataset is broken into batches to prevent your machine from running out of memory.  The CIFAR-10 dataset consists of 5 batches, named `data_batch_1`, `data_batch_2`, etc.. Each batch contains the labels and images that are one of the following:
* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

Understanding a dataset is part of making predictions on the data.  Play around with the code cell below by changing the `batch_id` and `sample_id`. The `batch_id` is the id for a batch (1-5). The `sample_id` is the id for a image and label pair in the batch.

Ask yourself "What are all possible labels?", "What is the range of values for the image data?", "Are the labels in order or random?".  Answers to questions like these will help you preprocess the data and end up with better predictions.


```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import helper
import numpy as np

# Explore the dataset
batch_id = 3
sample_id = 3
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
```

    
    Stats of batch 3:
    Samples: 10000
    Label Counts: {0: 994, 1: 1042, 2: 965, 3: 997, 4: 990, 5: 1029, 6: 978, 7: 1015, 8: 961, 9: 1029}
    First 20 Labels: [8, 5, 0, 6, 9, 2, 8, 3, 6, 2, 7, 4, 6, 9, 0, 0, 7, 3, 7, 2]
    
    Example of Image 3:
    Image - Min Value: 40 Max Value: 255
    Image - Shape: (32, 32, 3)
    Label - Label Id: 6 Name: frog
    


![png](output_3_1.png)


## Implement Preprocess Functions
### Normalize
In the cell below, implement the `normalize` function to take in image data, `x`, and return it as a normalized Numpy array. The values should be in the range of 0 to 1, inclusive.  The return object should be the same shape as `x`.


```python
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    return (x / 255.0)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_normalize(normalize)
```

    Tests Passed
    

### One-hot encode
Just like the previous code cell, you'll be implementing a function for preprocessing.  This time, you'll implement the `one_hot_encode` function. The input, `x`, are a list of labels.  Implement the function to return the list of labels as One-Hot encoded Numpy array.  The possible values for labels are 0 to 9. The one-hot encoding function should return the same encoding for each value between each call to `one_hot_encode`.  Make sure to save the map of encodings outside the function.

**Hint:**

Look into LabelBinarizer in the preprocessing module of sklearn.


```python
from sklearn import preprocessing

ohe = preprocessing.LabelBinarizer().fit(range(10))

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    return ohe.transform(x)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_one_hot_encode(one_hot_encode)
```

    Tests Passed
    

### Randomize Data
As you saw from exploring the data above, the order of the samples are randomized.  It doesn't hurt to randomize it again, but you don't need to for this dataset.

## Preprocess all the data and save it
Running the code cell below will preprocess all the CIFAR-10 data and save it to file. The code below also uses 10% of the training data for validation.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
```

# Check Point
This is your first checkpoint.  If you ever decide to come back to this notebook or have to restart the notebook, you can start from here.  The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
```

## Build the network
For the neural network, you'll build each layer into a function.  Most of the code you've seen has been outside of functions. To test your code more thoroughly, we require that you put each layer in a function.  This allows us to give you better feedback and test for simple mistakes using our unittests before you submit your project.

>**Note:** If you're finding it hard to dedicate enough time for this course each week, we've provided a small shortcut to this part of the project. In the next couple of problems, you'll have the option to use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages to build each layer, except the layers you build in the "Convolutional and Max Pooling Layer" section.  TF Layers is similar to Keras's and TFLearn's abstraction to layers, so it's easy to pickup.

>However, if you would like to get the most out of this course, try to solve all the problems _without_ using anything from the TF Layers packages. You **can** still use classes from other packages that happen to have the same name as ones you find in TF Layers! For example, instead of using the TF Layers version of the `conv2d` class, [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d), you would want to use the TF Neural Network version of `conv2d`, [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d). 

Let's begin!

### Input
The neural network needs to read the image data, one-hot encoded labels, and dropout keep probability. Implement the following functions
* Implement `neural_net_image_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
 * Set the shape using `image_shape` with batch size set to `None`.
 * Name the TensorFlow placeholder "x" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
* Implement `neural_net_label_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
 * Set the shape using `n_classes` with batch size set to `None`.
 * Name the TensorFlow placeholder "y" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).
* Implement `neural_net_keep_prob_input`
 * Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) for dropout keep probability.
 * Name the TensorFlow placeholder "keep_prob" using the TensorFlow `name` parameter in the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder).

These names will be used at the end of the project to load your saved model.

Note: `None` for shapes in TensorFlow allow for a dynamic size.


```python
import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
    shape = [None]
    for i in range(len(image_shape)):
        shape.append(image_shape[i])
    return tf.placeholder(tf.float32, shape=shape, name="x")


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, shape=[None, n_classes], name="y")


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, name="keep_prob")


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)
```

    Image Input Tests Passed.
    Label Input Tests Passed.
    Keep Prob Tests Passed.
    

### Convolution and Max Pooling Layer
Convolution layers have a lot of success with images. For this code cell, you should implement the function `conv2d_maxpool` to apply convolution then max pooling:
* Create the weight and bias using `conv_ksize`, `conv_num_outputs` and the shape of `x_tensor`.
* Apply a convolution to `x_tensor` using weight and `conv_strides`.
 * We recommend you use same padding, but you're welcome to use any padding.
* Add bias
* Add a nonlinear activation to the convolution.
* Apply Max Pooling using `pool_ksize` and `pool_strides`.
 * We recommend you use same padding, but you're welcome to use any padding.

**Note:** You **can't** use [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) for **this** layer, but you can still use TensorFlow's [Neural Network](https://www.tensorflow.org/api_docs/python/tf/nn) package. You may still use the shortcut option for all the **other** layers.

** Hint: **

When unpacking values as an argument in Python, look into the [unpacking](https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists) operator. 


```python
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    
    # Create weights and bias
    color_channels = x_tensor.shape[3].value
    weights = tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1], color_channels, conv_num_outputs], stddev=0.01))

    bias = tf.Variable(tf.zeros(conv_num_outputs))
    
    # Apply a convolution to x_tensor using weight and conv_strides.
    convstrides = [1, conv_strides[0], conv_strides[1], 1]
    layer = tf.nn.conv2d(x_tensor, weights, strides=convstrides, padding='SAME')
    
    # Add bias
    layer = tf.nn.bias_add(layer, bias)
    
    # Add a nonlinear activation to the convolution.
    layer = tf.nn.relu(layer)
    
    # Apply Max Pooling using pool_ksize and pool_strides.
    ksize =[1, pool_ksize[0], pool_ksize[1], 1]
    poolstrides = [1, pool_strides[0], pool_strides[1], 1]
    layer = tf.nn.max_pool(layer, ksize=ksize, strides=poolstrides, padding='SAME')
    return layer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_con_pool(conv2d_maxpool)
```

    Tests Passed
    

### Flatten Layer
Implement the `flatten` function to change the dimension of `x_tensor` from a 4-D tensor to a 2-D tensor.  The output should be the shape (*Batch Size*, *Flattened Image Size*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.


```python
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    # I tried the following code and didn't work:
    '''shape = x_tensor.get_shape().as_list()
    shape = [shape[0], shape[1]*shape[2]*shape[3]]
    return tf.reshape(x_tensor, shape=shape)'''
    return tf.contrib.layers.flatten(x_tensor)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_flatten(flatten)
```

    Tests Passed
    

### Fully-Connected Layer
Implement the `fully_conn` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.


```python
def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    
    weights = tf.Variable(tf.truncated_normal([x_tensor.shape[1].value, num_outputs], stddev=0.01))
    bias = tf.Variable(tf.zeros([num_outputs]))
    
    layer = tf.matmul(x_tensor, weights)
    layer = tf.nn.bias_add(layer, bias)
    
    layer = tf.nn.relu(layer)
    
    return layer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_fully_conn(fully_conn)
```

    Tests Passed
    

### Output Layer
Implement the `output` function to apply a fully connected layer to `x_tensor` with the shape (*Batch Size*, *num_outputs*). Shortcut option: you can use classes from the [TensorFlow Layers](https://www.tensorflow.org/api_docs/python/tf/layers) or [TensorFlow Layers (contrib)](https://www.tensorflow.org/api_guides/python/contrib.layers) packages for this layer. For more of a challenge, only use other TensorFlow packages.

**Note:** Activation, softmax, or cross entropy should **not** be applied to this.


```python
def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    
    weights = tf.Variable(tf.truncated_normal([x_tensor.shape[1].value, num_outputs], stddev=0.01))
    bias = tf.Variable(tf.zeros([num_outputs]))
    
    layer = tf.matmul(x_tensor, weights)
    layer = tf.nn.bias_add(layer, bias)
    
    return layer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_output(output)
```

    Tests Passed
    

### Create Convolutional Model
Implement the function `conv_net` to create a convolutional neural network model. The function takes in a batch of images, `x`, and outputs logits.  Use the layers you created above to create this model:

* Apply 1, 2, or 3 Convolution and Max Pool layers
* Apply a Flatten Layer
* Apply 1, 2, or 3 Fully Connected Layers
* Apply an Output Layer
* Return the output
* Apply [TensorFlow's Dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) to one or more layers in the model using `keep_prob`. 


```python
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    
    conv_ksize = [4, 4]
    conv_strides = [1, 1]
    pool_ksize = [2, 2]
    pool_strides = [2, 2]
    
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv_layer = conv2d_maxpool(x, 16, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv_layer = conv2d_maxpool(conv_layer, 32, conv_ksize, conv_strides, pool_ksize, pool_strides)
    

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    flat_layer = flatten(conv_layer)
    

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    fully_connected = fully_conn(flat_layer, 256)
    fully_connected = fully_conn(flat_layer, 512)
    fully_connected = tf.nn.dropout(fully_connected, keep_prob)
    
    
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    output_layer = output(fully_connected, 10)
    
    
    # TODO: return output
    return output_layer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)
```

    Neural Network Built!
    

## Train the Neural Network
### Single Optimization
Implement the function `train_neural_network` to do a single optimization.  The optimization should use `optimizer` to optimize in `session` with a `feed_dict` of the following:
* `x` for image input
* `y` for labels
* `keep_prob` for keep probability for dropout

This function will be called for each batch, so `tf.global_variables_initializer()` has already been called.

Note: Nothing needs to be returned. This function is only optimizing the neural network.


```python
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
    session.run(optimizer, feed_dict={
        x:feature_batch, 
        y:label_batch, 
        keep_prob:keep_probability})


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_train_nn(train_neural_network)
```

    Tests Passed
    

### Show Stats
Implement the function `print_stats` to print loss and validation accuracy.  Use the global variables `valid_features` and `valid_labels` to calculate validation accuracy.  Use a keep probability of `1.0` to calculate the loss and validation accuracy.


```python
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    cost = session.run(cost, feed_dict={
        x:valid_features,
        y:valid_labels,
        keep_prob:1.0
    })
    accuracy = session.run(accuracy, feed_dict={
        x:valid_features,
        y:valid_labels,
        keep_prob:1.0
    })
    
    print("Loss: %.5f Accuracy: %.5f" %(cost, accuracy))
```

### Hyperparameters
Tune the following parameters:
* Set `epochs` to the number of iterations until the network stops learning or start overfitting
* Set `batch_size` to the highest number that your machine has memory for.  Most people set them to common sizes of memory:
 * 64
 * 128
 * 256
 * ...
* Set `keep_probability` to the probability of keeping a node using dropout


```python
# TODO: Tune Parameters
epochs = 20
batch_size = 256
keep_probability = 0.75
```

### Train on a Single CIFAR-10 Batch
Instead of training the neural network on all the CIFAR-10 batches of data, let's use a single batch. This should save time while you iterate on the model to get a better accuracy.  Once the final validation accuracy is 50% or greater, run the model on all the data in the next section.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)
```

    Checking the Training on a Single Batch...
    Epoch  1, CIFAR-10 Batch 1:  Loss: 2.06743 Accuracy: 0.24520
    Epoch  2, CIFAR-10 Batch 1:  Loss: 2.03996 Accuracy: 0.24440
    Epoch  3, CIFAR-10 Batch 1:  Loss: 1.97722 Accuracy: 0.27500
    Epoch  4, CIFAR-10 Batch 1:  Loss: 1.92909 Accuracy: 0.29700
    Epoch  5, CIFAR-10 Batch 1:  Loss: 1.88231 Accuracy: 0.31260
    Epoch  6, CIFAR-10 Batch 1:  Loss: 1.80482 Accuracy: 0.33880
    Epoch  7, CIFAR-10 Batch 1:  Loss: 1.70491 Accuracy: 0.38780
    Epoch  8, CIFAR-10 Batch 1:  Loss: 1.64138 Accuracy: 0.40940
    Epoch  9, CIFAR-10 Batch 1:  Loss: 1.59024 Accuracy: 0.42500
    Epoch 10, CIFAR-10 Batch 1:  Loss: 1.55658 Accuracy: 0.43660
    Epoch 11, CIFAR-10 Batch 1:  Loss: 1.54229 Accuracy: 0.43900
    Epoch 12, CIFAR-10 Batch 1:  Loss: 1.50837 Accuracy: 0.45260
    Epoch 13, CIFAR-10 Batch 1:  Loss: 1.49572 Accuracy: 0.45740
    Epoch 14, CIFAR-10 Batch 1:  Loss: 1.48064 Accuracy: 0.46080
    Epoch 15, CIFAR-10 Batch 1:  Loss: 1.46960 Accuracy: 0.47040
    Epoch 16, CIFAR-10 Batch 1:  Loss: 1.47013 Accuracy: 0.46760
    Epoch 17, CIFAR-10 Batch 1:  Loss: 1.44162 Accuracy: 0.47880
    Epoch 18, CIFAR-10 Batch 1:  Loss: 1.42353 Accuracy: 0.48540
    Epoch 19, CIFAR-10 Batch 1:  Loss: 1.43074 Accuracy: 0.48140
    Epoch 20, CIFAR-10 Batch 1:  Loss: 1.42657 Accuracy: 0.48120
    

### Fully Train the Model
Now that you got a good accuracy with a single CIFAR-10 batch, try it with all five batches.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
```

    Training...
    Epoch  1, CIFAR-10 Batch 1:  Loss: 2.08680 Accuracy: 0.22080
    Epoch  1, CIFAR-10 Batch 2:  Loss: 1.98915 Accuracy: 0.26640
    Epoch  1, CIFAR-10 Batch 3:  Loss: 1.96552 Accuracy: 0.26040
    Epoch  1, CIFAR-10 Batch 4:  Loss: 1.87127 Accuracy: 0.31500
    Epoch  1, CIFAR-10 Batch 5:  Loss: 1.76033 Accuracy: 0.35860
    Epoch  2, CIFAR-10 Batch 1:  Loss: 1.67881 Accuracy: 0.39520
    Epoch  2, CIFAR-10 Batch 2:  Loss: 1.64883 Accuracy: 0.39880
    Epoch  2, CIFAR-10 Batch 3:  Loss: 1.61915 Accuracy: 0.41360
    Epoch  2, CIFAR-10 Batch 4:  Loss: 1.59429 Accuracy: 0.41980
    Epoch  2, CIFAR-10 Batch 5:  Loss: 1.53812 Accuracy: 0.43760
    Epoch  3, CIFAR-10 Batch 1:  Loss: 1.53366 Accuracy: 0.44560
    Epoch  3, CIFAR-10 Batch 2:  Loss: 1.50966 Accuracy: 0.44340
    Epoch  3, CIFAR-10 Batch 3:  Loss: 1.52713 Accuracy: 0.44860
    Epoch  3, CIFAR-10 Batch 4:  Loss: 1.48529 Accuracy: 0.45920
    Epoch  3, CIFAR-10 Batch 5:  Loss: 1.47175 Accuracy: 0.46420
    Epoch  4, CIFAR-10 Batch 1:  Loss: 1.45778 Accuracy: 0.47480
    Epoch  4, CIFAR-10 Batch 2:  Loss: 1.45869 Accuracy: 0.46920
    Epoch  4, CIFAR-10 Batch 3:  Loss: 1.42270 Accuracy: 0.48840
    Epoch  4, CIFAR-10 Batch 4:  Loss: 1.42687 Accuracy: 0.48520
    Epoch  4, CIFAR-10 Batch 5:  Loss: 1.40209 Accuracy: 0.48600
    Epoch  5, CIFAR-10 Batch 1:  Loss: 1.41489 Accuracy: 0.48760
    Epoch  5, CIFAR-10 Batch 2:  Loss: 1.39830 Accuracy: 0.49320
    Epoch  5, CIFAR-10 Batch 3:  Loss: 1.38172 Accuracy: 0.49720
    Epoch  5, CIFAR-10 Batch 4:  Loss: 1.36800 Accuracy: 0.51180
    Epoch  5, CIFAR-10 Batch 5:  Loss: 1.33810 Accuracy: 0.51780
    Epoch  6, CIFAR-10 Batch 1:  Loss: 1.33916 Accuracy: 0.51720
    Epoch  6, CIFAR-10 Batch 2:  Loss: 1.32776 Accuracy: 0.51940
    Epoch  6, CIFAR-10 Batch 3:  Loss: 1.33236 Accuracy: 0.51900
    Epoch  6, CIFAR-10 Batch 4:  Loss: 1.31045 Accuracy: 0.53320
    Epoch  6, CIFAR-10 Batch 5:  Loss: 1.31295 Accuracy: 0.52340
    Epoch  7, CIFAR-10 Batch 1:  Loss: 1.30867 Accuracy: 0.52980
    Epoch  7, CIFAR-10 Batch 2:  Loss: 1.31193 Accuracy: 0.52460
    Epoch  7, CIFAR-10 Batch 3:  Loss: 1.28490 Accuracy: 0.53460
    Epoch  7, CIFAR-10 Batch 4:  Loss: 1.26218 Accuracy: 0.54440
    Epoch  7, CIFAR-10 Batch 5:  Loss: 1.27023 Accuracy: 0.53620
    Epoch  8, CIFAR-10 Batch 1:  Loss: 1.28388 Accuracy: 0.53760
    Epoch  8, CIFAR-10 Batch 2:  Loss: 1.26642 Accuracy: 0.54240
    Epoch  8, CIFAR-10 Batch 3:  Loss: 1.24342 Accuracy: 0.54900
    Epoch  8, CIFAR-10 Batch 4:  Loss: 1.22712 Accuracy: 0.56080
    Epoch  8, CIFAR-10 Batch 5:  Loss: 1.22504 Accuracy: 0.55840
    Epoch  9, CIFAR-10 Batch 1:  Loss: 1.25194 Accuracy: 0.54880
    Epoch  9, CIFAR-10 Batch 2:  Loss: 1.22201 Accuracy: 0.55600
    Epoch  9, CIFAR-10 Batch 3:  Loss: 1.21979 Accuracy: 0.55480
    Epoch  9, CIFAR-10 Batch 4:  Loss: 1.19918 Accuracy: 0.56100
    Epoch  9, CIFAR-10 Batch 5:  Loss: 1.19188 Accuracy: 0.56760
    Epoch 10, CIFAR-10 Batch 1:  Loss: 1.23740 Accuracy: 0.55640
    Epoch 10, CIFAR-10 Batch 2:  Loss: 1.18767 Accuracy: 0.56640
    Epoch 10, CIFAR-10 Batch 3:  Loss: 1.17573 Accuracy: 0.57700
    Epoch 10, CIFAR-10 Batch 4:  Loss: 1.16543 Accuracy: 0.57720
    Epoch 10, CIFAR-10 Batch 5:  Loss: 1.18214 Accuracy: 0.57180
    Epoch 11, CIFAR-10 Batch 1:  Loss: 1.20518 Accuracy: 0.56940
    Epoch 11, CIFAR-10 Batch 2:  Loss: 1.15843 Accuracy: 0.58080
    Epoch 11, CIFAR-10 Batch 3:  Loss: 1.14703 Accuracy: 0.57680
    Epoch 11, CIFAR-10 Batch 4:  Loss: 1.13753 Accuracy: 0.58900
    Epoch 11, CIFAR-10 Batch 5:  Loss: 1.14205 Accuracy: 0.58660
    Epoch 12, CIFAR-10 Batch 1:  Loss: 1.17562 Accuracy: 0.57620
    Epoch 12, CIFAR-10 Batch 2:  Loss: 1.12004 Accuracy: 0.60080
    Epoch 12, CIFAR-10 Batch 3:  Loss: 1.12090 Accuracy: 0.59480
    Epoch 12, CIFAR-10 Batch 4:  Loss: 1.11472 Accuracy: 0.59960
    Epoch 12, CIFAR-10 Batch 5:  Loss: 1.13371 Accuracy: 0.59200
    Epoch 13, CIFAR-10 Batch 1:  Loss: 1.12332 Accuracy: 0.60260
    Epoch 13, CIFAR-10 Batch 2:  Loss: 1.09667 Accuracy: 0.60140
    Epoch 13, CIFAR-10 Batch 3:  Loss: 1.10219 Accuracy: 0.60320
    Epoch 13, CIFAR-10 Batch 4:  Loss: 1.11197 Accuracy: 0.60140
    Epoch 13, CIFAR-10 Batch 5:  Loss: 1.13920 Accuracy: 0.58700
    Epoch 14, CIFAR-10 Batch 1:  Loss: 1.12362 Accuracy: 0.59520
    Epoch 14, CIFAR-10 Batch 2:  Loss: 1.07238 Accuracy: 0.61340
    Epoch 14, CIFAR-10 Batch 3:  Loss: 1.08154 Accuracy: 0.60820
    Epoch 14, CIFAR-10 Batch 4:  Loss: 1.07456 Accuracy: 0.61480
    Epoch 14, CIFAR-10 Batch 5:  Loss: 1.09012 Accuracy: 0.60660
    Epoch 15, CIFAR-10 Batch 1:  Loss: 1.11539 Accuracy: 0.59820
    Epoch 15, CIFAR-10 Batch 2:  Loss: 1.05900 Accuracy: 0.62020
    Epoch 15, CIFAR-10 Batch 3:  Loss: 1.07182 Accuracy: 0.60900
    Epoch 15, CIFAR-10 Batch 4:  Loss: 1.06607 Accuracy: 0.61800
    Epoch 15, CIFAR-10 Batch 5:  Loss: 1.12564 Accuracy: 0.59680
    Epoch 16, CIFAR-10 Batch 1:  Loss: 1.10381 Accuracy: 0.60580
    Epoch 16, CIFAR-10 Batch 2:  Loss: 1.06320 Accuracy: 0.61860
    Epoch 16, CIFAR-10 Batch 3:  Loss: 1.06788 Accuracy: 0.61660
    Epoch 16, CIFAR-10 Batch 4:  Loss: 1.05117 Accuracy: 0.62320
    Epoch 16, CIFAR-10 Batch 5:  Loss: 1.07922 Accuracy: 0.61620
    Epoch 17, CIFAR-10 Batch 1:  Loss: 1.08353 Accuracy: 0.61440
    Epoch 17, CIFAR-10 Batch 2:  Loss: 1.05374 Accuracy: 0.62020
    Epoch 17, CIFAR-10 Batch 3:  Loss: 1.08307 Accuracy: 0.61060
    Epoch 17, CIFAR-10 Batch 4:  Loss: 1.04624 Accuracy: 0.62700
    Epoch 17, CIFAR-10 Batch 5:  Loss: 1.08784 Accuracy: 0.61400
    Epoch 18, CIFAR-10 Batch 1:  Loss: 1.10561 Accuracy: 0.60780
    Epoch 18, CIFAR-10 Batch 2:  Loss: 1.05879 Accuracy: 0.62040
    Epoch 18, CIFAR-10 Batch 3:  Loss: 1.08782 Accuracy: 0.61880
    Epoch 18, CIFAR-10 Batch 4:  Loss: 1.03800 Accuracy: 0.63340
    Epoch 18, CIFAR-10 Batch 5:  Loss: 1.07889 Accuracy: 0.62200
    Epoch 19, CIFAR-10 Batch 1:  Loss: 1.11045 Accuracy: 0.60920
    Epoch 19, CIFAR-10 Batch 2:  Loss: 1.06295 Accuracy: 0.61760
    Epoch 19, CIFAR-10 Batch 3:  Loss: 1.08842 Accuracy: 0.61780
    Epoch 19, CIFAR-10 Batch 4:  Loss: 1.03302 Accuracy: 0.64000
    Epoch 19, CIFAR-10 Batch 5:  Loss: 1.04813 Accuracy: 0.62600
    Epoch 20, CIFAR-10 Batch 1:  Loss: 1.08162 Accuracy: 0.62440
    Epoch 20, CIFAR-10 Batch 2:  Loss: 1.06630 Accuracy: 0.61580
    Epoch 20, CIFAR-10 Batch 3:  Loss: 1.08203 Accuracy: 0.61860
    Epoch 20, CIFAR-10 Batch 4:  Loss: 1.08075 Accuracy: 0.62340
    Epoch 20, CIFAR-10 Batch 5:  Loss: 1.10241 Accuracy: 0.61260
    

# Checkpoint
The model has been saved to disk.
## Test Model
Test your model against the test dataset.  This will be your final accuracy. You should have an accuracy greater than 50%. If you don't, keep tweaking the model architecture and parameters.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()
```

    Testing Accuracy: 0.62001953125
    
    


![png](output_36_1.png)


## Why 50-80% Accuracy?
You might be wondering why you can't get an accuracy any higher. First things first, 50% isn't bad for a simple CNN.  Pure guessing would get you 10% accuracy. That's because there are many more techniques that can be applied to your model and we recemmond that once you are done with this project, you explore!

## Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook.  Save the notebook file as "dlnd_image_classification.ipynb" and save it as a HTML file under "File" -> "Download as".  Include the "helper.py" and "problem_unittests.py" files in your submission.


```python

```
