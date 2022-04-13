'''
This script constructs a 50-layer ResNet configured for image classification. By default, the network
is configured for binary classification on 64x64 RGB images, but can be configured for other image sizes,
greyscale images, and/or multiclass classification by modifying the arguments supplied to the call to resnet().
'''

from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import random_uniform, glorot_uniform 


# Create identity block consisting of three Conv2D blocks
def id_block(X, f, filters, training=True, initializer=random_uniform, seed=0):
        
    # Get filters
    F1, F2, F3 = filters
    
    # Get input for later use in skip connection
    X_input = X
    
    # Conv block 1
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=seed))(X)
    X = BatchNormalization(axis = 3)(X, training = training) 
    X = Activation('relu')(X)
    
    ## Conv block 2
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=seed))(X)
    X = BatchNormalization(axis = 3)(X, training = training)
    X = Activation('relu')(X)

    ## Conv block 3
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=seed))(X)
    X = BatchNormalization(axis = 3)(X, training = training)
    
    ## Add skip connection to main path
    X = Add()([X, X_input])
    X = Activation('relu')(X)

    return X

# Create convolutional block consisting of three Conv2D blocks in main path and one Conv2D block in skip connection
def conv_block(X, f, filters, s = 2, training=True, initializer=glorot_uniform, seed=0):
       
    # Get filters
    F1, F2, F3 = filters
    
    # Get input for later use in skip connection
    X_input = X

    # Conv block 1
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s,s), padding='valid', kernel_initializer = initializer(seed=seed))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)

    # Conv block 2
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding='same', kernel_initializer = initializer(seed=seed))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)

    # Conv block 3
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding='valid', kernel_initializer = initializer(seed=seed))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    
    # Skip connection
    X_input = Conv2D(filters = F3, kernel_size = 1, strides = (s,s), padding='valid', kernel_initializer = initializer(seed=seed))(X_input)
    X_input = BatchNormalization(axis = 3)(X_input, training=training)
    
    # Add skip connection to main path
    X = Add()([X, X_input])
    X = Activation('relu')(X)
    
    return X

# Create 50 layer ResNet from identity and convolutional blocks
# Default input is 64x64 RGB images
# Default output is one of two classes; i.e., binary classification
def resnet(input_shape=(64,64,3), classes=2, seed=0):
    
    # Define input layer
    X_input = Input(input_shape)

    # Zero-pad
    X = ZeroPadding2D((3,3))(X_input)
    
    # Layer set 1
    X = Conv2D(64, (7,7), strides = (2,2), kernel_initializer = glorot_uniform(seed=seed))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), strides=(2,2))(X)

    # Layer set 2
    X = conv_block(X, f = 3, filters = [64,64,256], s = 1)
    X = id_block(X, 3, [64,64,256])
    X = id_block(X, 3, [64,64,256])

    # Layer set 3
    X = conv_block(X, 3, [128,128,512])
    X = id_block(X, 3, [128,128,512])
    X = id_block(X, 3, [128,128,512])
    X = id_block(X, 3, [128,128,512])
    
    # Layer set 4
    X = conv_block(X, 3, [256,256,1024])
    X = id_block(X, 3, [256,256,1024])
    X = id_block(X, 3, [256,256,1024])
    X = id_block(X, 3, [256,256,1024])
    X = id_block(X, 3, [256,256,1024])
    X = id_block(X, 3, [256,256,1024])

    # Layer set 5
    X = conv_block(X, 3, [512,512,2048])
    X = id_block(X, 3, [512,512,2048])
    X = id_block(X, 3, [512,512,2048])

    # Average pooling
    X = AveragePooling2D((2,2))(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X)

    return model


# Instantiate 50-layer ResNet and output model summary
# A function call is provided to instantiate a ResNet for 64x64 RGB images and binary classification
# For color images of other sizes, provide (pixel_width, pixel_height, 3) as input_shape arg
# For greyscale images, provide (pixel_width, pixel_height, 1) as input_shape arg
# For multiclass classification, provide number of classes as classes arg
model = resnet(input_shape = (64,64,3), classes = 2)
print(model.summary())