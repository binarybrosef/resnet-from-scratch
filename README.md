# resnet-from-scratch
A 50-layer ResNet built from scratch in TensorFlow.

## ResNet Architecture
A ResNet - a portmanteau of 'residual' and 'network' - employs the so-called residual block in which skip connections propagate activations from earlier layers to later layers while skipping intervening layer(s). This enables residual blocks to learn identity functions, which, among other factors, facilitates training of deep ResNets that stack many layers of residual blocks while avoiding the vanishing gradient problem. ResNets have accordingly gained popularity for their ability to be deeply trained and provide high performance on complex datasets.

## ResNet50
Of a variety of ResNets that have been popularized, the ResNet50 is among the simpler of the ResNets, comprising fifty layers and over forty convolutions. A ResNet50 can be conceived of as being built from two fundamental components: an identity block, consisting of three convolutions and a skip connection, and a convolutional block, consisting of three convolutions and a skip connection having a fourth convolution. This ResNet 50 also employs ReLu activations, batch normalization (along the color channels axis), and both average and maximum pooling. 

## Construction in TensorFlow
TensorFlow's functional API is used to construct a ResNet50 from scratch. The following TensorFlow layers are used to construct identity and convolutional blocks:
- `Add`
- `Activation`
- `BatchNormalization`
- `Conv2D`

TensorFlow's `random_uniform` and `glorot_uniform` are used as weight initializers. 

## Script Function and Use
`model.py` constructs a 50-layer ResNet from scratch. Layers are built in a manual fashion to demonstrate how a ResNet50 can be constructed "by-hand" using TensorFlow's funtional API. More specifically, `model.py` constructs a graph model of a ResNet50 but does not compile or fit the model on any data. 

By default, a ResNet50 is constructed that is configured for binary classification on 64x64 RGB images. Alternative configurations can be specified by modifying arguments provided to the call to `resnet()` at the end of `model.py`.
- For alternative sizes of color images, specify `(image_width, image_height, 3)` as `input_shape`
- For greyscale images, specify `(image_width, image_height, 1)` as `input_shape`
- For multiclass classification, specify the desired number of classes as `classes`

The `evaluate()` method may be used to inference the model on user-provided images, following compilation and fitting of the model to training data.
