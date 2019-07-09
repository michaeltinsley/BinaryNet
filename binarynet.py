from typing import Tuple, Optional

import larq
import tensorflow as tf


class BinaryNetConvBlock(tf.keras.Model):

    def __init__(self, conv_filters: int, initial_block: bool = False, input_shape: Optional[Tuple[int, int, int]] = (96,96,3)) -> None:
        """
        Creates a BinaryNet convolution block, comprising of
        - Binary 2D convolution
        - Batch normalisation
        - Binary 2D convolution
        - Max Pooling
        - Batch normalisation
        :param conv_filters: The number of convolutional filters within the block
        :param initial_block: Boolean, True if the input block, False otherwise
        :param input_shape: If initial_block == True, provide an input shape
        """
        super(BinaryNetConvBlock, self).__init__(name='BinaryNetConvBlock')
        kwargs = dict(
            kernel_size=3,
            padding='same',
            input_quantizer='ste_sign',
            kernel_quantizer='ste_sign',
            kernel_constraint='weight_clip',
            use_bias=False
        )

        # First convolution
        if initial_block:  # If building the first block, do not quantize the input
            initial_block_kwargs = kwargs.copy()
            initial_block_kwargs['input_quantizer'] = None
            self.conv_1 = larq.layers.QuantConv2D(conv_filters, input_shape=input_shape, **initial_block_kwargs)
        else:  # Else, do apply quantization to the input
            self.conv_1 = larq.layers.QuantConv2D(conv_filters, **kwargs)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(momentum=0.999, scale=False)
        self.conv_2 = larq.layers.QuantConv2D(conv_filters, **kwargs)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(momentum=0.999, scale=False)

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """
        Builds the BinaryNet convolution block.
        :param input_tensor: The input to the model
        :return: The built block
        """
        x = self.conv_1(input_tensor)
        x = self.batch_norm_1(x)
        x = self.conv_2(x)
        x = self.max_pool(x)
        x = self.batch_norm_2(x)

        return x


class BinaryNetFCBlock(tf.keras.Model):

    def __init__(self, num_units: int) -> None:
        """
        Creates a BinaryNet fully connected block, comprising of
        - Binary densely connected layer
        - Batch normalisation
        :param num_units: The number of neuron units within the layer
        """
        super(BinaryNetFCBlock, self).__init__(name='BinaryNetFCBlock')
        kwargs = dict(
            input_quantizer='ste_sign',
            kernel_quantizer='ste_sign',
            kernel_constraint='weight_clip',
            use_bias=False
        )

        self.fc = larq.layers.QuantDense(num_units, **kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization(momentum=0.999, scale=False)

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """
        Builds the BinaryNet fully connected block.
        :param input_tensor: The input to the model
        :return: The built block
        """
        x = self.fc(input_tensor)
        x = self.batch_norm(x)
        return x


class BinaryNet(tf.keras.Model):

    def __init__(self, input_shape: Tuple[int, int, int] = (96, 96, 3), num_classes: int = 10) -> None:
        """
        Builds the BinaryNet model using tf.Keras and Larq for binarised layers.
        :param input_shape: The image input shape
        :param num_classes: The number of output classes
        """
        super(BinaryNet, self).__init__(name='BinaryNet')
        num_classes = 1 if num_classes == 2 else num_classes  # If binary classification, only 1 output

        # Layers
        self.conv_block_1 = BinaryNetConvBlock(128, initial_block=True, input_shape=input_shape)
        self.conv_block_2 = BinaryNetConvBlock(256, initial_block=False)
        self.conv_block_3 = BinaryNetConvBlock(512, initial_block=False)

        self.flatten = tf.keras.layers.Flatten()

        self.fully_connected_1 = BinaryNetFCBlock(1024)
        self.fully_connected_2 = BinaryNetFCBlock(1024)
        self.fully_connected_3 = BinaryNetFCBlock(num_classes)

        if num_classes == 1:
            self.network_output = tf.keras.layers.Activation('sigmoid')
        else:
            self.network_output = tf.keras.layers.Activation('softmax')

    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """
        Builds the network when called.
        :param input_tensor: The input to the model
        :return: The output tensor.
        """
        x = self.network_input(input_tensor)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        x = self.flatten(x)

        x = self.fully_connected_1(x)
        x = self.fully_connected_2(x)
        x = self.fully_connected_3(x)

        x = self.output(x)

        return x
