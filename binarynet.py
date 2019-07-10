import larq as lq
import tensorflow as tf


class BinaryNet(tf.keras.Model):

    def __init__(self, num_classes: int) -> None:
        """
        Builds the BinaryNet model in TensorFlow 2.0 and Larq.
        :param num_classes: The number of output classes
        """
        super(BinaryNet, self).__init__(name='BinaryNet')

        kwargs = {
            'input_quantizer': 'ste_sign',
            'kernel_quantizer': 'ste_sign',
            'kernel_constraint': 'weight_clip',
            'use_bias': False
        }
        # Conv Block 1
        self.conv_1_1 = lq.layers.QuantConv2D(128, 3,
                                              padding='same',
                                              kernel_quantizer='ste_sign',
                                              kernel_constraint='weight_clip',
                                              use_bias=False)
        self.bn_1_1 = tf.keras.layers.BatchNormalization(momentum=0.999, scale=False)
        self.conv_1_2 = lq.layers.QuantConv2D(128, 3, padding='same', **kwargs)
        self.max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.bn_1_2 = tf.keras.layers.BatchNormalization(momentum=0.999, scale=False)

        # Conv Block 2
        self.conv_2_1 = lq.layers.QuantConv2D(256, 3, padding='same', **kwargs)
        self.bn_2_1 = tf.keras.layers.BatchNormalization(momentum=0.999, scale=False)
        self.conv_2_2 = lq.layers.QuantConv2D(256, 3, padding='same', **kwargs)
        self.max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.bn_2_2 = tf.keras.layers.BatchNormalization(momentum=0.999, scale=False)

        # Conv Block 3
        self.conv_3_1 = lq.layers.QuantConv2D(512, 3, padding='same', **kwargs)
        self.bn_3_1 = tf.keras.layers.BatchNormalization(momentum=0.999, scale=False)
        self.conv_3_2 = lq.layers.QuantConv2D(512, 3, padding='same', **kwargs)
        self.max_pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.bn_3_2 = tf.keras.layers.BatchNormalization(momentum=0.999, scale=False)

        # Flatten
        self.flatten_4 = tf.keras.layers.Flatten()

        # FC Block
        self.fc_5 = lq.layers.QuantDense(1024, **kwargs)
        self.bn_5 = tf.keras.layers.BatchNormalization(momentum=0.999, scale=False)

        # FC Block
        self.fc_6 = lq.layers.QuantDense(1024, **kwargs)
        self.bn_6 = tf.keras.layers.BatchNormalization(momentum=0.999, scale=False)

        # FC Block
        self.fc_7 = lq.layers.QuantDense(num_classes, **kwargs)
        self.bn_7 = tf.keras.layers.BatchNormalization(momentum=0.999, scale=False)

        # Output
        self.network_output = tf.keras.layers.Softmax()

    def call(self, inputs: tf.Tensor) -> tf.keras.Model:
        """
        Builds the BinaryNet model when called.
        :param inputs: The input tensor
        :return: The built BinaryNet model
        """

        x = self.conv_1_1(inputs)
        x = self.bn_1_1(x)
        x = self.conv_1_2(x)
        x = self.max_pool_1(x)
        x = self.bn_1_2(x)

        x = self.conv_2_1(x)
        x = self.bn_2_1(x)
        x = self.conv_2_2(x)
        x = self.max_pool_2(x)
        x = self.bn_2_2(x)

        x = self.conv_3_1(x)
        x = self.bn_3_1(x)
        x = self.conv_3_2(x)
        x = self.max_pool_3(x)
        x = self.bn_3_2(x)

        x = self.flatten_4(x)

        x = self.fc_5(x)
        x = self.bn_5(x)

        x = self.fc_6(x)
        x = self.bn_6(x)

        x = self.fc_7(x)
        x = self.bn_7(x)

        x = self.network_output(x)

        return x

if __name__ == 'main':
    model = BinaryNet()
    