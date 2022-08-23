"""Define Mobile RRN architecture.

Mobile RRN is a lite version of Revisiting Temporal Modeling (RRN) which is a recurrent network for
video super-resolution to run on mobile.

Each Mobile RRN cell firstly concatenate input sequence LR frames and hidden state.
Then, forwarding it through several residual blocks to output prediction and update hidden state.

Reference paper https://arxiv.org/abs/2008.05765
Reference github https://github.com/junpan19/RRN/
"""

import tensorflow as tf
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras import Model


class MobileRRN(tf.keras.Model):
    """Implement Mobile RRN architecture.

    Attributes:
        scale: An `int` indicates the upsampling rate.
        base_channels: An `int` represents the number of base channels.
    """

    def __init__(self,):
        """Initialize `RRN`."""
        super().__init__()
        in_channels = 3
        out_channels = 3
        block_num = 3  # the number of residual block in RNN cell

        self.base_channels = 16
        self.scale = 4

        # first conv
        self.conv_first_forward = tf.keras.layers.Conv2D(self.base_channels, 3, 1, padding='SAME', activation='relu')
        self.conv_first_backward = tf.keras.layers.Conv2D(self.base_channels, 3, 1, padding='SAME', activation='relu')

        self.recon_trunk_forward = make_layer(ConvBlock, block_num, base_channels=self.base_channels)
        self.recon_trunk_backward = make_layer(ConvBlock, block_num, base_channels=self.base_channels)
        self.recon_trunk = make_layer(ConvBlock, block_num, base_channels=self.base_channels)

        self.conv_fusion = tf.keras.layers.Conv2D(self.base_channels, 3, 1, padding='SAME', activation='relu')
        self.conv_last = tf.keras.layers.Conv2D(self.scale * self.scale * out_channels, 3, 1, padding='SAME')

        self.conv_hidden_forward = tf.keras.layers.Conv2D(self.base_channels, 3, 1, padding='SAME')
        self.conv_hidden_backward = tf.keras.layers.Conv2D(self.base_channels, 3, 1, padding='SAME')

    def call(self, inputs, training=False):
        """Forward the given input.

        Args:
            inputs: An input `Tensor` and an `Tensor` represents the hidden state.
            training: A `bool` indicates whether the current process is training or testing.

        Returns:
            An output `Tensor`.
        """
        x, hidden = inputs
        x1 = x[:, :, :, :3]
        x2 = x[:, :, :, 3:6]
        x3 = x[:, :, :, 6:]
        # _, h, w, _ = x1.shape.as_list()
        shape = tf.shape(x1)

        hidden_forward = hidden[:, :, :, :self.base_channels]
        hidden_backward = hidden[:, :, :, self.base_channels:]

        x_forward = tf.concat((x1, x2, hidden_forward), axis=-1)
        x_backward = tf.concat((x3, x2, hidden_backward), axis=-1)

        x_forward = self.conv_first_forward(x_forward)
        x_backward = self.conv_first_backward(x_backward)

        x_forward = self.recon_trunk_forward(x_forward)
        x_backward = self.recon_trunk_backward(x_backward)

        hidden_forward = self.conv_hidden_forward(x_forward)
        hidden_backward = self.conv_hidden_backward(x_backward)
        hidden = tf.concat((hidden_forward, hidden_backward), axis=-1)

        out = tf.concat((x_forward, x_backward), axis=-1)
        out = self.conv_fusion(out)
        out = self.recon_trunk(out)
        out = self.conv_last(out)

        out = tf.nn.depth_to_space(out, self.scale)
        bilinear = tf.image.resize(x2, size=(shape[1] * self.scale, shape[2] * self.scale))
        out = out + bilinear

        if not training:
            out = tf.clip_by_value(out, 0, 255)

        return out, hidden


    def model(self):
        x1 = Input(shape=(None, None, 9))
        x2 = Input(shape=(None, None, self.base_channels*2))
        return Model(inputs=[x1, x2], outputs=self.call([x1, x2]))




class ConvBlock(tf.keras.Model):
    """block."""

    def __init__(self, base_channels):
        """Initialize `Block`.

        Args:
            base_channels: An `int` represents the number of base channels.
        """
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            base_channels, kernel_size=3, strides=1, padding='SAME', activation='relu',
            kernel_initializer=glorot_normal(), bias_initializer='zeros'
        )

    def call(self, x):
        """Forward the given input.

        Args:
            x: An input `Tensor`.

        Returns:
            An output `Tensor`.
        """
        
        return self.conv(x)


def make_layer(basic_block, block_num, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block: A `nn.module` represents the basic block.
        block_num: An `int` represents the number of blocks.

    Returns:
        An `nn.Sequential` stacked blocks.
    """
    model = tf.keras.Sequential()
    for _ in range(block_num):
        model.add(basic_block(**kwarg))
    return model
