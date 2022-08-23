"""Convert keras model to tflite."""

import argparse

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.lite.python import interpreter as interpreter_wrapper

from util import plugin


def _parse_argument():
    """Return arguments for conversion."""
    parser = argparse.ArgumentParser(description='Conversion.')
    parser.add_argument('--model_path', help='Path of model file.', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model class.', type=str, required=True)
    parser.add_argument(
        '--input_shapes', help='Series of the input shapes split by `:`.', required=True
    )
    parser.add_argument('--ckpt_path', help='Path of checkpoint.', type=str, required=True)
    parser.add_argument('--output_tflite', help='Path of output tflite.', type=str, required=True)

    args = parser.parse_args()

    return args


def main(args):
    """Run main function for converting keras model to tflite.

    Args:
        args: A `dict` contain augments.
    """
    # prepare model
    model_builder = plugin.plugin_from_file(args.model_path, args.model_name, tf.keras.Model)
    model = model_builder()

    # load checkpoint
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(args.ckpt_path).expect_partial()

    model_ = model.model()
    model_.summary()
    model_.save(f'{args.model_name}_pd')

    input_tensors = []
    for input_shape in args.input_shapes.split(':'):
        input_shape = list(map(int, input_shape.split(',')))
        input_shape = [None if x == -1 else x for x in input_shape]
        input_tensor = Input(shape=input_shape[1:], batch_size=input_shape[0])
        input_tensors.append(input_tensor)

    print(input_tensors[0].shape, input_tensors[1].shape)
    model = tf.keras.models.load_model(f'{args.model_name}_pd')
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape(input_tensors[0].shape)
    concrete_func.inputs[1].set_shape(input_tensors[1].shape)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    # model.build(input_shape=[input_tensors[0].shape, input_tensors[1].shape])
    # model_(input_tensors)
    # model.summary()

    # convert the keras model
    # converter = tf.lite.TFLiteConverter.from_keras_model(model_)
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True
    tflite_model = converter.convert()

    # save the tflite
    with open(args.output_tflite, 'wb') as f:
        f.write(tflite_model)

    interpreter = interpreter_wrapper.Interpreter(model_path=args.output_tflite, num_threads=32)

    input_details = interpreter.get_input_details()
    print(input_details)

    output_details = interpreter.get_output_details()
    print(output_details)

if __name__ == '__main__':
    arguments = _parse_argument()
    main(arguments)