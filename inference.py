import tensorflow as tf
slim = tf.contrib.slim

def inference(X, Y, is_training=True):
    with slim.arg_scope([slim.model_variable], device='/cpu:0'):
        prediction, tensor_collection = prime_classifier(inputs = X, is_training = is_training)
        tf.losses.sigmoid_cross_entropy(Y, prediction)
        losses = tf.get_collection(key=tf.GraphKeys.LOSSES)
        regularization_losses = tf.get_collection(key=tf.GraphKeys.LOSSES)
        total_loss = tf.add_n(inputs = [losses] + [regularization_losses], name = 'total_loss')
        return total_loss, prediction


def prime_classifier(inputs, is_training=True, scope="deep_regression"):
    """Creates the prime classification model.

    Args:
        inputs: A node that yields a `Tensor` of size [batch_size, dimensions].
        is_training: Whether or not we're currently training the model.
        scope: An optional variable_op scope for the model.

    Returns:
        predictions: 1-D `Tensor` of shape [batch_size] of responses.
        tensor_collection: A dict of end points representing the hidden layers.
    """
    with tf.variable_scope(scope, 'deep_regression', [inputs]):
        tensor_collection = {}
        # Set the default weight _regularizer and acvitation for each fully_connected layer.
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.sigmoid,
                            weights_regularizer=slim.l2_regularizer(0.01)):

            # Creates a fully connected layer from the inputs with 32 hidden units.
            net = slim.fully_connected(inputs, 15, scope='fc1')
            tensor_collection['fc1'] = net

            # Adds a dropout layer to prevent over-fitting.
            net = slim.dropout(net, 0.8, is_training=is_training)

            # Adds another fully connected layer with 16 hidden units.
            net = slim.fully_connected(net, 30, scope='fc2')
            tensor_collection['fc2'] = net

            # Creates a fully-connected layer with a single hidden unit. Note that the
            # layer is made linear by setting activation_fn=None.
            predictions = slim.fully_connected(net, 1, activation_fn=None, scope='prediction')
            tensor_collection['out'] = predictions

            return predictions, tensor_collection


