import tensorflow as tf
from inference import inference
slim = tf.contrib.slim
from data_utils import make_df
from data_utils import BinaryPrimalityRandomIterator
from tqdm import tqdm
import pdb

def _setup_training_op(X, Y, global_step, optimizer):
    """Sets up inference (predictions), loss calculation, and minimization
    based on the input optimizer.
    Args:
        bin_rep: Batch of preprocessed examples dequeued from the input
            pipeline.
        primality: Confidence maps of ground truth joints.
        global_step: Training step counter.
        optimizer: Optimizer to minimize the loss function.
    Returns: Operation to run a training step.
    """
    with tf.device(device_name_or_function='/gpu:0'):
        total_loss, logits = inference(X, Y, is_training=True)
        grads = optimizer.compute_gradients(loss = total_loss)

    apply_gradient_op = optimizer.apply_gradients(grads_and_vars=grads, global_step=global_step)

    return apply_gradient_op, total_loss, logits


def setup_training(X, Y):
    """Sets up the entire training graph, including input pipeline, inference
    back-propagation and summaries.
    Args:
        FLAGS: The set of flags passed via command line. See the definitions at
            the top of this file.
    Returns:
        (num_batches_per_epoch, train_op, train_loss, global_step) tuple needed
        to run training steps.
    """
    ############### Setup Optimizer
    global_step = tf.get_variable(
        name='global_step',
        shape=[],
        dtype=tf.int64,
        initializer=tf.constant_initializer(0),
        trainable=False)

    RMSPROP_DECAY = 0.9
    RMSPROP_MOMENTUM = 0.9
    RMSPROP_EPSILON = 1.0

    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01,
                                          decay=RMSPROP_DECAY,
                                          momentum=RMSPROP_MOMENTUM,
                                          epsilon=RMSPROP_EPSILON)
    #

    train_op, train_loss, train_logits = _setup_training_op(X, Y, global_step, optimizer)

    return train_op, train_loss, train_logits, global_step


def train():
    """
    """
    with tf.device('/cpu:0'):
        X = tf.placeholder(shape=(None, 15), dtype = tf.float64, name = 'Binary_Representation')
        Y = tf.placeholder(shape=(None, 1), dtype = tf.int32, name = 'Primality')

        train_op, train_loss, train_logits, global_step = setup_training(X,Y)

        # Validation subgraph
        with tf.device(device_name_or_function='/gpu:0'):
          valid_loss_op, val_logits = inference(X,Y)

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        Data = make_df()
        train_data = Data[0:int(0.8*len(Data))]
        valid_data = Data[int(0.8*len(Data)):len(Data)]
        train_iterator = BinaryPrimalityRandomIterator(train_data, batch_size=64)
        valid_iterator = BinaryPrimalityRandomIterator(valid_data, batch_size=64)
        train_losses, valid_losses = [], []
        # This loop runs through all the data for n_epochs many times
        for current_epoch in range(10):
            # This loop runs through all the data once in steps given by self.batch_size
            # Better would be to make the train_iterator a generator
            # Fix this
            step, mean_loss = 0, 0
            curr_train_epoch = train_iterator.epoch
            while train_iterator.epoch < curr_train_epoch:
                step += 1
                batch = train_iterator.next_batch(increment=True)
                feed = {X: batch[1], Y: batch[2]} # Element zero is the number written in the decimal system.
                _, mean_loss_batch = sess.run([train_op, train_loss], feed_dict = feed)
                print(mean_loss_batch)
                mean_loss += mean_loss_batch
                train_losses.append(mean_loss / step)

            curr_valid_epoch = valid_iterator.epoch
            step, mean_loss = 0, 0
            while valid_iterator.epoch == curr_valid_epoch:
                step += 1 #Just for taking the mean at the end of each epoch
                batch = valid_iterator.next_batch(increment=True)
                feed = {X: batch[1], Y: batch[2]}
                mean_loss_batch, valid_pred = sess.run([valid_loss_op, val_logits], feed_dict=feed)
                mean_loss += mean_loss_batch
                valid_losses.append(mean_loss / step)
            print('Accuracy after epoch', current_epoch,
                  ' - train loss:', train_losses,
                  '- validation loss:', valid_losses)
            #print('Prediction:', valid_pred)
            #print('Real Output:', batch)

        return train_losses, valid_losses

if __name__=="__main__":
    train()
