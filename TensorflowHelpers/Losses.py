# some usefull losses
import tensorflow as tf

def l2(pred, true, name="loss_l2"):
    return tf.nn.l2_loss(pred-true, name=name)

def rmse(pred, true, name="loss_rmse"):
    return tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.pow(pred-true, 2), axis=1), name="mse"), name=name)

def pinball(pred, true, q, name="loss_pinball"):
    loss = tf.abs(pred-true, name="abs")
    loss = tf.add(loss, q*tf.cast(tf.greater(pred, true), tf.float32), name="add_upper_case")
    loss = tf.add(loss, (1.0-q)*tf.cast(tf.less(pred, true), tf.float32), name="add_lower_case")
    loss = tf.reduce_sum(loss, axis=1, name="sum_var")
    loss = tf.reduce_mean(loss, name="mean_examples")
    loss = tf.identity(loss, name=name)
    return loss

def pinball_multi_q(pred, true, qs, name="loss_pinball"):
    loss = tf.constant(0., tf.float32)
    for q in qs:
        loss = tf.add(loss,pinball(pred, true, q))
    loss = tf.identity(loss, name=name)
    return loss
