import tensorflow as tf

class FeedForward:

    def __init__(self, layer_dims=None, activation=tf.nn.relu):
        self.layer_dims = layer_dims
        self.num_layers = len(self.layer_dims) - 1
        self.inputs = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.layer_dims[0]],
                                     name='inputs')
        self.targets = tf.placeholder(dtype=tf.float32,
                                      shape=[None, self.layer_dims[-1]],
                                      name='labels')
        self.keep_probs = [tf.placeholder(dtype=tf.float32, name='keep_prob'+str(_)) for _ in range(self.num_layers)]
        self.activation = activation
        self.weights = None
        self.biases = None


    def linear_layer(self, inputs, input_dim, output_dim):
        stddev_w = tf.div(1.0, tf.sqrt(tf.to_float(input_dim)))
        weights = tf.Variable(tf.truncated_normal(shape=[input_dim, output_dim],
                                                mean=0.0,
                                                stddev=stddev_w))
        self.weights.append(weights)
        stddev_b = tf.sqrt(tf.div(2.0, tf.to_float(input_dim)))
        biases = tf.Variable(tf.truncated_normal(shape=[output_dim],
                                                mean=0.0,
                                                stddev=stddev_b))
        self.biases.append(biases)
        op = tf.add(tf.matmul(inputs, weights), biases)
        return op


    def nonlinear_layer(self, inputs, input_dim, output_dim):
        linear_trans = self.linear_layer(inputs, input_dim, output_dim)
        nonlinear_trans = self.activation(linear_trans)
        return nonlinear_trans


    def get_output(self):
        self.weights = []
        self.biases = []

        h = self.nonlinear_layer(self.inputs, self.layer_dims[0], self.layer_dims[1])
        for i in range(1, self.num_layers-1):
            dropped_h = tf.nn.dropout(h, self.keep_probs[i])
            h = self.nonlinear_layer(dropped_h, self.layer_dims[i], self.layer_dims[i+1])
        dropped_h = tf.nn.dropout(h, self.keep_probs[-1])
        op = self.linear_layer(h, self.layer_dims[-2], self.layer_dims[-1])
        return op


    def get_targets(self):
        return self.targets


    def get_train_feed_dict(self, inputs, targets=None, keep_probs=None):
        if len(keep_probs) != self.num_layers:
            raise ValueError('dropout keep_prob must be defined for all layers except the last')
        feed_dict = {self.keep_probs[i]: prob for i, prob in zip(range(self.num_layers), keep_probs)}
        feed_dict[self.inputs] = inputs
        feed_dict[self.targets] = targets
        return feed_dict


    def get_test_feed_dict(self, inputs):
        keep_probs = [1.0 for _ in range(self.num_layers)]
        feed_dict = {self.keep_probs[i]: prob for i, prob in zip(range(self.num_layers), keep_probs)}
        feed_dict[self.inputs] = inputs
        return feed_dict
