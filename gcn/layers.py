import scipy.sparse as sp
import tensorflow as tf

from gcn import utils
from gcn.inits import *

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}
log = tf.get_logger()


def get_layer_uid(layer_name=""):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.compat.v1.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.compat.v1.sparse_retain(x, dropout_mask)
    return pre_out * (1.0 / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.compat.v1.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def batch_norm(x, n_out, phase_train, scope="bn"):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.compat.v1.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name="beta", trainable=True)
        gamma = tf.Variable(
            tf.constant(1.0, shape=[n_out]), name="gamma", trainable=True
        )
        batch_mean, batch_var = tf.nn.moments(x, [0, 1], name="moments")
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(
            phase_train,
            mean_var_with_update,
            lambda: (ema.average(batch_mean), ema.average(batch_var)),
        )
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {"name", "logging"}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, "Invalid keyword argument: " + kwarg
        name = kwargs.get("name")
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + "_" + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get("logging", False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + "/inputs", inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + "/outputs", outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + "/vars/" + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""

    def __init__(
        self,
        input_dim,
        output_dim,
        placeholders,
        dropout=0.0,
        sparse_inputs=False,
        act=tf.nn.relu,
        bias=False,
        featureless=False,
        **kwargs,
    ):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders["dropout"]
        else:
            self.dropout = 0.0

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim
        self.phase_train = placeholders["phase_train"]

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders["num_features_nonzero"]

        with tf.compat.v1.variable_scope(self.name + "_vars"):
            self.vars["weights"] = glorot([input_dim, output_dim], name="weights")
            if self.bias:
                self.vars["bias"] = zeros([output_dim], name="bias")

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.compat.v1.nn.dropout(x, 1 - self.dropout)

        # transform
        output = dot(x, self.vars["weights"], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars["bias"]

        # mean, var = tf.nn.moments(output, [0, 1], name='moments')
        # output = tf.nn.batch_normalization(output, mean, var, offset=None, scale=None, variance_epsilon=1e-5)

        # output = tf.compat.v1.contrib.layers.batch_norm(inputs=output, decay=0.999, center=False, scale=False, epsilon=1e-3,
        #                                       updates_collections=None,is_training=self.phase_train,reuse=False,
        #                                       fused=False)

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(
        self,
        input_dim,
        output_dim,
        placeholders,
        dropout=0.0,
        sparse_inputs=False,
        act=tf.nn.relu,
        bias=False,
        featureless=False,
        **kwargs,
    ):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders["dropout"]
        else:
            self.dropout = 0.0

        self.act = act
        self.support = placeholders["support"]
        self.phase_train = placeholders["phase_train"]
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders["num_features_nonzero"]

        with tf.compat.v1.variable_scope(self.name + "_vars"):
            for i in range(len(self.support)):
                self.vars["weights_" + str(i)] = glorot(
                    [input_dim, output_dim], name="weights_" + str(i)
                )
            if self.bias:
                self.vars["bias"] = zeros([output_dim], name="bias")

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.compat.v1.nn.dropout(x, 1 - self.dropout)
        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(
                    x, self.vars["weights_" + str(i)], sparse=self.sparse_inputs
                )
            else:
                pre_sup = self.vars["weights_" + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # mean, var = tf.nn.moments(output,[0)
        # output = tf.nn.batch_normalization(output, mean, var)

        # bias
        if self.bias:
            output += self.vars["bias"]

        # output = tf.compat.v1.contrib.layers.batch_norm(inputs=output, decay=0.999, center=False, scale=False, epsilon=1e-3,
        #                                       updates_collections=None,is_training=self.phase_train,reuse=False,
        #                                       fused=False)

        # mean, var = tf.nn.moments(output, [0, 1], name='moments')
        # output = tf.nn.batch_normalization(output, mean, var, offset = None, scale = None, variance_epsilon=1e-5)

        return self.act(output)


class CayleyConvolution(Layer):
    """Graph convolution layer using Cayley polynomials with learnable spectral zoom."""

    def __init__(
        self,
        input_dim,
        output_dim,
        placeholders,
        adj_normalized,
        order,
        jacobi_iteration,
        h_update_threshold=1e-3,  # Threshold for support update
        dropout=0.0,
        sparse_inputs=False,
        act=tf.nn.relu,
        bias=False,
        featureless=False,
        **kwargs,
    ):
        super(CayleyConvolution, self).__init__(**kwargs)

        self.adj_normalized = utils.scipy_sparse_to_tf_sparse(adj_normalized)
        self.order = order
        self.jacobi_iterations = jacobi_iteration

        self.h_update_threshold = h_update_threshold

        # # Cache for support matrices
        # self.cached_support = None
        # self.cached_h = None

        if dropout:
            self.dropout = placeholders["dropout"]
        else:
            self.dropout = 0.0

        self.act = act
        # self.support = placeholders["support"] not needed here
        self.phase_train = placeholders["phase_train"]
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders["num_features_nonzero"]

        # Initialize variables
        with tf.compat.v1.variable_scope(self.name + "_vars"):
            # Spectral zoom parameter
            self.vars["h"] = tf.Variable(1.0, name="spectral_zoom")

            # Real weight for j=0
            self.vars["weights_0"] = glorot([input_dim, output_dim], name="weights_0")

            # Complex weights for each power j≥1
            for i in range(1, order + 1):
                self.vars[f"weights_{i}_real"] = glorot(
                    [input_dim, output_dim], name=f"weights_{i}_real"
                )
                self.vars[f"weights_{i}_imag"] = glorot(
                    [input_dim, output_dim], name=f"weights_{i}_imag"
                )

    # def _get_support(self):
    #     """Get support matrices using TF control flow."""
    #     should_update = tf.logical_or(
    #         tf.logical_not(self.vars["is_initialized"]),
    #         tf.abs(self.vars["h"] - self.vars["previous_h"]) > self.h_update_threshold,
    #     )

    #     def update_support():
    #         new_support = self._compute_support()

    #         update_ops = []
    #         for i, support_matrix in enumerate(new_support):
    #             nnz = tf.shape(support_matrix.indices)[0]

    #             # Cast types explicitly
    #             indices = tf.cast(support_matrix.indices, tf.int64)
    #             values = tf.cast(support_matrix.values, tf.complex64)

    #             # Create padding with matching types
    #             indices_padding = tf.zeros(
    #                 [tf.maximum(0, self.max_nnz - nnz), 2], dtype=tf.int64
    #             )
    #             values_padding = tf.zeros(
    #                 [tf.maximum(0, self.max_nnz - nnz)], dtype=tf.complex64
    #             )

    #             # Concatenate instead of pad
    #             padded_indices = tf.concat([indices, indices_padding], axis=0)
    #             padded_values = tf.concat([values, values_padding], axis=0)

    #             update_ops.extend(
    #                 [
    #                     tf.compat.v1.assign(
    #                         self.vars[f"cached_support_{i}_indices"], padded_indices
    #                     ),
    #                     tf.compat.v1.assign(
    #                         self.vars[f"cached_support_{i}_values"], padded_values
    #                     ),
    #                     tf.compat.v1.assign(
    #                         self.vars[f"cached_support_{i}_nnz"], tf.cast(nnz, tf.int32)
    #                     ),
    #                     tf.compat.v1.assign(
    #                         self.vars[f"cached_support_{i}_dense_shape"],
    #                         tf.cast(support_matrix.dense_shape, tf.int64),
    #                     ),
    #                 ]
    #             )

    #         update_ops.extend(
    #             [
    #                 tf.compat.v1.assign(self.vars["previous_h"], self.vars["h"]),
    #                 tf.compat.v1.assign(self.vars["is_initialized"], True),
    #             ]
    #         )

    #         with tf.control_dependencies(update_ops):
    #             return new_support

    #     def get_cached():
    #         return [
    #             tf.SparseTensor(
    #                 indices=self.vars[f"cached_support_{i}_indices"][
    #                     : self.vars[f"cached_support_{i}_nnz"]
    #                 ],
    #                 values=self.vars[f"cached_support_{i}_values"][
    #                     : self.vars[f"cached_support_{i}_nnz"]
    #                 ],
    #                 dense_shape=self.vars[f"cached_support_{i}_dense_shape"],
    #             )
    #             for i in range(self.order + 1)
    #         ]

    #     return tf.cond(should_update, update_support, get_cached)

    def _compute_support(self):
        """Compute Cayley polynomial support matrices using TF sparse ops."""
        h = self.vars["h"]

        # Compute Laplacian in sparse format
        I = tf.sparse.eye(self.adj_normalized.shape[0], dtype=tf.float32)
        neg_adj = tf.sparse.map_values(lambda x: -x, self.adj_normalized)
        laplacian = tf.sparse.add(I, neg_adj)

        # Scale Laplacian by h
        h_lap = tf.sparse.map_values(lambda x: h * x, laplacian)

        # Convert to complex
        h_lap_c = tf.cast(h_lap, tf.complex64)
        I_c = tf.cast(I, tf.complex64)

        # Initialize support list with identity (C⁰)
        support = [I_c]

        # Keep everything sparse for Jacobi iterations
        prev_power = I_c
        for j in range(1, self.order + 1):
            # Compute (h∆ - iI)x while keeping sparsity
            x = tf.sparse.add(
                tf.sparse.sparse_dense_matmul(h_lap_c, tf.sparse.to_dense(prev_power)),
                tf.sparse.map_values(lambda x: -tf.complex(0.0, 1.0) * x, prev_power),
            )

            # Jacobi iterations in sparse format
            y_next = x
            for _ in range(self.jacobi_iterations):
                # Get diagonal elements
                diag = tf.sparse.reduce_sum(h_lap_c, axis=1) + tf.complex(0.0, 1.0)
                diag_inv = 1.0 / diag

                # Compute residual maintaining sparsity where possible
                h_lap_y = tf.sparse.sparse_dense_matmul(h_lap_c, y_next)
                y_i = tf.complex(0.0, 1.0) * y_next
                residual = x - (h_lap_y + y_i)

                # Update
                y_next = y_next + tf.multiply(diag_inv[:, None], residual)

            # Convert to sparse before adding to support
            y_next_sparse = tf.sparse.from_dense(y_next)
            support.append(y_next_sparse)
            prev_power = y_next_sparse

        return support

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.compat.v1.nn.dropout(x, 1 - self.dropout)

        # Get support matrices
        support = self._compute_support()

        # First term (j=0) is real
        if not self.featureless:
            pre_sup = dot(x, self.vars["weights_0"], sparse=self.sparse_inputs)
        else:
            pre_sup = self.vars["weights_0"]

        pre_sup_complex = tf.cast(pre_sup, tf.complex64)
        output = tf.sparse.sparse_dense_matmul(support[0], pre_sup_complex)
        output = tf.cast(tf.math.real(output), tf.float32)

        # Complex terms (j≥1)
        complex_sum = tf.zeros_like(output, dtype=tf.complex64)

        for i in range(1, len(support)):
            if not self.featureless:
                weights = tf.complex(
                    self.vars[f"weights_{i}_real"], self.vars[f"weights_{i}_imag"]
                )
                x_complex = (
                    tf.cast(x, tf.complex64)
                    if not self.sparse_inputs
                    else tf.sparse.map_values(lambda v: tf.cast(v, tf.complex64), x)
                )
                pre_sup = dot(x_complex, weights, sparse=self.sparse_inputs)
            else:
                pre_sup = tf.complex(
                    self.vars[f"weights_{i}_real"], self.vars[f"weights_{i}_imag"]
                )

            # Keep using sparse matrix multiplication
            term = tf.sparse.sparse_dense_matmul(support[i], pre_sup)
            complex_sum += term

            # Optional: clear the support matrix from memory after use
            support[i] = None

        output = output + 2 * tf.cast(tf.math.real(complex_sum), tf.float32)
        return self.act(output)
