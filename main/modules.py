import tensorflow as tf
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):


    def __init__(self, hidden_size, keep_prob):
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks, scopename):
        with vs.variable_scope(scopename):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out




class BiDAF(object):

    def __init__(self, keep_prob, vec_size):

        self.keep_prob = keep_prob
        self.vec_size = vec_size
        self.S_W = tf.get_variable('S_W', [vec_size*3], tf.float32,
            tf.contrib.layers.xavier_initializer())


    def build_graph(self, q, q_mask, c, c_mask):

        with vs.variable_scope("BiDAF"):

            c_expand = tf.expand_dims(c,2)  #[batch,N,1,2h]
            print(c_expand)
            q_expand = tf.expand_dims(q,1)  #[batch,1,M,2h]
            print(q_expand)
            c_pointWise_q = c_expand * q_expand  #[batch,N,M,2h]
            print(c_pointWise_q)

            c_input = tf.tile(c_expand, [1, 1, tf.shape(q)[1], 1])
            q_input = tf.tile(q_expand, [1, tf.shape(c)[1], 1, 1])

            concat_input = tf.concat([c_input, q_input, c_pointWise_q], -1) # [batch,N,M,6h]
            print(concat_input)


            similarity=tf.reduce_sum(concat_input * self.S_W, axis=3)  #[batch,N,M]
            print(similarity)

            # Calculating context to question attention
            similarity_mask = tf.expand_dims(q_mask, 1) # shape (batch_size, 1, M)
            print(similarity_mask)
            _, c2q_dist = masked_softmax(similarity, similarity_mask, 2) # shape (batch_size, N, M). take softmax over q
            print(c2q_dist)

            # Use attention distribution to take weighted sum of values
            c2q = tf.matmul(c2q_dist, q) # shape (batch_size, N, vec_size)
            print(c2q)

            # Calculating question to context attention c_dash
            S_max = tf.reduce_max(similarity, axis=2) # shape (batch, N)
            print(S_max)
            _, c_dash_dist = masked_softmax(S_max, c_mask, 1) # distribution of shape (batch, N)
            print(c_dash_dist)
            c_dash_dist_expand = tf.expand_dims(c_dash_dist, 1) # shape (batch, 1, N)
            print(c_dash_dist_expand)
            c_dash = tf.matmul(c_dash_dist_expand, c) # shape (batch_size, 1, vec_size)
            print(c_dash)

            c_c2q = c * c2q # shape (batch, N, vec_size)
            print(c_c2q)

            c_c_dash = c * c_dash # shape (batch, N, vec_size)
            print(c_c_dash)

            # concatenate the output
            output = tf.concat([c2q, c_c2q, c_c_dash], axis=2) # (batch_size, N, vec_size * 3)
            print(output)



            output = tf.nn.dropout(output, self.keep_prob)
            print(output)

            return output


class SimpleSoftmaxLayer(object):

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):

        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):

    def __init__(self, keep_prob, key_vec_size, value_vec_size):

        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):

        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            print("Basic attn keys", keys.shape)
            print("Basic attn values", values_t.shape)
            print("Basic attn logits", attn_logits.shape)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):

    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
