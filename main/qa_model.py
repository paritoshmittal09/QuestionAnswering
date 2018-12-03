from __future__ import absolute_import
from __future__ import division
import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate_paritosh import exact_match_score, f1_score
from modules import  *

logging.basicConfig(level=logging.INFO)


class QAModel(object):
    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        print "Initializing the QAModel..."
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        _, _, num_chars = self.create_char_dicts()
        self.char_vocab = num_chars

        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_graph()
            self.add_loss()

        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)


        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_placeholders(self):


        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())



    def add_embedding_layer(self, emb_matrix):

        with vs.variable_scope("embeddings"):

            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") 

            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)


    def build_graph(self):

		last_dim_concat = self.context_embs.get_shape().as_list()[-1]
        for i in range(2):
            self.context_embs = self.highway(self.context_embs, last_dim_concat, scope_name='highway', carry_bias=-1.0)
                #reuse variables for qn_embs
            self.qn_embs = self.highway(self.qn_embs, last_dim_concat, scope_name='highway', carry_bias=-1.0)

        encoder = RNNEncoder(self.FLAGS.hidden_size_encoder, self.keep_prob)
        context_hiddens = encoder.build_graph(self.context_embs, self.context_mask, scopename='RNNEncoder') # (batch_size, context_len, hidden_size*2)
        question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask, scopename='RNNEncoder') # (batch_size, question_len, hidden_size*2)


        if self.FLAGS.bidaf_attention:

            attn_layer = BiDAF(self.keep_prob, self.FLAGS.hidden_size_encoder * 2)
            attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens,
                                                 self.context_mask)  # attn_output is shape (batch_size, context_len, hidden_size_encoder*6)

            self.bidaf_attention = attn_output
            self.bidaf_attention = tf.reduce_max(self.bidaf_attention, axis=2)  # shape (batch_size, seq_len)
            print("Shape bidaf before softmax", self.bidaf_attention.shape)

            # Take softmax over sequence
            _, self.bidaf_attention_probs = masked_softmax(self.bidaf_attention, self.context_mask, 1)  


            blended_reps = tf.concat([context_hiddens, attn_output], axis=2)  # (batch_size, context_len, hidden_size_encoder*8)

        else: 


            last_dim = context_hiddens.get_shape().as_list()[-1]
            print("last dim", last_dim)

            attn_layer = BasicAttn(self.keep_prob, last_dim,
                                   last_dim)
            _, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask,
                                                    context_hiddens)  

            blended_reps = tf.concat([context_hiddens, attn_output], axis=2)  


		
		
			
		if self.FLAGS.add_Modelling_LSTM:	
			            ## add a modeling layer
            modeling_layer = RNNEncoder(self.FLAGS.hidden_size_modeling, self.keep_prob)
            attention_hidden = modeling_layer.build_graph(blended_reps,
                                                  self.context_mask, scopename='add_Modelling_LSTM') 

            blended_reps = attention_hidden # for the final layer 
		
		
		blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hiddsen_size_fully_connected)

       with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask)

       with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask)


    def add_loss(self):
        with vs.variable_scope("loss"):

            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
            tf.summary.scalar('loss_start', self.loss_start) # log to tensorboard

            # Calculate loss for prediction of end position
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # Add the two losses
            self.loss = self.loss_start + self.loss_end
            tf.summary.scalar('loss', self.loss)


    def run_train_iter(self, session, batch, summary_writer):
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout


        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span


        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask

        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end
        
	def get_start_end_pos(self, session, batch):
		start_dist, end_dist = self.get_prob_dists(session, batch)
    	maxprob = 0
    	if self.FLAGS.smart_span:

            curr_batch_size = batch.batch_size
            start_pos = np.empty(shape = (curr_batch_size), dtype=int)
            end_pos = np.empty(shape=(curr_batch_size), dtype=int)
            maxprob = np.empty(shape=(curr_batch_size), dtype=float)

            for j in range(curr_batch_size):  # for each row
            ## Take argmax of start and end dist in a window such that  i <= j <= i + 15
                maxprod = 0
                chosen_start = 0
                chosen_end = 0
                for i in range(self.FLAGS.context_len-16):
                    end_dist_subset = end_dist[j,i:i+16]
                    end_prob_max = np.amax(end_dist_subset)
                    end_idx = np.argmax(end_dist_subset)
                    start_prob = start_dist[j,i]
                    prod = end_prob_max*start_prob
                    if prod > maxprod:
                        maxprod = prod
                        chosen_start = i
                        chosen_end = chosen_start+end_idx

                start_pos[j] = chosen_start
                end_pos[j] = chosen_end
                maxprob[j] = round(maxprod,4)

		else:
			start_pos = np.argmax(start_dist, axis=1)
            end_pos = np.argmax(end_dist, axis=1)
            
        return start_pos, end_pos, maxprob

    def get_attention_dist(self, session, batch):


        start_dist, end_dist = self.get_prob_dists(session, batch)

        return start_dist




    def matrix_multiplication(self, mat, weight):
        # [batch_size, seq_len, hidden_size] * [hidden_size, p] = [batch_size, seq_len, p]

        mat_shape = mat.get_shape().as_list()  
        weight_shape = weight.get_shape().as_list()  
        assert (mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]])  # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight)  # [batch_size * n, p]

    def highway(self, x, size, scope_name, carry_bias=-1.0):

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            W_T = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight_transform")
            b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")

            W = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight")
            b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")

        T = tf.sigmoid(self.matrix_multiplication(x, W_T) + b_T, name="transform_gate")
        H = tf.nn.relu(self.matrix_multiplication(x, W) + b, name="activation")

        print("shape H, T: ", H.shape, T.shape)
        C = tf.subtract(1.0, T, name="carry_gate")

        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
        return y


    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):

        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=False):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic)

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=1000):
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0

        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=False):

            pred_start_pos, pred_end_pos, _ = self.get_start_end_pos(session, batch)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist() # list length batch_size
            pred_end_pos = pred_end_pos.tolist() # list length batch_size

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em



                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):

        # Print number of model parameters
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        get_start_end_posget_start_end_posbestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            # Loop over batches
            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

                # Run training iteration
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)

                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                if global_step % self.FLAGS.eval_every == 0:


                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)



        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
