from __future__ import absolute_import
from __future__ import division

import os
import io
import json
import sys
import logging

import tensorflow as tf

from qa_model import QAModel
from vocab import get_mapping


logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") 
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") 


# High-level options
tf.app.flags.DEFINE_integer("gpu", 0,)
tf.app.flags.DEFINE_string("mode", "train") # or test
tf.app.flags.DEFINE_string("experiment_name", "MSR", )
tf.app.flags.DEFINE_integer("num_epochs", 3)

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "")
tf.app.flags.DEFINE_float("dropout", 0.4, "")
tf.app.flags.DEFINE_integer("batch_size", 60, "")
tf.app.flags.DEFINE_integer("hidden_size_encoder", 150, "") 
tf.app.flags.DEFINE_integer("hidden_size_fully_connected", 200, "")
tf.app.flags.DEFINE_integer("context_len", 225, "")
tf.app.flags.DEFINE_integer("question_len", 35, "")
tf.app.flags.DEFINE_integer("embedding_size", 100, "")

tf.app.flags.DEFINE_bool("bidaf_attention", True, "")
tf.app.flags.DEFINE_bool("add_Modelling_LSTM", True, "") # False for fully connected layer
tf.app.flags.DEFINE_bool("smart_span", True, "")  

tf.app.flags.DEFINE_integer("hidden_size_modeling", 150, "Size of modeling layer")  #forbidaf

# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every",10, "")
tf.app.flags.DEFINE_integer("save_every", 10, "")
tf.app.flags.DEFINE_integer("eval_every", 10, "")
tf.app.flags.DEFINE_integer("keep", 1, "")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "")

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def initialize_model(session, model, dir,mode="train"):
	if mode == "test":
		ckpt = tf.train.get_checkpoint_state(dir)
    	v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    	if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
    		model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
		session.run(tf.global_variables_initializer())


def main(unused_argv):
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    # Define train_dir
    if not FLAGS.experiment_name and not FLAGS.train_dir:
        raise Exception("You need to specify either --experiment_name or --train_dir")
        
    FLAGS.train_dir = FLAGS.train_dir or os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)
    bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")

    FLAGS.glove_path = FLAGS.data_dir + "/SG_gensim.txt" # Change depending 
    emb_matrix, word2id, id2word = get_mapping(FLAGS.glove_path, FLAGS.embedding_size)

    train_context_path = FLAGS.data_dir + "/train.context"
    train_qn_path = FLAGS.data_dir + "/train.question"
    train_ans_path = FLAGS.data_dir + "/train.span"
    dev_context_path = FLAGS.data_dir + "/dev.context"
    dev_qn_path = FLAGS.data_dir + "/dev.question"
    dev_ans_path = FLAGS.data_dir + "/dev.span"
    test_context_path = FLAGS.data_dir + "/test.context"
    test_qn_path = FLAGS.data_dir + "/test.question"
    test_ans_path = FLAGS.data_dir + "/test.span"

    qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix)

    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if FLAGS.mode == "train":

        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
            json.dump(FLAGS.__flags, fout)

        if not os.path.exists(bestmodel_dir):
            os.makedirs(bestmodel_dir)

        with tf.Session(config=config) as sess:
            initialize_model(sess, qa_model, FLAGS.train_dir)
            qa_model.train(sess, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path)
    elif Flags.mode == "test":
    	with tf.Session(config=config) as sess:
            initialize_model(sess, qa_model, bestmodel_dir, "test")	
			f1,EM = qa_model.check_f1_em(sess, test_context_path, test_qn_path, test_ans_path, "test", num_samples=100)
    else:
        raise Exception("Unexpected values %s" % FLAGS.mode)

	

if __name__ == "__main__":
    tf.app.run()
