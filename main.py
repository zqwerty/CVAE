import os
import tensorflow as tf
import numpy as np
import time
from CVAE import CVAE
from utils import data_process, build_vocab, train_batch, eval_batches, infer
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_post", "weibo_pair.post", "data_post")
tf.app.flags.DEFINE_string("data_response", "weibo_pair.response", "data_response")
# tf.app.flags.DEFINE_string("data_ref", "sim_response70w", "data_ref response")
tf.app.flags.DEFINE_boolean("split", True, "whether the data have been split in to train/dev/test")
tf.app.flags.DEFINE_integer("train_size", 690000, "train_size")
tf.app.flags.DEFINE_integer("valid_size", 10000, "valid_size")
tf.app.flags.DEFINE_integer("test_size", 5000, "test_size")
tf.app.flags.DEFINE_string("word_vector", "../vector.txt", "word vector")

tf.app.flags.DEFINE_string("data_dir", "../weibo_pair", "data_dir")
tf.app.flags.DEFINE_string("train_dir", "./train2seq2seq10", "train_dir")
tf.app.flags.DEFINE_string("log_dir", "./log2seq2seq10", "log_dir")
tf.app.flags.DEFINE_string("attn_mode", "Luong", "attn_mode")
tf.app.flags.DEFINE_string("opt", "SGD", "optimizer")
tf.app.flags.DEFINE_string("infer_path_post", "../weibo_pair/test.weibo_pair.post", "path of the post file to be infer")
# tf.app.flags.DEFINE_string("infer_path_ref", "", "path of the ref file to be infer")
tf.app.flags.DEFINE_string("infer_path_resp", "./test.infer", "path of the resp file to be infer")
tf.app.flags.DEFINE_string("save_para_path", "", "path of the trained model, default latest in train_dir")

tf.app.flags.DEFINE_boolean("use_lstm", False, "use_lstm")
tf.app.flags.DEFINE_boolean("is_train", True, "is_train")
tf.app.flags.DEFINE_boolean("bi_encode", False, "bidirectional encoder")

tf.app.flags.DEFINE_integer("batch_size", 128, "batch_size")
tf.app.flags.DEFINE_integer("embed_size", 100, "embed_size")
tf.app.flags.DEFINE_integer("num_units", 512, "num_units")
tf.app.flags.DEFINE_integer("num_layers", 2, "num_layers")
# tf.app.flags.DEFINE_integer("recog_hidden_units", 512, "recognition network MLP hidden layer units")
# tf.app.flags.DEFINE_integer("prior_hidden_units", 512, "prior network MLP hidden layer units")
tf.app.flags.DEFINE_integer("z_dim", 128, "num_units")
# tf.app.flags.DEFINE_integer("full_kl_step", 50000, "kl_weight increase from 0 to 1 linearly in full_kl_step")
tf.app.flags.DEFINE_integer("beam_width", 5, "beam_width")
tf.app.flags.DEFINE_integer("max_decode_len", 128, "max_decode_len")
tf.app.flags.DEFINE_integer("vocab_size", 40000, "vocab_size")
tf.app.flags.DEFINE_integer("save_every_n_iteration", 1000, "save_every_n_iteration")

tf.app.flags.DEFINE_float("l2_loss_weight", 0.001, "l2 regularization weight")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "learning rate")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "learning rate decay factor")
tf.app.flags.DEFINE_float("momentum", 0.9, "momentum")
tf.app.flags.DEFINE_float("keep_prob", 0.8, "keep_prob")


def main(unused_argv):
    dp = data_process(FLAGS)
    print(FLAGS.__flags)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if FLAGS.is_train:
            train_data = dp.load_train_data()
            valid_data = dp.load_valid_data()
            # test_data = dp.load_test_data()
            vocab, embed = build_vocab(FLAGS.word_vector, FLAGS.embed_size, FLAGS.vocab_size, train_data)
            s2s = CVAE(embed=embed, tfFLAGS=FLAGS)
            if tf.train.get_checkpoint_state(FLAGS.train_dir):
                print("Reading model parameters from %s" % FLAGS.train_dir)
                s2s.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
                train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'))
            else:
                print("Created model with fresh parameters.")
                s2s.initialize(sess, vocab=vocab)
                train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'), sess.graph)

            valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'valid'))
            summary_list = ['ppl', 'sen_loss', 'l2_loss', 'loss']
            summary_num = len(summary_list)
            summary_placeholders = [tf.placeholder(tf.float32) for i in range(summary_num)]
            summary_op = [tf.summary.scalar(summary_list[i], summary_placeholders[i]) for i in range(summary_num)]

            summary_sum = np.zeros((summary_num,))
            time_step = 0
            previous_losses = [1e18] * 3
            while True:
                start_time = time.time()

                train_batched = train_batch(train_data, FLAGS.batch_size)
                ops = s2s.step(sess, train_batched, is_train=True)
                for i in range(summary_num):
                    summary_sum[i] += ops[i + 1]

                global_step = s2s.global_step.eval()
                time_step += (time.time() - start_time)

                if global_step % FLAGS.save_every_n_iteration == 0:
                    time_step /= FLAGS.save_every_n_iteration
                    summary_sum /= FLAGS.save_every_n_iteration

                    if FLAGS.opt == 'SGD' and summary_sum[1] > max(previous_losses):
                        sess.run(s2s.learning_rate_decay_op)
                    previous_losses = previous_losses[1:] + [summary_sum[1]]

                    summary_sum[0] = np.exp(summary_sum[0])
                    feed_dict = dict(zip(summary_placeholders, summary_sum))

                    summaries = sess.run(summary_op, feed_dict=feed_dict)
                    for s in summaries:
                        train_writer.add_summary(summary=s, global_step=global_step)
                    print("global step %d step-time %.4f learning_rate %f"
                          % (global_step,
                             time_step,
                             s2s.learning_rate.eval() if FLAGS.opt=='SGD' else .0))
                    print ''
                    for i in range(summary_num):
                        print 'train '+summary_list[i]+': %f' % summary_sum[i]

                    summary_sum = np.zeros((summary_num,))

                    for batch in eval_batches(valid_data, FLAGS.batch_size):
                        ops = s2s.step(sess, batch, is_train=False)
                        for i in range(summary_num):
                            summary_sum[i] += ops[i]

                    summary_sum /= FLAGS.valid_size // FLAGS.batch_size

                    summary_sum[0] = np.exp(summary_sum[0])
                    feed_dict = dict(zip(summary_placeholders, summary_sum))

                    summaries = sess.run(summary_op, feed_dict=feed_dict)
                    for s in summaries:
                        valid_writer.add_summary(summary=s, global_step=global_step)

                    print ''
                    for i in range(summary_num):
                        print 'valid '+summary_list[i] + ': %f' % summary_sum[i]

                    summary_sum = np.zeros((summary_num,))
                    time_step = 0

                    s2s.saver.save(sess, "%s/model.ckpt" % FLAGS.train_dir, global_step=global_step)

        else:
            if FLAGS.save_para_path=='':
                model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            else:
                model_path = FLAGS.save_para_path
            print model_path
            saver = tf.train.import_meta_graph(model_path + '.meta')
            saver.restore(sess, model_path)
            infer(sess, FLAGS.infer_path_post, FLAGS.infer_path_resp, FLAGS.batch_size)


if __name__ == '__main__':
    tf.app.run()
