import random
import os
import tensorflow as tf
import numpy as np
import time
from CVAE import CVAE, _START_VOCAB, _PAD, _EOS
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_post", "weibo_pair.post", "data_post")
tf.app.flags.DEFINE_string("data_response", "weibo_pair.response", "data_response")
tf.app.flags.DEFINE_string("data_ref", "sim_response70w", "data_ref response")
tf.app.flags.DEFINE_boolean("split", True, "whether the data have been split in to train/dev/test")
tf.app.flags.DEFINE_integer("train_size", 690000, "train_size")
tf.app.flags.DEFINE_integer("valid_size", 10000, "valid_size")
tf.app.flags.DEFINE_integer("test_size", 5000, "test_size")
tf.app.flags.DEFINE_string("word_vector", "../../../vector.txt", "word vector")

tf.app.flags.DEFINE_string("data_dir", "../../../weibo_pair", "data_dir")
tf.app.flags.DEFINE_string("train_dir", "./train70w3", "train_dir")
tf.app.flags.DEFINE_string("log_dir", "./log70w3", "log_dir")
tf.app.flags.DEFINE_string("attn_mode", "Luong", "attn_mode")
tf.app.flags.DEFINE_string("opt", "SGD", "optimizer")
tf.app.flags.DEFINE_string("infer_path_post", "../../dataset/parsed/test.post", "path of the post file to be infer")
tf.app.flags.DEFINE_string("infer_path_ref", "", "path of the ref file to be infer")
tf.app.flags.DEFINE_string("infer_path_resp", "./test.infer", "path of the resp file to be infer")
tf.app.flags.DEFINE_string("save_para_path", "./train70w3/model.ckpt-00202000", "path of the trained model, default latest in train_dir")

tf.app.flags.DEFINE_boolean("use_lstm", False, "use_lstm")
tf.app.flags.DEFINE_boolean("is_train", True, "is_train")
tf.app.flags.DEFINE_boolean("bi_encode", True, "bidirectional encoder")

tf.app.flags.DEFINE_integer("batch_size", 128, "batch_size")
tf.app.flags.DEFINE_integer("embed_size", 100, "embed_size")
tf.app.flags.DEFINE_integer("num_units", 512, "num_units")
tf.app.flags.DEFINE_integer("num_layers", 1, "num_layers")
tf.app.flags.DEFINE_integer("recog_hidden_units", 512, "recognition network MLP hidden layer units")
tf.app.flags.DEFINE_integer("prior_hidden_units", 512, "prior network MLP hidden layer units")
tf.app.flags.DEFINE_integer("z_dim", 128, "num_units")
tf.app.flags.DEFINE_integer("full_kl_step", 50000, "kl_weight increase from 0 to 1 linearly in full_kl_step")
tf.app.flags.DEFINE_integer("beam_width", 5, "beam_width")
tf.app.flags.DEFINE_integer("max_decode_len", 128, "max_decode_len")
tf.app.flags.DEFINE_integer("vocab_size", 40000, "vocab_size")
tf.app.flags.DEFINE_integer("save_every_n_iteration", 1000, "save_every_n_iteration")

tf.app.flags.DEFINE_float("learning_rate", 0.5, "learning rate")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.95, "learning rate")
tf.app.flags.DEFINE_float("momentum", 0.9, "momentum")
tf.app.flags.DEFINE_float("keep_prob", 0.8, "keep_prob")


class data_process(object):
    def __init__(self,
                 tfFLAGS):
        self.data_dir = tfFLAGS.data_dir

        self.train_post = os.path.join(self.data_dir, 'gen/train70w_post')
        self.train_response = os.path.join(self.data_dir, 'gen/train70w_response')
        self.train_ref = os.path.join(self.data_dir, 'gen/train70w_ref')

        self.valid_post = os.path.join(self.data_dir, 'gen/valid70w_post')
        self.valid_response = os.path.join(self.data_dir, 'gen/valid70w_response')
        self.valid_ref = os.path.join(self.data_dir, 'gen/valid70w_ref')

        self.test_post = os.path.join(self.data_dir, 'gen/test70w_post')
        self.test_response = os.path.join(self.data_dir, 'gen/test70w_response')
        self.test_ref = os.path.join(self.data_dir, 'gen/test70w_ref')
        if not tfFLAGS.split:
            self.data_post = os.path.join(self.data_dir, tfFLAGS.data_post)
            self.data_response = os.path.join(self.data_dir, tfFLAGS.data_response)
            self.data_ref = os.path.join(self.data_dir, tfFLAGS.data_ref)
            self._split(tfFLAGS.train_size, tfFLAGS.valid_size, tfFLAGS.test_size)
        self.vocab_size = tfFLAGS.vocab_size

    def _split(self, train_size, valid_size, test_size):
        total_size = train_size+valid_size+test_size
        sel = random.sample(range(total_size), total_size)
        valid_dict = {}.fromkeys(sel[:valid_size])
        test_dict = {}.fromkeys(sel[-test_size:])

        train_post = open(self.train_post, 'wb')
        train_response = open(self.train_response, 'wb')
        train_ref = open(self.train_ref, 'wb')

        valid_post = open(self.valid_post, 'wb')
        valid_response = open(self.valid_response, 'wb')
        valid_ref = open(self.valid_ref, 'wb')

        test_from = open(self.test_post, 'wb')
        test_response = open(self.test_response, 'wb')
        test_ref = open(self.test_ref, 'wb')

        with open(self.data_post) as fpost, open(self.data_response) as fresp, open(self.data_ref) as fref:
            cntline = 0
            for post, resp, ref in zip(fpost,fresp,fref):
                ref = ref.split('\t')[1]+'\n'
                if cntline in valid_dict:
                    valid_post.write(post)
                    valid_response.write(resp)
                    valid_ref.write(ref)
                elif cntline in test_dict:
                    test_from.write(post)
                    test_response.write(resp)
                    test_ref.write(ref)
                else:
                    train_post.write(post)
                    train_response.write(resp)
                    train_ref.write(ref)
                cntline += 1
                if cntline == total_size:
                    break

        train_post.close()
        train_response.close()
        train_ref.close()
        valid_post.close()
        valid_response.close()
        valid_ref.close()
        test_from.close()
        test_response.close()
        test_ref.close()
        print "split completed"
        raw_input()

    def build_vocab(self,
                    data):
        print("Creating vocabulary...")
        vocab = {}
        for i, pair in enumerate(data):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            for token in pair['post'] + pair['response'] + pair['ref']:
                if token in vocab:
                    vocab[token] += 1
                else:
                    vocab[token] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > self.vocab_size:
            vocab_list = vocab_list[:self.vocab_size]

        if not os.path.exists(FLAGS.word_vector):
            print("Cannot find word vectors")
            embed = None
            return vocab_list, embed

        print("Loading word vectors...")
        vectors = {}
        with open(FLAGS.word_vector) as f:
            for i, line in enumerate(f):
                if i % 100000 == 0:
                    print("    processing line %d" % i)
                s = line.strip()
                word = s[:s.find(' ')]
                vector = s[s.find(' ') + 1:]
                vectors[word] = vector

        embed = []
        pre_vector = 0
        for word in vocab_list:
            if word in vectors:
                vector = map(float, vectors[word].split())
                pre_vector += 1
            else:
                vector = np.zeros(FLAGS.embed_size, dtype=np.float32)
            embed.append(vector)
        embed = np.array(embed, dtype=np.float32)
        print ("%d word vectors pre-trained" % pre_vector)

        return vocab_list, embed

    def load_train_data(self):
        return self.load_data(self.train_post, self.train_response, self.train_ref)

    def load_valid_data(self):
        return self.load_data(self.valid_post, self.valid_response, self.valid_ref)

    def load_test_data(self):
        return self.load_data(self.test_post, self.test_response, self.test_ref)

    def load_data(self,
                  post_f,
                  resp_f,
                  ref_f):
        f1 = open(post_f)
        f2 = open(resp_f)
        f3 = open(ref_f)
        post = [line.strip().split() for line in f1.readlines()]
        response = [line.strip().split() for line in f2.readlines()]
        ref = [line.strip().split() for line in f3.readlines()]
        data = []
        for p, r, ri in zip(post, response, ref):
            data.append({'post': p, 'response': r, 'ref': ri})
        f1.close()
        f2.close()
        f3.close()
        return data

    def gen_batched_data(self,
                         data):
        post_len = max([len(item['post']) for item in data]) + 1
        decoder_len = max([len(item['response']) for item in data]) + 1
        ref_len = max([len(item['ref']) for item in data]) + 1

        posts, responses, posts_length, responses_length = [], [], [], []
        refs, refs_length = [],[]

        def padding(sent, l):
            return sent + [_EOS] + [_PAD] * (l - len(sent) - 1)

        for item in data:
            posts.append(padding(item['post'], post_len))
            responses.append(padding(item['response'], decoder_len))
            posts_length.append(len(item['post']) + 1)
            responses_length.append(len(item['response']) + 1)

            refs.append(padding(item['ref'], ref_len))
            refs_length.append(len(item['ref']) + 1)

        batched_data = {'post': np.array(posts),
                        'response': np.array(responses),
                        'post_len': posts_length,
                        'response_len': responses_length,
                        'ref': np.array(refs),
                        'ref_len': refs_length}
        return batched_data

    def train_batch(self, data, batch_size):
        selected_data = [random.choice(data) for _ in range(batch_size)]
        batched_data = self.gen_batched_data(selected_data)
        return batched_data

    def eval_batches(self, data, batch_size):
        for i in range(len(data)//batch_size):
            batched_data = self.gen_batched_data(data[i*batch_size:(i+1)*batch_size])
            yield batched_data

    def infer(self,
              sess):
        def cut_eos(sentence):
            if sentence.find('EOS') != -1:
                return sentence[:sentence.find('EOS')]
            return sentence
        if FLAGS.infer_path_post is not "" and FLAGS.infer_path_ref is not "":
            f1 = open(FLAGS.infer_path_post)
            post = [line.strip().split() for line in f1]
            f1.close()
            f1 = open(FLAGS.infer_path_ref)
            ref = [line.strip().split() for line in f1]
            f1.close()
            f2 = open(FLAGS.infer_path_resp, 'wb')

            data = []
            for (p, r) in zip(post,ref):
                data.append({'post': p, 'response': [], 'ref': r})

            id = 0
            for j in range((len(data)+FLAGS.batch_size-1) // FLAGS.batch_size):
                batch = self.gen_batched_data(data[j * FLAGS.batch_size:(j + 1) * FLAGS.batch_size])
                input_feed = {
                    'input/post_string:0': batch['post'],
                    'input/post_len:0': batch['post_len'],
                    'input/response_string:0': batch['response'],
                    'input/response_len:0': batch['response_len'],
                    'input/ref_string:0': batch['ref'],
                    'input/ref_len:0': batch['ref_len'],
                    'input/use_prior:0': True
                }
                # res = sess.run(['decode_1/inference:0', 'decode_2/beam_out:0'], input_feed)
                print id
                for i in range(len(batch['post_len'])):
                    print >> f2, 'post: ' + ' '.join(post[id])
                    print >> f2, 'ref: ' + ' '.join(ref[id])
                    res = sess.run(['decode_1/inference:0'], input_feed)
                    print >> f2, 'infer: ' + cut_eos(' '.join(res[0][i]))
                    res = sess.run(['decode_1/inference:0'], input_feed)
                    print >> f2, 'infer: ' + cut_eos(' '.join(res[0][i]))
                    res = sess.run(['decode_1/inference:0'], input_feed)
                    print >> f2, 'infer: ' + cut_eos(' '.join(res[0][i]))
                    res = sess.run(['decode_1/inference:0'], input_feed)
                    print >> f2, 'infer: ' + cut_eos(' '.join(res[0][i]))
                    res = sess.run(['decode_1/inference:0'], input_feed)
                    print >> f2, 'infer: ' + cut_eos(' '.join(res[0][i]))
                    # print >> f2, 'beam: ' + cut_eos(' '.join(res[1][i, :, 0]))
                    print >> f2, ''

                    id += 1

            f2.close()
        else:
            while True:
                infer_data = {}
                infer_data['post'] = raw_input('post > ').strip().split()
                infer_data['ref'] = raw_input('ref > ').strip().split()
                infer_data['response'] = '233'.strip().split()
                infer_data = [infer_data]
                batch = self.gen_batched_data(infer_data)
                input_feed = {
                    'input/post_string:0': batch['post'],
                    'input/post_len:0': batch['post_len'],
                    'input/response_string:0': batch['response'],
                    'input/response_len:0': batch['response_len'],
                    'input/ref_string:0': batch['ref'],
                    'input/ref_len:0': batch['ref_len'],
                    'input/use_prior:0': True
                }
                res = sess.run(['decode_1/inference:0'], input_feed)
                print 'infer: ' + cut_eos(' '.join(res[0][0]))
                res = sess.run(['decode_1/inference:0'], input_feed)
                print 'infer: ' + cut_eos(' '.join(res[0][0]))
                res = sess.run(['decode_1/inference:0'], input_feed)
                print 'infer: ' + cut_eos(' '.join(res[0][0]))
                res = sess.run(['decode_1/inference:0'], input_feed)
                print 'infer: ' + cut_eos(' '.join(res[0][0]))
                res = sess.run(['decode_1/inference:0'], input_feed)
                print 'infer: ' + cut_eos(' '.join(res[0][0]))

                # res = sess.run(['decode_1/inference:0', 'decode_2/beam_out:0'], input_feed)
                # inference = cut_eos(' '.join(res[0][0]))
                # beam_out = [cut_eos(' '.join(res[1][0, :, i])) for i in range(FLAGS.beam_width)]
                # print 'inference > ' + inference
                # for i in range(FLAGS.beam_width):
                #     print 'beam > ' + beam_out[i]

    def infer_func(self, model, sess, data):
        d = []
        for item in data:
            d.append({'post': item['post'], 'response': item['response'], 'ref': item['context_response']})
        batch = self.gen_batched_data(d)
        input_feed = {
            'input/post_string:0': batch['post'],
            'input/post_len:0': batch['post_len'],
            'input/response_string:0': batch['response'],
            'input/response_len:0': batch['response_len'],
            'input/ref_string:0': batch['ref'],
            'input/ref_len:0': batch['ref_len'],
            'input/use_prior:0': True
        }
        res = sess.run(['decode_1/inference:0'], input_feed)
        return res[0]


def main(unused_argv):
    dp = data_process(FLAGS)
    print(FLAGS.__flags)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if FLAGS.is_train:
            train_data = dp.load_train_data()
            valid_data = dp.load_valid_data()
            test_data = dp.load_test_data()
            vocab, embed = dp.build_vocab(train_data)
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
            summary_list = ['ppl', 'elbo', 'sen_loss', 'kl_loss', 'avg_kld', 'kl_weights']
            summary_num = len(summary_list)
            summary_placeholders = [tf.placeholder(tf.float32) for i in range(summary_num)]
            summary_op = [tf.summary.scalar(summary_list[i], summary_placeholders[i]) for i in range(summary_num)]

            summary_sum = np.zeros((summary_num,))
            # train_loss = 0
            time_step = 0
            previous_losses = [1e18] * 3
            while True:
                start_time = time.time()

                train_batch = dp.train_batch(train_data, FLAGS.batch_size)
                ops = s2s.step(sess, train_batch, is_train=True)
                for i in range(summary_num):
                    summary_sum[i] += ops[i+1]

                global_step = s2s.global_step.eval()
                # train_loss += loss
                time_step += (time.time() - start_time)

                if global_step % FLAGS.save_every_n_iteration == 0:
                    time_step /= FLAGS.save_every_n_iteration
                    summary_sum /= FLAGS.save_every_n_iteration

                    if FLAGS.opt == 'SGD' and summary_sum[1] > max(previous_losses):
                        sess.run(s2s.learning_rate_decay_op)
                    previous_losses = previous_losses[1:] + [summary_sum[1]]

                    # for ppl
                    summary_sum[0] = np.exp(summary_sum[0])
                    feed_dict = dict(zip(summary_placeholders,summary_sum))

                    summaries = sess.run(summary_op, feed_dict=feed_dict)
                    for s in summaries:
                        train_writer.add_summary(summary=s, global_step=global_step)
                    print("global step %d step-time %.4f learning_rate %f"
                          % (global_step,
                             time_step,
                             s2s.learning_rate.eval() if FLAGS.opt=='SGD' else .0))
                    for i in range(summary_num):
                        print 'train '+summary_list[i]+': %f' % summary_sum[i]

                    summary_sum = np.zeros((summary_num,))

                    for batch in dp.eval_batches(valid_data,FLAGS.batch_size):
                        ops = s2s.step(sess, batch, is_train=False)
                        for i in range(summary_num):
                            summary_sum[i] += ops[i]

                    summary_sum /= FLAGS.valid_size // FLAGS.batch_size

                    summary_sum[0] = np.exp(summary_sum[0])
                    feed_dict = dict(zip(summary_placeholders, summary_sum))

                    summaries = sess.run(summary_op, feed_dict=feed_dict)
                    for s in summaries:
                        valid_writer.add_summary(summary=s, global_step=global_step)

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
            print 'read paras from %s' % model_path
            saver = tf.train.import_meta_graph(model_path + '.meta')
            saver.restore(sess, model_path)
            dp.infer(sess)


if __name__ == '__main__':
    tf.app.run()
