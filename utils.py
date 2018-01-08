import tensorflow as tf
import numpy as np
from itertools import izip
import random
import os

_PAD = b"PAD"
_GO = b"GO"
_EOS = b"EOS"
_UNK = b"UNK"
_START_VOCAB = [_PAD, _EOS, _GO, _UNK]

PAD_ID = 0
EOS_ID = 1
GO_ID = 2
UNK_ID = 3


def gaussian_kld(mu_1, logvar_1, mu_2, logvar_2):
    kld = -0.5 * tf.reduce_sum(1 + (logvar_1 - logvar_2)
                               - tf.div(tf.pow(mu_2 - mu_1, 2), tf.exp(logvar_2))
                               - tf.div(tf.exp(logvar_1), tf.exp(logvar_2)), reduction_indices=1)
    return kld


def sample_gaussian(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    z = mu + tf.multiply(std, epsilon)
    return z


def build_vocab(word_vector_path, embed_size, vocab_size, data):
    print("Creating vocabulary...")
    vocab = {}
    for i, pair in enumerate(data):
        if i % 100000 == 0:
            print("    processing line %d" % i)
        for token in pair['post'] + pair['response']:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > vocab_size:
        vocab_list = vocab_list[:vocab_size]

    if not os.path.exists(word_vector_path):
        print("Cannot find word vectors")
        embed = None
        return vocab_list, embed

    print("Loading word vectors...")
    vectors = {}
    with open(word_vector_path) as f:
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
            vector = np.zeros(embed_size, dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)
    print ("%d word vectors pre-trained" % pre_vector)

    return vocab_list, embed


def train_batch(data, batch_size):
    selected_data = [random.choice(data) for _ in range(batch_size)]
    batched_data = gen_batched_data(selected_data)
    return batched_data


def eval_batches(data, batch_size):
    for i in range(len(data)//batch_size):
        batched_data = gen_batched_data(data[i*batch_size:(i+1)*batch_size])
        yield batched_data


def infer(sess, infer_path_post, infer_path_resp, batch_size=128):
    if infer_path_post is not "":
        f1 = open(infer_path_post)
        post = [line.strip().split() for line in f1]
        f1.close()
        f2 = open(infer_path_resp, 'wb')

        data = []
        for p in post:
            data.append({'post': p, 'response': []})

        id = 0
        for j in range((len(data)+batch_size-1) // batch_size):
            batch = gen_batched_data(data[j * batch_size:(j + 1) * batch_size])
            input_feed = {
                'input/post_string:0': batch['post'],
                'input/post_len:0': batch['post_len'],
                'input/response_string:0': batch['response'],
                'input/response_len:0': batch['response_len'],
                'input/use_encoder:0': True,
            }
            res = sess.run(['decode_1/inference:0', 'decode_2/beam_out:0'], input_feed)
            print id
            for i in range(len(batch['post_len'])):
                print >> f2, 'post: ' + ' '.join(post[id])
                print >> f2, 'infer: ' + cut_eos(' '.join(res[0][i]))
                print >> f2, 'beam: ' + cut_eos(' '.join(res[1][i, :, 0]))
                print >> f2, ''

                id += 1

        f2.close()
    else:
        while True:
            infer_data = {}
            infer_data['post'] = raw_input('post > ').strip().split()
            infer_data['response'] = '233'.strip().split()
            infer_data = [infer_data]
            batch = gen_batched_data(infer_data)

            z1 = get_enc_z(sess, batch)
            decode_from_z(sess, batch, z1)

            infer_data = {}
            infer_data['post'] = raw_input('post > ').strip().split()
            infer_data['response'] = '233'.strip().split()
            infer_data = [infer_data]
            batch = gen_batched_data(infer_data)

            z2 = get_enc_z(sess, batch)
            decode_from_z(sess, batch, z2)

            interpolate(sess, z1, z2, 10, batch)


def gen_batched_data(data):
    encoder_len = max([len(item['post']) for item in data]) + 1
    decoder_len = max([len(item['response']) for item in data]) + 1

    posts, responses, posts_length, responses_length = [], [], [], []

    def padding(sent, l):
        return sent + [_EOS] + [_PAD] * (l - len(sent) - 1)

    for item in data:
        posts.append(padding(item['post'], encoder_len))
        responses.append(padding(item['response'], decoder_len))
        posts_length.append(len(item['post']) + 1)
        responses_length.append(len(item['response']) + 1)

    batched_data = {'post': np.array(posts),
                    'response': np.array(responses),
                    'post_len': posts_length,
                    'response_len': responses_length}
    return batched_data


def cut_eos(sentence):
    if sentence.find('EOS') != -1:
        return sentence[:sentence.find('EOS')]
    return sentence


def get_enc_z(sess, data):
    input_feed = {
        'input/post_string:0': data['post'],
        'input/post_len:0': data['post_len'],
        'input/response_string:0': data['response'],
        'input/response_len:0': data['response_len'],
        'input/use_encoder:0': True,
    }
    output_feed = ['hidden/enc_z/MatMul:0']
    return sess.run(output_feed, input_feed)[0]


def decode_from_z(sess, data, z):
    input_feed = {
        'input/post_string:0': data['post'],
        'input/post_len:0': data['post_len'],
        'input/response_string:0': data['response'],
        'input/response_len:0': data['response_len'],
        'input/use_encoder:0': False,
        'input/input_z:0': z
    }
    output_feed = ['decode_1/inference:0', 'decode_2/beam_out:0']
    res = sess.run(output_feed, input_feed)
    inference = cut_eos(' '.join(res[0][0]))
    # beam_out = [cut_eos(' '.join(res[1][0, :, i])) for i in range(FLAGS.beam_width)]
    print 'inference > ' + inference
    # for i in range(FLAGS.beam_width):
    #     print 'beam > ' + beam_out[i]


def interpolate(sess, z1, z2, n_sample, data):
    pts = []
    for s, e in zip(z1[0].tolist(), z2[0].tolist()):
        pts.append(np.linspace(s, e, n_sample))
    pts = np.array(pts)
    pts = pts.T
    for pt in pts:
        decode_from_z(sess, data, pt.reshape(1, 128))


class data_process(object):
    def __init__(self,
                 tfFLAGS):
        self.data_dir = tfFLAGS.data_dir
        self.train_from = os.path.join(self.data_dir, 'train.weibo_pair.post')
        self.train_to = os.path.join(self.data_dir, 'train.weibo_pair.response')
        self.valid_from = os.path.join(self.data_dir, 'valid.weibo_pair.post')
        self.valid_to = os.path.join(self.data_dir, 'valid.weibo_pair.response')
        self.test_from = os.path.join(self.data_dir, 'test.weibo_pair.post')
        self.test_to = os.path.join(self.data_dir, 'test.weibo_pair.response')
        if not tfFLAGS.split:
            self.data_from = os.path.join(self.data_dir, tfFLAGS.data_from)
            self.data_to = os.path.join(self.data_dir, tfFLAGS.data_to)
            self._split(tfFLAGS.train_size, tfFLAGS.valid_size, tfFLAGS.test_size)
        self.vocab_size = tfFLAGS.vocab_size

    def _split(self, train_size, valid_size, test_size):
        total_size = train_size+valid_size+test_size
        sel = random.sample(range(total_size), total_size)
        valid_dict = {}.fromkeys(sel[:valid_size])
        test_dict = {}.fromkeys(sel[-test_size:])
        train_from = open(self.train_from,'wb')
        train_to = open(self.train_to,'wb')
        valid_from = open(self.valid_from,'wb')
        valid_to = open(self.valid_to,'wb')
        test_from = open(self.test_from,'wb')
        test_to = open(self.test_to,'wb')

        with open(self.data_from) as ff, open(self.data_to) as ft:
            cntline = 0
            for post, resp in izip(ff,ft):
                if cntline in valid_dict:
                    valid_from.write(post)
                    valid_to.write(resp)
                elif cntline in test_dict:
                    test_from.write(post)
                    test_to.write(resp)
                else:
                    train_from.write(post)
                    train_to.write(resp)
                cntline += 1
                if cntline == total_size:
                    break

        train_from.close()
        train_to.close()
        valid_from.close()
        valid_to.close()
        test_from.close()
        test_to.close()
        print "split completed"

    def load_train_data(self):
        return self.load_data(self.train_from,self.train_to)

    def load_valid_data(self):
        return self.load_data(self.valid_from, self.valid_to)

    def load_test_data(self):
        return self.load_data(self.test_from, self.test_to)

    def load_data(self,
                  post_f,
                  resp_f):
        f1 = open(post_f)
        f2 = open(resp_f)
        post = [line.strip().split() for line in f1.readlines()]
        response = [line.strip().split() for line in f2.readlines()]
        data = []
        for p, r in zip(post, response):
            data.append({'post': p, 'response': r})
        f1.close()
        f2.close()
        return data


