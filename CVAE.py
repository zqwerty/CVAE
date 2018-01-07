import tensorflow as tf
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.python.layers.core import Dense
from tensorflow.python.framework import constant_op
from utils import UNK_ID, _UNK, GO_ID, EOS_ID, gaussian_kld, sample_gaussian


class CVAE(object):
    def __init__(self,
                 tfFLAGS,
                 embed=None):
        self.vocab_size = tfFLAGS.vocab_size
        self.embed_size = tfFLAGS.embed_size
        self.num_units = tfFLAGS.num_units
        self.num_layers = tfFLAGS.num_layers
        self.beam_width = tfFLAGS.beam_width
        self.use_lstm = tfFLAGS.use_lstm
        self.attn_mode = tfFLAGS.attn_mode
        self.train_keep_prob = tfFLAGS.keep_prob
        self.max_decode_len = tfFLAGS.max_decode_len
        self.bi_encode = tfFLAGS.bi_encode
        # self.recog_hidden_units = tfFLAGS.recog_hidden_units
        # self.prior_hidden_units = tfFLAGS.prior_hidden_units
        self.z_dim = tfFLAGS.z_dim
        self.full_kl_step = tfFLAGS.full_kl_step
        self.min_kl = tfFLAGS.min_kl
        self.l2_loss_weight = tfFLAGS.l2_loss_weight

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.max_gradient_norm = 5.0
        if tfFLAGS.opt == 'SGD':
            self.learning_rate = tf.Variable(float(tfFLAGS.learning_rate),
                                             trainable=False, dtype=tf.float32)
            self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * tfFLAGS.learning_rate_decay_factor)
            self.opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif tfFLAGS.opt == 'Momentum':
            self.opt = tf.train.MomentumOptimizer(learning_rate=tfFLAGS.learning_rate, momentum=tfFLAGS.momentum)
        else:
            self.opt = tf.train.AdamOptimizer(learning_rate=tfFLAGS.learning_rate)

        self._make_input(embed)

        with tf.variable_scope("output_layer"):
            self.output_layer = Dense(self.vocab_size,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      use_bias=False)

        with tf.variable_scope("encoders", initializer=tf.orthogonal_initializer()):
            # self.enc_post_outputs, self.enc_post_state = self._build_encoder(scope='post_encoder',
            #                                                                  inputs=self.enc_post,
            #                                                                  sequence_length=self.post_len)
            # self.enc_ref_outputs, self.enc_ref_state = self._build_encoder(scope='ref_encoder', inputs=self.enc_ref,
            #                                                                sequence_length=self.ref_len)
            self.enc_response_outputs, self.enc_response_state = self._build_encoder(scope='resp_encoder',
                                                                                     inputs=self.enc_response,
                                                                                     sequence_length=self.response_len)

            # self.post_state = self._get_representation_from_enc_state(self.enc_post_state)
            # self.ref_state = self._get_representation_from_enc_state(self.enc_ref_state)
            self.response_state = self._get_representation_from_enc_state(self.enc_response_state)
            # self.cond_embed = tf.concat([self.post_state, self.ref_state], axis=-1)

        # with tf.variable_scope("hidden"):
        #     self.enc_z = tf.layers.dense(inputs=self.post_state, units=self.z_dim, activation=None, use_bias=False,
        #                                  name='enc_z')

        with tf.variable_scope("RecognitionNetwork"):
            recog_input = self.response_state
            recog_mulogvar = tf.layers.dense(inputs=recog_input, units=self.z_dim * 2, activation=None)
            recog_mu, recog_logvar = tf.split(recog_mulogvar, 2, axis=-1)
            self.mu = tf.identity(recog_mu, name='mu')
            self.recog_z = tf.identity(sample_gaussian(recog_mu, recog_logvar), name='recog_z')

            # recog_input = tf.concat([self.cond_embed, self.response_state], axis=-1)
            # recog_hidden = tf.layers.dense(inputs=recog_input, units=self.recog_hidden_units, activation=tf.nn.tanh)
            # recog_mulogvar = tf.layers.dense(inputs=recog_hidden, units=self.z_dim*2, activation=None)
            # # recog_mulogvar = tf.layers.dense(inputs=recog_input, units=self.z_dim * 2, activation=None)
            # recog_mu, recog_logvar = tf.split(recog_mulogvar, 2, axis=-1)

        with tf.variable_scope("PriorNetwork"):
            prior_mu, prior_logvar = tf.zeros_like(recog_mu), tf.ones_like(recog_logvar)
            self.prior_z = tf.identity(sample_gaussian(prior_mu, prior_logvar), name='prior_z')
        #     prior_input = self.cond_embed
        #     prior_hidden = tf.layers.dense(inputs=prior_input, units=self.prior_hidden_units, activation=tf.nn.tanh)
        #     prior_mulogvar = tf.layers.dense(inputs=prior_hidden, units=self.z_dim*2, activation=None)
        #     prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=-1)

        with tf.variable_scope("GenerationNetwork"):
            latent_sample = tf.cond(self.use_sample,
                                    lambda: tf.cond(self.use_encoder,
                                                    lambda: self.recog_z,
                                                    lambda: self.prior_z),
                                    lambda: tf.cond(self.use_encoder,
                                                    lambda: self.mu,
                                                    lambda: self.input_z),
                                    name='latent_sample')
            # latent_sample = tf.cond(self.use_encoder,
            #                         lambda: self.enc_z,
            #                         lambda: self.input_z),
            #                         name='latent_sample')

            # gen_input = tf.concat([self.cond_embed, latent_sample], axis=-1)
            if self.use_lstm:
                self.dec_init_state = tuple(
                    [tf.contrib.rnn.LSTMStateTuple(
                        # c=tf.layers.dense(inputs=gen_input, units=self.num_units, activation=None),
                        # h=tf.layers.dense(inputs=gen_input, units=self.num_units, activation=None))
                        c=tf.layers.dense(inputs=latent_sample, units=self.num_units, activation=None, use_bias=False),
                        h=tf.layers.dense(inputs=latent_sample, units=self.num_units, activation=None, use_bias=False))
                        for _ in
                        range(self.num_layers)])
                print self.dec_init_state
            else:
                self.dec_init_state = tuple(
                    # [tf.layers.dense(inputs=gen_input, units=self.num_units, activation=None) for _ in
                    [tf.layers.dense(inputs=latent_sample, units=self.num_units, activation=None, use_bias=False)
                     for _ in range(self.num_layers)])

            kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
            self.avg_kld = tf.reduce_mean(kld)
            self.kl_weights = tf.minimum(tf.to_float(self.global_step) / self.full_kl_step, 1.0)
            self.kl_loss = self.kl_weights * tf.maximum(self.avg_kld, self.min_kl)

        self._build_decoder()

        # Calculate and clip gradients
        params = tf.trainable_variables()
        self.l2_loss = self.l2_loss_weight * tf.reduce_sum([tf.nn.l2_loss(v) for v in params])
        self.elbo = self.sen_loss + self.kl_loss
        self.loss = self.elbo + self.l2_loss
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.train_op = self.opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                                    max_to_keep=1, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        for var in tf.trainable_variables():
            print var
        tf.summary.histogram('recog_mu', recog_mu)
        tf.summary.histogram('recog_logvar', recog_logvar)
        tf.summary.histogram('recog_z', self.recog_z)
        tf.summary.histogram('prior_z', self.prior_z)
        self.merge_summary_op = tf.summary.merge_all()

    def _make_input(self, embed):
        self.symbol2index = MutableHashTable(
            key_dtype=tf.string,
            value_dtype=tf.int64,
            default_value=UNK_ID,
            shared_name="in_table",
            name="in_table",
            checkpoint=True)
        self.index2symbol = MutableHashTable(
            key_dtype=tf.int64,
            value_dtype=tf.string,
            default_value=_UNK,
            shared_name="out_table",
            name="out_table",
            checkpoint=True)
        with tf.variable_scope("input"):
            self.post_string = tf.placeholder(tf.string, (None,None), 'post_string')
            # self.ref_string = tf.placeholder(tf.string, (None, None), 'ref_string')
            self.response_string = tf.placeholder(tf.string, (None, None), 'response_string')

            self.post = self.symbol2index.lookup(self.post_string)
            self.post_len = tf.placeholder(tf.int32, (None,), 'post_len')
            # self.ref = self.symbol2index.lookup(self.ref_string)
            # self.ref_len = tf.placeholder(tf.int32, (None,), 'ref_len')
            self.response = self.symbol2index.lookup(self.response_string)
            self.response_len = tf.placeholder(tf.int32, (None,), 'response_len')

            with tf.variable_scope("embedding") as scope:
                if embed is None:
                    # initialize the embedding randomly
                    self.emb_enc = self.emb_dec = tf.get_variable(
                        "emb_share", [self.vocab_size, self.embed_size], dtype=tf.float32
                    )
                else:
                    # initialize the embedding by pre-trained word vectors
                    print "share pre-trained embed"
                    self.emb_enc = self.emb_dec = tf.get_variable('emb_share', dtype=tf.float32, initializer=embed)

            self.enc_post = tf.nn.embedding_lookup(self.emb_enc, self.post)
            # self.enc_ref = tf.nn.embedding_lookup(self.emb_enc, self.ref)
            self.enc_response = tf.nn.embedding_lookup(self.emb_enc, self.response)

            self.batch_len = tf.shape(self.response)[1]
            self.batch_size = tf.shape(self.response)[0]
            self.response_input = tf.concat([tf.ones((self.batch_size, 1), dtype=tf.int64) * GO_ID,
                                             tf.split(self.response, [self.batch_len - 1, 1], axis=1)[0]], 1)
            self.dec_inp = tf.nn.embedding_lookup(self.emb_dec, self.response_input)

            self.keep_prob = tf.placeholder_with_default(1.0, ())
            self.use_encoder = tf.placeholder(dtype=tf.bool, name="use_encoder")
            self.input_z = tf.placeholder_with_default(tf.zeros((1, self.z_dim), dtype=tf.float32),
                                                       (1, self.z_dim),
                                                       name="input_z")
            self.use_sample = tf.placeholder(dtype=tf.bool, name="use_sample")
            # self.use_prior = tf.placeholder(dtype=tf.bool, name="use_prior")

    def _build_encoder(self, scope, inputs, sequence_length):
        with tf.variable_scope(scope):
            if self.bi_encode:
                cell_fw, cell_bw = self._build_biencoder_cell()
                outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=inputs,
                    sequence_length=sequence_length,
                    dtype=tf.float32
                )
                enc_outputs = tf.concat(outputs, axis=-1)
                enc_state = []
                for i in range(self.num_layers):
                    if self.use_lstm:
                        encoder_state_c = tf.concat([states[0][i].c,states[1][i].c], axis=-1)
                        encoder_state_h = tf.concat([states[0][i].h,states[1][i].h], axis=-1)
                        enc_state.append(tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h))
                    else:
                        enc_state.append(tf.concat([states[0][i],states[1][i]], axis=-1))
                enc_state = tuple(enc_state)
                return enc_outputs, enc_state
            else:
                enc_cell = self._build_encoder_cell()
                enc_outputs, enc_state = tf.nn.dynamic_rnn(
                    cell=enc_cell,
                    inputs=inputs,
                    sequence_length=sequence_length,
                    dtype=tf.float32
                )
                return enc_outputs, enc_state

    def _get_representation_from_enc_state(self, enc_state):
        if self.use_lstm:
            return tf.concat([state.h for state in enc_state], axis=-1)
        else:
            return tf.concat(enc_state, axis=-1)

    def _build_decoder(self):
        with tf.variable_scope("decode", initializer=tf.orthogonal_initializer()):
            dec_cell, init_state = self._build_decoder_cell(self.enc_post_outputs, self.post_len, self.dec_init_state)

            train_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=self.dec_inp,
                sequence_length=self.response_len
            )
            train_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell,
                helper=train_helper,
                initial_state=init_state,
                output_layer=self.output_layer
            )
            train_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=train_decoder,
                maximum_iterations=self.max_decode_len,
            )
            logits = train_output.rnn_output

            mask = tf.sequence_mask(self.response_len, self.batch_len, dtype=tf.float32)

            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.response, logits=logits)
            crossent = tf.reduce_sum(crossent * mask)
            self.sen_loss = crossent / tf.to_float(self.batch_size)

            # ppl(loss avg) across each timestep, the same as :
            # self.loss = tf.contrib.seq2seq.sequence_loss(train_output.rnn_output,
            #                                              self.response,
            #                                              mask)
            self.ppl_loss = crossent / tf.reduce_sum(mask)

            # add kld:
            # self.elbo = self.sen_loss + self.kl_loss

            # # Calculate and clip gradients
            # params = tf.trainable_variables()
            # gradients = tf.gradients(self.elbo, params)
            # clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            # self.train_op = self.opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
            #
            self.train_out = self.index2symbol.lookup(tf.cast(train_output.sample_id, tf.int64), name='train_out')

        with tf.variable_scope("decode", reuse=True):
            dec_cell, init_state = self._build_decoder_cell(self.enc_post_outputs, self.post_len, self.dec_init_state)

            start_tokens = tf.tile(tf.constant([GO_ID], dtype=tf.int32), [self.batch_size])
            end_token = EOS_ID
            infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.emb_dec,
                start_tokens,
                end_token
            )
            infer_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell,
                helper=infer_helper,
                initial_state=init_state,
                output_layer=self.output_layer
            )
            infer_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=infer_decoder,
                maximum_iterations=self.max_decode_len,
            )

            self.inference = self.index2symbol.lookup(tf.cast(infer_output.sample_id, tf.int64), name='inference')

        with tf.variable_scope("decode", reuse=True):
            dec_init_state = tf.contrib.seq2seq.tile_batch(self.dec_init_state, self.beam_width)
            enc_outputs = tf.contrib.seq2seq.tile_batch(self.enc_post_outputs, self.beam_width)
            post_len = tf.contrib.seq2seq.tile_batch(self.post_len, self.beam_width)

            dec_cell, init_state = self._build_decoder_cell(enc_outputs, post_len, dec_init_state,
                                                            beam_width=self.beam_width)

            beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=dec_cell,
                embedding=self.emb_dec,
                start_tokens=tf.ones_like(self.post_len) * GO_ID,
                end_token=EOS_ID,
                initial_state=init_state,
                beam_width=self.beam_width,
                output_layer=self.output_layer
            )
            beam_output, _, beam_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=beam_decoder,
                maximum_iterations=self.max_decode_len,
            )

            self.beam_out = self.index2symbol.lookup(tf.cast(beam_output.predicted_ids, tf.int64), name='beam_out')

    def _build_encoder_cell(self):
        if self.use_lstm:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_units), self.keep_prob) for _ in range(self.num_layers)])
        else:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.num_units), self.keep_prob) for _ in range(self.num_layers)])
        return cell

    def _build_biencoder_cell(self):
        if self.use_lstm:
            cell_fw = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_units / 2), self.keep_prob) for _ in range(self.num_layers)])
            cell_bw = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_units / 2), self.keep_prob) for _ in range(self.num_layers)])
        else:
            cell_fw = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.num_units / 2), self.keep_prob) for _ in range(self.num_layers)])
            cell_bw = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.num_units / 2), self.keep_prob) for _ in range(self.num_layers)])
        return cell_fw, cell_bw

    def _build_decoder_cell(self, memory, memory_len, encode_state, beam_width=1):
        if self.use_lstm:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(self.num_units), self.keep_prob) for _ in range(self.num_layers)])
        else:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.num_units), self.keep_prob) for _ in range(self.num_layers)])
        if self.attn_mode=='Luong':
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.num_units,
                memory=memory,
                memory_sequence_length=memory_len,
                scale=True
            )
        elif self.attn_mode=='Bahdanau':
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.num_units,
                memory=memory,
                memory_sequence_length=memory_len,
                scale=True
            )
        else:
            return cell, encode_state
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=self.num_units,
        )
        return attn_cell, attn_cell.zero_state(self.batch_size * beam_width, tf.float32).clone(
            cell_state=encode_state)

    def initialize(self, sess, vocab):
        op_in = self.symbol2index.insert(constant_op.constant(vocab),
                                         constant_op.constant(range(len(vocab)), dtype=tf.int64))
        op_out = self.index2symbol.insert(constant_op.constant(range(len(vocab)), dtype=tf.int64),
                                          constant_op.constant(vocab))
        sess.run(tf.global_variables_initializer())
        sess.run([op_in, op_out])

    def step(self, sess, data, is_train=False):
        input_feed = {
            self.post_string: data['post'],
            self.post_len: data['post_len'],
            # self.ref_string: data['ref'],
            # self.ref_len: data['ref_len'],
            self.response_string: data['response'],
            self.response_len: data['response_len'],
            self.use_encoder: True,
            self.use_sample: True
            # self.use_prior: is_train,
        }
        if is_train:
            output_feed = [self.train_op,
                           self.ppl_loss,
                           self.elbo,
                           self.sen_loss,
                           self.kl_loss,
                           self.avg_kld,
                           self.kl_weights,
                           self.l2_loss,
                           self.loss,
                           self.merge_summary_op
                           # self.kl_loss,
                           # self.avg_kld,
                           # self.kl_weights,
                           # self.post_string,
                           # self.response_string,
                           # self.train_out,
                           # self.inference,
                           # self.beam_out,
                           ]
            input_feed[self.keep_prob] = self.train_keep_prob
        else:
            output_feed = [self.ppl_loss,
                           self.elbo,
                           self.sen_loss,
                           self.kl_loss,
                           self.avg_kld,
                           self.kl_weights,
                           self.l2_loss,
                           self.loss,
                           self.merge_summary_op
                           # self.kl_loss,
                           # self.avg_kld,
                           # self.kl_weights,
                           # self.post_string,
                           # self.response_string,
                           # self.train_out,
                           # self.inference,
                           # self.beam_out,
                           ]
        return sess.run(output_feed, input_feed)
