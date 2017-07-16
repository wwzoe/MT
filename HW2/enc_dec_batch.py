# coding: utf-8

# In[ ]:



# In[ ]:

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.functions.array import concat

# In[ ]:

from nmt_config import *


# In[ ]:

# In[ ]:

class EncoderDecoder(Chain):

    def __init__(self, vsize_enc, vsize_dec,
                 nlayers_enc, nlayers_dec,
                 n_units, gpuid, attn=False):
        '''
        vsize:   vocabulary size
        nlayers: # layers
        attn:    if True, use attention
        '''
        super(EncoderDecoder, self).__init__()
        #--------------------------------------------------------------------
        # add encoder layers
        #--------------------------------------------------------------------

        # add embedding layer
        self.add_link("embed_enc", L.EmbedID(vsize_enc, n_units))

        #add CNN layer
        self.cnn_enc=["L{0:d}_cnn".format(i) for i in range(1,9)]
        for i,cnn_name in enumerate(self.cnn_enc):
            self.add_link(cnn_name,L.Convolution2D(1,1,(i+1,200),stride=(1,200)))
        # self.add_link("cnn_enc",L.Convolution2D(1，1，(64,i)))
        #add highway layer
        self.hw_enc=["L{0:d}_hw".format(i) for i in range(1,5)]
        for i,hw_name in enumerate(self.hw_enc):
            self.add_link(hw_name,L.Highway(4))

        # add LSTM layers
        self.lstm_enc = ["L{0:d}_enc".format(i) for i in range(nlayers_enc)]
        for lstm_name in self.lstm_enc:
            self.add_link(lstm_name, L.LSTM(n_units, n_units))

        # reverse LSTM layer
        self.lstm_rev_enc = ["L{0:d}_rev_enc".format(i) for i in range(nlayers_enc)]
        for lstm_name in self.lstm_rev_enc:
            self.add_link(lstm_name, L.LSTM(n_units, n_units))

        #--------------------------------------------------------------------
        # add decoder layers
        #--------------------------------------------------------------------

        # add embedding layer
        self.add_link("embed_dec", L.EmbedID(vsize_dec, 2*n_units))

        # add LSTM layers
        self.lstm_dec = ["L{0:d}_dec".format(i) for i in range(nlayers_dec)]
        for lstm_name in self.lstm_dec:
            self.add_link(lstm_name, L.LSTM(2*n_units, 2*n_units))

        if attn > 0:
            # add context layer for attention
            self.add_link("context", L.Linear(4*n_units, 2*n_units))
        self.attn = attn

        # add output layer
        self.add_link("out", L.Linear(2*n_units, vsize_dec))

        # Store GPU id
        self.gpuid = gpuid
        self.n_units = n_units

        xp = cuda.cupy if self.gpuid >= 0 else np

        # create masking array for pad id
        self.mask_pad_id = xp.ones(vsize_dec, dtype=xp.float32)
        # make the class weight for pad id equal to 0
        # this way loss will not be computed for this predicted loss
        self.mask_pad_id[0] = 0

    def reset_state(self):
        # reset the state of LSTM layers
        for lstm_name in self.lstm_enc + self.lstm_rev_enc + self.lstm_dec:
            self[lstm_name].reset_state()
        self.loss = 0

    def set_decoder_state(self):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # set the hidden and cell state of the first LSTM in the decoder
        # concatenate cell state of both enc LSTMs
        c_state = F.concat((self[self.lstm_enc[-1]].c, self[self.lstm_rev_enc[-1]].c))
        # concatenate hidden state of both enc LSTMs
        h_state = F.concat((self[self.lstm_enc[-1]].h, self[self.lstm_rev_enc[-1]].h))
        # h_state = F.split((self.enc_states), [:len(self.enc_states.data)])[0]
        self[self.lstm_dec[0]].set_state(c_state, h_state)

    '''
    Function to feed an input word through the embedding and lstm layers
        args:
        embed_layer: embeddings layer to use
        lstm_layer:  list of lstm layer names
    '''
    def conv_layer(self,word,embed_layer,cnn_layer,train):
        
        embed_id = embed_layer(word)
        dim0=word.data.shape[0]
        dim2=word.data.shape[1]
        print(embed_id.data.shape)
        first_cnn=True
        for i,cnn_layer in enumerate(self.cnn_enc):
            padding=np.zeros((dim0,i,200),dtype=np.float32)
            cnn_in=F.concat((embed_id,padding),axis=1)
            print（embed_id）
            cnn_in=F.reshape(cnn_in,(dim0,1,dim2+i,200))
            #print(cnn_in.data.shape)
            conv_out=self[cnn_layer](cnn_in)


            #conv_out=F.reshape(conv_out,(dim0,dim2,conv_out.data.shape[3]))
            if first_cnn:
              conv_out_all=conv_out
              first_cnn=False
            else:
              conv_out_all=F.concat((conv_out_all, conv_out),axis=3)

        conv_out_all=F.swapaxes(conv_out_all, 2, 3)
        #print(conv_out_all.data.shape)    
        #conv_out_all=F.reshape(conv_out_all,(dim0,1,dim2,1600))
        #print(conv_out_all.data.shape)
        pooling_out= F.max_pooling_2d(conv_out_all, (1,2),(1,2))
        #print(pooling_out.data.shape)
        
        return pooling_out

    def feed_lstm_encode(self, word, embed_layer, lstm_layer_list, train):
        if train:
        #embed_id = embed_layer(word)
            hs = self[lstm_layer_list[0]](word)
            # feed into remaining LSTM layers
            for lstm_layer in lstm_layer_list[1:]:
                hs = F.dropout(self[lstm_layer](hs), ratio=0.2, train=train)
        else:
            embed_id = embed_layer(word)
            hs = self[lstm_layer_list[0]](embed_id)
            # feed into remaining LSTM layers
            for lstm_layer in lstm_layer_list[1:]:
                hs = F.dropout(self[lstm_layer](hs), ratio=0.2, train=train)

    def feed_lstm_decode(self, word, embed_layer, lstm_layer_list, train):
        # get embedding
        #print(word.data.shape)
        embed_id = embed_layer(word)

        # feed into first LSTM layer
        hs = self[lstm_layer_list[0]](embed_id)
        # feed into remaining LSTM layers
        for lstm_layer in lstm_layer_list[1:]:
            hs = F.dropout(self[lstm_layer](hs), ratio=0.2, train=train)


    def encode(self, word, lstm_layer_list, train):
        self.feed_lstm_encode(word, self.embed_enc, lstm_layer_list, train)

    def decode(self, word, train):
        self.feed_lstm_decode(word, self.embed_dec, self.lstm_dec, train)



    #--------------------------------------------------------------------
    # For batch size > 1
    #--------------------------------------------------------------------
    def pad_list(self, data, lim, at_start=True, type=1):
        xp = cuda.cupy if self.gpuid >= 0 else np
        if at_start:
            ret_data = [PAD_ID]*(lim - len(data)) + data
        else:
            ret_data = data + [PAD_ID]*(lim - len(data))
        if type==1:
            out=xp.asarray(ret_data, dtype=xp.int32)
        else:
            out=xp.asarray(ret_data, dtype=xp.float32)
        return out

  #--------------------------------------------------------------------
    # For batch size > 1
    #--------------------------------------------------------------------
    def encode_batch(self, cnn_output, train=True):
        #xp = cuda.cupy if self.gpuid >= 0 else np
        # convert list of tokens into chainer variable list
        padding=np.zeros((cnn_output.data.shape[0],200-cnn_output.data.shape[1],cnn_output.data.shape[2]),dtype=np.float32)
        padding_array=F.concat((cnn_output.data,padding),axis=1)

        #var_rev_en = (Variable(rev_encoder_batch.T, volatile=(not train)))
        a=padding_array.data
        b=padding_array.data[::-1]
        #print(a[:,:,0:1]-b[:,:,5:6])
        #print(padding_array[])
        #print(cnn_output.data.shape)
        masks=np.min(cnn_output.data,axis=1)
        #print(masks)

        first_entry = True

        #seq_len, batch_size = cnn_output.shape
        seq_len= cnn_output.shape[2]
        batch_size=cnn_output.shape[0]

        if self.attn:
            self.mask = self.xp.expand_dims(masks != 0, -1)
            self.minf = Variable(self.xp.full((batch_size, seq_len, 1), -1000.,
                                 dtype=self.xp.float32), volatile=not train)

        # for all sequences in the batch, feed the characters one by one
        #print(seq_len)  #seq_len=bucket_width
        #for i in range(seq_len):
            # encode tokens
         #   w = var_en[i]
            #print(w.data.shape)
        #    rev_w = var_rev_en[i]

            #print(w.data.shape)
        for i in range(0,cnn_output.data.shape[2]):
            w=a[0:batch_size,0:200,i:i+1]
            w=w.reshape(batch_size,200)
            #print(w.shape)
            rev_w=b[0:batch_size,0:200,i:i+1]
            rev_w=rev_w.reshape(batch_size,200)
            #print(w)
            self.encode(w, self.lstm_enc, train)
            self.encode(rev_w, self.lstm_rev_enc, train)

            if first_entry == False:

                self.forward_states = F.concat((self.forward_states,
                                                F.reshape(self[self.lstm_enc[-1]].h,
                                                shape=(batch_size, 1, self.n_units))), axis=1)

                self.backward_states = F.concat((F.reshape(self[self.lstm_rev_enc[-1]].h,
                                                shape=(batch_size, 1, self.n_units)),
                                                self.backward_states), axis=1)
            else:
                self.forward_states = F.reshape(self[self.lstm_enc[-1]].h,
                                                shape=(batch_size, 1, self.n_units))

                self.backward_states = F.reshape(self[self.lstm_rev_enc[-1]].h,
                                                shape=(batch_size, 1, self.n_units))


                first_entry = False

        self.enc_states = F.concat((self.forward_states, self.backward_states), axis=2)


    #--------------------------------------------------------------------
    # For batch size > 1
    #--------------------------------------------------------------------
    def decode_batch(self, decoder_batch, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # convert list of tokens into chainer variable list
        var_dec = (Variable(decoder_batch.T, volatile=(not train)))

        # Initialise first decoded word to GOID
        pred_word = var_dec[0]

        loss = 0

        seq_len, batch_size = var_dec.shape
        # for all sequences in the batch, feed the characters one by one
        for i in range(1, seq_len):
            # encode tokens
            self.decode(pred_word, train)

            if self.attn:
                cv, _ = self.compute_context_vector()
                cv_hdec = F.concat((cv, self[self.lstm_dec[-1]].h), axis=1)
                ht = F.tanh(self.context(cv_hdec))
                predicted_out = self.out(ht)
            else:
                predicted_out = self.out(self[self.lstm_dec[-1]].h)

            prob = F.softmax(predicted_out)
            pred_word = F.expand_dims(F.argmax(prob, axis=1), -1)

            w = var_dec[i]
            loss_arr = F.softmax_cross_entropy(predicted_out, w,
                                               class_weight=self.mask_pad_id)
            loss += loss_arr

        return loss

    #--------------------------------------------------------------------
    # For batch size > 1
    #--------------------------------------------------------------------
    def encode_decode_train_batch(self, batch_data, src_lim, tar_lim, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.reset_state()
        fwd_encoder_batch = xp.empty((0, src_lim), dtype=xp.int32)
        rev_encoder_batch = xp.empty((0, src_lim), dtype=xp.int32)
        cnn_batch = xp.empty((0, src_lim), dtype=xp.int32)
        decoder_batch = xp.empty((0, tar_lim+2), dtype=xp.int32)

        for src, tar in batch_data:
            fwd_encoder_batch = xp.vstack((fwd_encoder_batch, self.pad_list(src, src_lim)))
            rev_encoder_batch = xp.vstack((rev_encoder_batch, self.pad_list(src[::-1], src_lim)))
            cnn_batch=xp.vstack((cnn_batch,self.pad_list(src, src_lim)))

            tar_data = [GO_ID] + tar + [EOS_ID]
            decoder_batch = xp.vstack((decoder_batch, self.pad_list(tar_data,
                                                                    tar_lim+2, at_start=False)))
        
        #print(fwd_encoder_batch.data.shape)
        cnn_input = (Variable(cnn_batch, volatile=(not train)))
        cnn_output= self.conv_layer(cnn_input,self.embed_enc,self.cnn_enc,train=train)
        #print(cnn_output.data.shape)
        cnn_output=F.reshape(cnn_output,(cnn_output.data.shape[0],cnn_output.data.shape[2],cnn_output.data.shape[3]))
        # encode list of words/tokens
        self.encode_batch(cnn_output, train=train)


        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode and compute loss
        self.loss = self.decode_batch(decoder_batch, train=train)

        return self.loss


# In[ ]:

    #--------------------------------------------------------------------
    # For SGD - Batch size = 1
    #--------------------------------------------------------------------
    def encode_list(self, in_word_list, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        # convert list of tokens into chainer variable list
        var_en = (Variable(xp.asarray(in_word_list, dtype=np.int32).reshape((-1,1)),
                           volatile=(not train)))

        var_rev_en = (Variable(xp.asarray(in_word_list[::-1], dtype=np.int32).reshape((-1,1)),
                           volatile=(not train)))

        first_entry = True


        # encode tokens
        for f_word, r_word in zip(var_en, var_rev_en):
            self.encode(f_word, self.lstm_enc, train)
            self.encode(r_word, self.lstm_rev_enc, train)

            if first_entry == False:
                forward_states = F.concat((forward_states, self[self.lstm_enc[-1]].h), axis=0)
                backward_states = F.concat((self[self.lstm_rev_enc[-1]].h, backward_states), axis=0)
            else:
                forward_states = self[self.lstm_enc[-1]].h
                backward_states = self[self.lstm_rev_enc[-1]].h
                first_entry = False

        self.enc_states = F.concat((forward_states, backward_states), axis=1)

    def compute_context_vector(self, batches=True):
        xp = cuda.cupy if self.gpuid >= 0 else np

        batch_size, n_units = self[self.lstm_dec[-1]].h.shape
        # attention weights for the hidden states of each word in the input list

        if batches:
            # masking pad ids for attention
            weights = F.batch_matmul(self.enc_states, self[self.lstm_dec[-1]].h)
            #print(self[self.lstm_dec[-1]].h.data.shape)
            #print(self.enc_states.data.shape)
            #print(weights.data.shape)
            #print(self.mask.data.shape)
            weights = F.where(self.mask, weights, self.minf)
            #print(weights.data.shape)
            alphas = F.softmax(weights)

            # compute context vector
            cv = F.reshape(F.batch_matmul(F.swapaxes(self.enc_states, 2, 1), alphas),
                                         shape=(batch_size, n_units))
        else:
            # without batches
            alphas = F.softmax(F.matmul(self[self.lstm_dec[-1]].h, self.enc_states, transb=True))
            # compute context vector
            if self.attn == SOFT_ATTN:
                cv = F.batch_matmul(self.enc_states, F.transpose(alphas))
                cv = F.transpose(F.sum(cv, axis=0))
            else:
                print("nothing to see here ...")

        return cv, alphas

    #--------------------------------------------------------------------
    # For SGD - Batch size = 1
    #--------------------------------------------------------------------
    def encode_decode_train(self, in_word_list, out_word_list, train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.reset_state()
        # Add GO_ID, EOS_ID to decoder input
        decoder_word_list = [GO_ID] + out_word_list + [EOS_ID]
        # encode list of words/tokens
        self.encode_list(in_word_list, train=train)
        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode and compute loss
        # convert list of tokens into chainer variable list
        var_dec = (Variable(xp.asarray(decoder_word_list, dtype=np.int32).reshape((-1,1)),
                            volatile=not train))
        # Initialise first decoded word to GOID
        pred_word = Variable(xp.asarray([GO_ID], dtype=np.int32), volatile=not train)

        # compute loss
        self.loss = 0
        # decode tokens
        for next_word_var in var_dec[1:]:
            self.decode(pred_word, train=train)
            if self.attn:
                cv, _ = self.compute_context_vector(batches=False)
                cv_hdec = F.concat((cv, self[self.lstm_dec[-1]].h), axis=1)
                ht = F.tanh(self.context(cv_hdec))
                predicted_out = self.out(ht)
            else:
                predicted_out = self.out(self[self.lstm_dec[-1]].h)
            # compute loss
            prob = F.softmax(predicted_out)
            pred_word = F.argmax(prob)
            pred_word = Variable(xp.asarray([pred_word.data], dtype=np.int32), volatile=not train)
            self.loss += F.softmax_cross_entropy(predicted_out, next_word_var)
        report({"loss":self.loss},self)

        return self.loss

    #--------------------------------------------------------------------
    # For SGD - Batch size = 1
    #--------------------------------------------------------------------
    def decoder_predict(self, start_word, max_predict_len=20):
        xp = cuda.cupy if self.gpuid >= 0 else np
        alpha_arr = xp.empty((0,self.enc_states.shape[0]), dtype=xp.float32)

        # return list of predicted words
        predicted_sent = []
        # load start symbol
        prev_word = Variable(xp.asarray([start_word], dtype=np.int32), volatile=True)
        pred_count = 0
        pred_word = None

        # start pred loop
        while pred_count < max_predict_len and pred_word != (EOS_ID) and pred_word != (PAD_ID):
            self.decode(prev_word, train=False)

            if self.attn:
                cv, alpha_list = self.compute_context_vector(batches=False)
                # concatenate hidden state
                cv_hdec = F.concat((cv, self[self.lstm_dec[-1]].h), axis=1)
                # add alphas row
                alpha_arr = xp.vstack((alpha_arr, alpha_list.data))

                ht = F.tanh(self.context(cv_hdec))
                prob = F.softmax(self.out(ht))
            else:
                prob = F.softmax(self.out(self[self.lstm_dec[-1]].h))

            if self.gpuid >= 0:
                prob = cuda.to_cpu(prob.data)[0].astype(np.float64)
            else:
                prob = prob.data[0].astype(np.float64)
            #prob /= np.sum(prob)
            #pred_word = np.random.choice(range(len(prob)), p=prob)
            pred_word = np.argmax(prob)
            predicted_sent.append(pred_word)
            prev_word = Variable(xp.asarray([pred_word], dtype=np.int32), volatile=True)
            pred_count += 1
        return predicted_sent, alpha_arr

    #--------------------------------------------------------------------
    # For SGD - Batch size = 1
    #--------------------------------------------------------------------
    def encode_decode_predict(self, in_word_list, max_predict_len=20):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.reset_state()
        # encode list of words/tokens
        in_word_list_no_padding = [w for w in in_word_list if w != PAD_ID]
        # enc_states = self.encode_list(in_word_list, train=False)
        self.encode_list(in_word_list, train=False)
        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode starting with GO_ID
        predicted_sent, alpha_arr = self.decoder_predict(GO_ID, max_predict_len)
        return predicted_sent, alpha_arr
# In[ ]:

