from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import sys

sys.stdout = open('pointerWithPreCov.log','w')

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

#data_dir = r'E:\master\key_word_extraction\keyphrase\data'
data_dir = r'/users5/yuzhouzhang/seq2seq+pointNet/MyPointerNet/data'
corpus_name = os.path.join(data_dir, 'xiao_mi')
datafile = os.path.join(corpus_name, 'train_xiaomi.txt')
testfile = os.path.join(corpus_name, 'test_xiaomi.txt')
save_dir = os.path.join(corpus_name, "save")

#vec_root = r'E:\master\key_word_extraction\data\wiki.zh'
vec_root = r'/users5/yuzhouzhang/word2vec'
vec = 'wiki.zh.vec'

# Default word tokens
UNK_token = 0  #
PAD_token = 1  # Used for padding short sentences
SOS_token = 5  # Start-of-sentence token
EOS_token = 5  # End-of-sentence token


class Voc:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.embedding = []
        self.num = 0
        self.num_words = 0
        self.hidden_size = 0

    def processEmbedding(self, vec_path):
        fin = open(vec_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        self.num_words, self.hidden_size = map(int, fin.readline().split())
        self.word2index['UNK'] = 0
        self.index2word[0] = 'UNK'
        self.embedding.append([random.random() for _ in range(self.hidden_size)])
        self.num_words += 1
        self.num += 1
        for line in fin:
            tokens = line.rstrip().split(' ')
            self.word2index[tokens[0]] = self.num
            self.index2word[self.num] = tokens[0]
            self.num += 1
            self.embedding.append(list(map(float, tokens[1:])))
        assert self.num == self.num_words

    def judgeWord(self, word):
        return word in self.word2index


voc = Voc()
voc.processEmbedding(os.path.join(vec_root, vec))

MAX_LENGTH = 1000  # Maximum sentence length to consider


def normalizeString(voc, s):
    return s
    # return ' '.join([word for word in s.split(' ') if voc.judgeWord(word)])


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


# Using the functions defined above, return a populated voc object and pairs list
def prepareFilterPairs(datafile, voc):
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    print("Start preparing training data ...")
    pairs = [[normalizeString(voc, s) for s in l.split('\t')] for l in lines]
    print("Read {!s} sentence pairs".format(len(pairs)))
    filtered_pairs = [pair for pair in pairs if filterPair(pair)]
    print("Trimmed to {!s} sentence pairs".format(len(filtered_pairs)))
    return filtered_pairs


pairs = prepareFilterPairs(datafile, voc)
for pair in pairs[:3]:
    print(pair)


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] if voc.judgeWord(word) else UNK_token for word in sentence.split(' ')] + [EOS_token]


def pointerFromSentence(sentence, target):
    dic = dict()
    for index, word in enumerate(sentence.split(' ')):
        dic[word] = index
    return [dic[word] for word in target.split(' ')] + [len(sentence.split(' '))]



def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def outputVar(l1, l2):
    # indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    indexes_batch = [pointerFromSentence(sentence, target) for sentence, target in zip(l1, l2)]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(input_batch, output_batch)
    return inp, lengths, output, mask, max_target_len


# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


# input_seq: batch of input sentences; shape=(max_length, batch_size)
# input_lengths: list of sentence lengths corresponding to each sentence in the batch; shape=(batch_size)
# hidden: hidden state; shape=(n_layers x num_directions, batch_size, hidden_size)
# outputs: output features from the last hidden layer of the GRU (sum of bidirectional outputs); shape=(max_length, batch_size, hidden_size)
# hidden: updated hidden state from GRU; shape=(n_layers x num_directions, batch_size, hidden_size)
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# encoder_output shape=(max_length, batch_size, hidden_size)
# The output of this module is a softmax normalized weights tensor of shape (batch_size, max_length).
class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.Wh = nn.Linear(hidden_size, hidden_size)
        self.Ws = nn.Linear(hidden_size, hidden_size)
        self.Wc = nn.Linear(1, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def score(self, hidden, encoder_output, coverage):
        # energy shape=(max_length, batch_size, hidden_size)
        energy = (self.Wh(hidden.expand(encoder_output.size(0), -1, -1)) + self.Ws(encoder_output) + self.Wc(coverage.t().unsqueeze(2))).tanh()
        return self.v(energy).squeeze(2)

    def forward(self, hidden, encoder_outputs, coverage):
        attn_energies = self.score(hidden, encoder_outputs, coverage)
        attn_energies = attn_energies.t()   # shape=(batch_size, max_length)
        return F.softmax(attn_energies, dim=1)

# input_step: one time step (one word) of input sequence batch; shape=(1, batch_size)
# last_hidden: final hidden layer of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)
# encoder_outputs: encoder model’s output; shape=(max_length, batch_size, hidden_size)
# coverage: shape=(max_length, batch_size)
# output: softmax normalized tensor giving probabilities of each word being the correct next word in the decoded sequence; shape=(batch_size, max_length)
# hidden: final hidden state of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)


class PointerDecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, n_layers=1, dropout=0.1):
        super(PointerDecoderRNN, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.attn = Attn(hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs, coverage):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs, coverage)
        # print('attn output: ', attn_weights)
        return attn_weights, hidden


def maskNLLLoss(inp, target, mask, coverage, t, lamda=1):  # inp, coverage shape = (batch_size, max_length)
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    crossloss = crossEntropy.masked_select(mask).mean()

    covloss = torch.min(inp, coverage)
    covloss = torch.sum(covloss.t()[t].masked_select(mask), dim=0)

    loss = crossloss + lamda * covloss
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, coverage, max_length=MAX_LENGTH):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    coverage = coverage.to(device)
    #print('to device: ', coverage)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, coverage
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)

            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], coverage, t)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

            # update coverage
            coverage = coverage + decoder_output

    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, coverage
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            # Calculate and accumulate loss
            mask_loss,  nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], coverage, t)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

            # update coverage
            coverage = torch.add(coverage, decoder_output)

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding,
               encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip,
               corpus_name, loadFilename):
    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]

        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        coverage = torch.zeros(batch_size, max(lengths))

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, coverage)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,
                                                                                          iteration / n_iteration * 100,
                                                                                          print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir,
                                     '{}_{}-{}-{}'.format(model_name, encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        coverage = torch.zeros(len(input_length), len(input_seq), device=device)

        # Iteratively decode one word token at a time
        for _ in range(10):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, coverage)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.LongTensor([[input_seq[pointer]] for pointer in decoder_input]).cuda()
            #decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    # decoded_words = [voc.index2word[token.item()] for token in tokens]
    sentence_index = sentence.strip().split(' ')
    sentence_index.append('EOS')
    decoded_words = [sentence_index[token.item()] for token in tokens]
    return decoded_words


def evaluateTest(encoder, decoder, searcher, voc):
    test_data = open(testfile, 'r').readlines()
    input_doc = [data.strip().split('\t')[0] for data in test_data]
    targets = [set(data.strip().split('\t')[1].split(' ')) for data in test_data]
    predicts = [set([x for x in evaluate(encoder, decoder, searcher, voc, doc) if not (x == 'EOS' or x == 'PAD')]) for doc in input_doc]
    for t, p in zip(targets, predicts):
        print('targets:', t, 'predicts:', p)
    interSec = [len([x for x in target if x in predict]) for target, predict in zip(targets, predicts)]
    P = [0.0 if len(NP) == 0 else float(TP) / float(len(NP)) for TP, NP in zip(interSec, predicts)]
    R = [0.0 if len(TN) == 0 else float(TP) / float(len(TN)) for TP, TN in zip(interSec, targets)]
    F = [0.0 if abs(p*r) < 1e-12 else 2*p*r / (p+r) for p, r in zip(P, R)]
    print('Average  P: {}; R: {}; F: {}'.format(sum(P)/len(P), sum(R)/len(R), sum(F)/len(F)))


# Configure models
model_name = 'pointerWithPreCov'
hidden_size = voc.hidden_size
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir,
#                            '{}_{}-{}-{}'.format(model_name, encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding.from_pretrained(torch.FloatTensor(voc.embedding))
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = PointerDecoderRNN(embedding, hidden_size, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
# evaluateInput(encoder, decoder, searcher, voc)
evaluateTest(encoder, decoder, searcher, voc)

