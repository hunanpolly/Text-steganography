import heapq
import os
import utils
import lm
import sys
import torch
import numpy as np
import random

import argparse


parser = argparse.ArgumentParser(description='Patient-HC.py')

parser.add_argument('-dataset', type=str, default=None, \
                    help='The training corpus [default:None]')
parser.add_argument('-generate_num', type=int, default=None, \
                    help='The number of generated stego text [default:None]')
parser.add_argument('-idx-gpu', type=str, default='0', \
                    help='The index of the gpu for runing [default:0]')


args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.idx_gpu


def build_min_heap(freqs, inds=None):
    '''Returns a min-heap of (frequency, token_index).'''
    inds = inds or range(len(freqs))
    # Add a counter in tuples for tiebreaking
    freq_index = [(freqs[ind], i, ind) for i, ind in enumerate(inds)]
    heapq.heapify(freq_index)
    return freq_index



def huffman_tree(heap):
    '''Returns the Huffman tree given a min_heap of indices and frequencies.'''
    # Add a counter in tuples for tiebreaking
    t = len(heap)
    # Runs for n iterations where n = len(heap)
    while len(heap) > 1:
        # Remove the smallest two nodes. O(log n)
        freq1, i1, ind1 = heapq.heappop(heap)
        freq2, i2, ind2 = heapq.heappop(heap)
        # Create a parent node for these two nodes
        parent_freq = freq1 + freq2
        # The left child is the one with the lowest frequency
        parent_ind = (ind1, ind2)
        # Insert this parent onde. O(log n)
        heapq.heappush(heap, (parent_freq, t, parent_ind))
        t += 1

    code_tree = heap[0][2]
    # Total runtime O(n log n)
    return code_tree



def tv_huffman(code_tree, p):
    '''Returns the total variation between a distribution over tokens and the 
       distrubution induced by a huffman coding of (a subset of) the tokens.

    Args:
        code_tree: tuple.
            Huffman codes as represented by a binary tree. 
            It might miss some token.
        p: array of size of the vocabulary.
           The distribution over all tokens.
    '''
    tot_l1 = 0
    # the tokens absent in the Huffman codes have probability 0
    absence = np.ones_like(p)
    tot_ce = 0
    stack = []
    stack.append((code_tree, 0))
    while len(stack) > 0:
        node, depth = stack.pop()
        if type(node) is tuple:
            #Expand the children
            left_child, right_child = node
            # Push the children and their depths onto the stack
            stack.append((left_child, depth + 1))
            stack.append((right_child, depth + 1))
        else:
            # a leaf node
            ind = node
            tot_l1 += abs(p[ind] - 2** (-depth))
            absence[ind] = 0
            # The KL divergence of true distribution || Huffman distribution
            tot_ce += p[ind] * depth + p[ind] * np.log2(p[ind]+1e-11)
    # Returns total variation
    return 0.5 * (tot_l1 + np.sum(absence * p)), tot_ce


def total_variation(p, q):
    '''Returns the totak variation of two distributions over a finite set.'''
    return 0.5 * np.sum(np.abs(p - q))



def main(args):
    # =====================
    # Hyper-parameters
    # =====================
    DATASET = args.dataset
    WORD_DROP = 10
    MIN_LEN = 5
    MAX_LEN = 200
    EMBED_SIZE = 800
    HIDDEN_DIM = 800
    NUM_LAYERS = 3
    DROPOUT_RATE = 0.0
    MAX_GENERATE_LENGTH = 200
    GENERATE_NUM = args.generate_num


    if DATASET == 'IMDB':
        LOAD_EPOCH = 29
    if DATASET == 'News':
        LOAD_EPOCH = 30
    if DATASET == 'Twitter':
        LOAD_EPOCH = 25


    all_var = locals()
    print()
    for var in all_var:
        if var != 'var_name':
            print('{0:15}'.format(var), all_var[var])
    print()

    # ============
    # loading data
    # ============
    data_path = '/data/wlpeng/Text_data/corpora/original/' + DATASET + '2020.txt'
    vocabulary = utils.Vocabulary(
                    data_path,
                    max_len = MAX_LEN,
                    min_len = MIN_LEN,
                    word_drop = WORD_DROP,
                    )

    # ==============
    # buliding model
    # ==============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))
    print()
    model = lm.LM(
                    vocab_size = vocabulary.vocab_size,
                    embed_size = EMBED_SIZE,
                    hidden_dim = HIDDEN_DIM,
                    num_layers = NUM_LAYERS,
                    dropout_rate = DROPOUT_RATE
                    )
    model.to(device)
    model.load_state_dict(torch.load('models/' + DATASET + '-' +\
                            str(LOAD_EPOCH) + '.pkl', map_location=device))
    print('checkpoint loaded...')
    print()

    print('start steganography...')

    tv_threshold_list = [0.5, 1.0, 1.5, 2.0, 2.5]
    for tv_threshold in tv_threshold_list:
        print('The tv_threshold is: ', tv_threshold)
        os.makedirs('stego/' + DATASET, exist_ok = True)
        # read bit streams
        with open('bit_stream/bit_stream.txt', 'r', encoding='utf8') as f:
            bit_stream = f.read().strip()
            bit_stream += bit_stream

        bit_stream = list(bit_stream)

        bit_index = int(torch.randint(0, high=1000, size=(1,)))

        model.eval()
        with torch.no_grad():
            stega_text = []
            stega_bits = []

            import time
            start = time.time()
            for num in range(GENERATE_NUM):
                if num != 0 and num%10 == 0:
                    print("Generated stego shentence is: ", num)
                stega_sentence = []
                embed_bit = ''
                x = torch.LongTensor([[vocabulary.w2i['_BOS']]]).to(device)
                samp = model.sample(x)
                stega_sentence.append(vocabulary.i2w[samp.reshape(-1)\
                                .cpu().numpy()[0]])
                x = torch.cat([x, samp], dim = 1)

                for i in range(MAX_GENERATE_LENGTH - 1):
                    if '_EOS' in stega_sentence:
                        break
                    # conditional probability distribution
                    log_prob = model(x)
                    prob = torch.exp(log_prob)[:, -1, :].reshape(-1)[:2000]
                    prob[1] = 0
                    prob = prob / prob.sum()
                    heap = build_min_heap(prob)
                    hc = huffman_tree(heap)
                    if tv_huffman(hc, prob.cpu().numpy())[0] <= tv_threshold:
                        # Huffman-decode the cipher text into a token
                        # Consume the cipher text until a token is generated
                        decoder_state = hc
                        while type(decoder_state) is tuple:
                            left, right = decoder_state
                            try:
                                bit = bit_stream.pop(0)
                            except IndexError:
                                bit = self.random.choice(2)
                            # 0 => left, 1 => right
                            decoder_state = left if bit == 0 else right
                            embed_bit += bit
                        # Decoder settles in a leaf node
                        ind = decoder_state # int type
                        bit_index += len(embed_bit)
                    else:
                        samp = model.sample(x)
                        ind = samp.reshape(-1).cpu().numpy()[0]

                    stega_sentence += [vocabulary.i2w[ind]]

                    if vocabulary.i2w[ind] == '_EOS':
                        break

                    x = torch.cat([x, torch.LongTensor(
                                            [[ind]]).to(device)], dim=1)

                # check
                if '_EOS' in stega_sentence:
                    stega_sentence.remove('_EOS')
                if (len(stega_sentence) <= MAX_LEN) and \
                            (len(stega_sentence) > MIN_LEN):
                    stega_text.append(stega_sentence)
                    stega_bits.append(embed_bit)

                exists = None
                if len(stega_text) != 0 and len(stega_text)%1 == 0:
                    if os.path.exists(
                        'stego/'+DATASET+'/P_HC_'+str(tv_threshold)+'.txt'):
                        with open('stego/' + DATASET + '/P_HC_' + \
                            str(tv_threshold)+'.txt', 'r',encoding='utf8') as f:
                            exists = f.readlines()

                    # write file
                    with open('stego/' + DATASET + '/P_HC_' + \
                        str(tv_threshold) + '.txt', 'a',encoding='utf8') as f:
                        for sentence in stega_text:
                            sen = ' '.join(sentence) + '\n'
                            if exists is not None and sen in exists:
                                continue
                            #f.write(' '.join(sentence) + '\n')
                            f.write(sen)
                    
                    with open('stego/' + DATASET + '/P_HC_' + \
                        str(tv_threshold)+'.bit', 'a', encoding='utf8') as f:

                        for bits in stega_bits:
                            f.write(bits + '\n')

                    stega_text = []
                    stega_bits = []

                            
                           




if __name__ == '__main__':
    main(args)
