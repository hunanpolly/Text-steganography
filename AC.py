import torch
import sys
import copy
import utils
import Huffman
import os
import lm
import numpy as np
from loguru import logger

import argparse

parser = argparse.ArgumentParser(description='AC')

parser.add_argument('-dataset', type=str, default='Twitter',\
                    help='The training corpus [default:Twitter]')
parser.add_argument('-generate-num', type=int, default=100, \
                    help='The number of generated stego text [default:100]')

parser.add_argument('-seed', type=int, default=123, \
                    help='The random seed for initialization [default:123]')
parser.add_argument('-idx-gpu', type=str, default='0', \
                    help='The number of gpu for training [default:0]')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.idx_gpu

# Setting the random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += int(bit)*(2**i)
    return res


def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}'%num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i



def prob_sort(model, inp):
    log_prob = model(inp)
    prob = torch.exp(log_prob)[:, -1 :].reshape(-1)
    prob[1] = 0
    probs = prob / prob.sum()
    probs, indices = prob.sort(descending = True)
    return probs, indices


def main(args):

    # ==================
    # hyper-parameters
    # ==================
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
    
    if DATASET == "IMDB":
        LOAD_EPOCH = 29
    if DATASET == "News":
        LOAD_EPOCH = 30
    if DATASET == 'Twitter':
        LOAD_EPOCH = 25 
    

    all_var = locals()
    print()
    for var in all_var:
        if var != 'var_name':
            print("{0:15} ".format(var), all_var[var])
    print()

    # ===========
    # data
    # ===========
    data_path = '/data/Text_data/corpora/original/' + DATASET + '2020.txt'
    vocabulary = utils.Vocabulary(
                    data_path,
                    max_len = MAX_LEN,
                    min_len = MIN_LEN,
                    word_drop = WORD_DROP,
                    )

    # ===========
    # building model
    # ===========

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    print()
    model = lm.LM(
                    vocab_size = vocabulary.vocab_size,
                    embed_size = EMBED_SIZE,
                    hidden_dim = HIDDEN_DIM,
                    num_layers = NUM_LAYERS,
                    dropout_rate = DROPOUT_RATE
                    )
    model.to(device)
    model.load_state_dict(torch.load('models/' + DATASET + '-' + \
                            str(LOAD_EPOCH) + '.pkl', map_location=device))
    print('checkpoint loaded...')
    print()

    print('start steganography...')
    
    num_bits_list = [1,2,3,4,5] 
    logger.add(DATASET + '_AC_{time}.log')
    for num_bits in num_bits_list:
        logger.info("The num_bits is: " + str(num_bits)) 
        os.makedirs('stego/' + DATASET, exist_ok=True)
        # read bit streams
        with open('bit_stream/bit_stream.txt', 'r', encoding='utf8') as f:
            bit_stream = f.read().strip()
            bit_stream += bit_stream

        bit_index = int(torch.randint(0, high=1000, size=(1,)))

        model.eval()
        with torch.no_grad():
            stega_text = []
            stega_bits = []
            
            precision = 26
            max_val = 2**precision
            threshold = 2**(-precision)
            cur_interval = [0, max_val]


            import time
            start = time.time()
            while len(stega_text) < GENERATE_NUM:
                if len(stega_text) % 10 == 0:
                    Log = "the number of stega_text: "+str(len(stega_text))
                    logger.info(Log)
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

                    probs, indices = prob_sort(model, x) 
                    
                    C_probs = probs[:2**num_bits]       #Candidate probs
                    C_indices = indices[:2**num_bits]   #Candidate indiceso

                    C_probs = list(C_probs.cpu().numpy())


                    cur_int_range = cur_interval[1] - cur_interval[0]
                    cur_threshold = 1 / cur_int_range
                    prob_temp_int = C_probs / sum(C_probs) * cur_int_range
                    prob_temp_int = np.int32(np.rint(prob_temp_int))

                    cum_probs = prob_temp_int.cumsum(0)

                    overfill_index = np.argwhere(cum_probs > cur_int_range)

                    if len(overfill_index) > 0:
                        cum_probs = cum_probs[:overfill_index[0][0]]

                    cum_probs += cur_int_range - cum_probs[-1]

                    probs_final = copy.deepcopy(cum_probs)
                    probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                    cum_probs += cur_interval[0]

                    message_bits = bit_stream[bit_index:bit_index+precision]

                    if bit_index+precision > len(bit_stream):
                        message_idx = message_bits + [0] * \
                                        (bit_index+precision-len(bit_stream))
                    message_idx = bits2int(reversed(message_bits))
                    selection = np.argwhere(cum_probs > message_idx)[0][0]

                    new_int_bottom = cum_probs[selection - 1] \
                                        if selection>0 else cur_interval[0]
                    new_int_top = cum_probs[selection]

                    new_int_bottom_bits_inc = list(reversed(
                                         int2bits(new_int_bottom, precision)))
                    new_int_top_bits_inc = list(reversed(
                                         int2bits(new_int_top-1, precision)))

                    num_bits_encoded = num_same_from_beg(
                                                new_int_bottom_bits_inc,
                                                new_int_top_bits_inc)

                    new_int_bottom_bits = new_int_bottom_bits_inc[ \
                                   num_bits_encoded:] + [0] * num_bits_encoded

                    new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:]\
                                       +  [1] * num_bits_encoded

                    cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
                    cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1

                    gen = int(indices[selection])

                    stega_sentence += [vocabulary.i2w[gen]]
                    embed_bit += bit_stream[bit_index:bit_index + \
                                                        num_bits_encoded]
                    bit_index += num_bits_encoded

                    if vocabulary.i2w[gen] == '_EOS':
                        break

                    x = torch.cat([x,torch.LongTensor(
                                        [[gen]]).to(device)], dim=1)
                
                # check
                if '_EOS' in stega_sentence:
                    stega_sentence.remove('_EOS')
                if (len(stega_sentence) <= MAX_LEN) and \
                                    (len(stega_sentence) > MIN_LEN):
                    stega_text.append(stega_sentence)
                    stega_bits.append(embed_bit)

            #write file
            print("Time_Cost: ", time.time() - start)
            with open('stego/' + DATASET + '/AC_' + str(2**num_bits) \
                                    + 'CW.txt', 'w', encoding = 'utf8') as f:
                for sentence in stega_text:
                    f.write(' '.join(sentence) + '\n')
            with open('stego/' + DATASET + '/AC_' + str(2**num_bits) \
                                    + 'CW.bit', 'w', encoding = 'utf8') as f:
                for bits in stega_bits:
                    f.write(bits + '\n')



if __name__ == '__main__':
    main(args)
