import torch
from torch import nn
import torch.optim as optim
import numpy as np
import random
import utils
import lm
import os
import argparse

parser = argparse.ArgumentParser(description='Training LM')


parser.add_argument('-dataset', type=str, default=None, \
                    help='The training corpus [default:None]')
parser.add_argument('-idx-gpu', type=str, default='0', \
                    help='The index of the gpu for runing [default:0]')
parser.add_argument('-seed', type=int, default=123, \
                    help='The random seed for initialization [default:123]')
parser.add_argument('-batch-size', type=int, default=32, \
                    help='The batch-size of the training phase [default:32]')


args = parser.parse_args()


# Setting the random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


os.environ["CUDA_VISIBLE_DEVICES"] = args.idx_gpu


def main(args):
    # =============
    # Hyer-parameters
    # =============
    DATASET = args.dataset
    RATIO = 0.9
    WORD_DROP = 10
    MIN_LEN = 5
    MAX_LEN = 200
    BATCH_SIZE = args.batch_size
    EMBED_SIZE = 800
    HIDDEN_DIM = 800
    NUM_LAYERS = 3
    DROPOUT_RATE = 0.5
    START_EPOCH = 0
    EPOCH = 30
    LEARNING_RATE = 0.001
    MAX_GENERATE_LENGTH = 20
    GENERATE_EVERY = 5
    PRINT_EVERY = 100
    SEED = args.seed


    all_var = locals()
    print()
    for var in all_var:
        if var != 'var_name':
            print("{0:15}  ".format(var), all_var[var])

    print()

    #==================
    # data
    #==================
    data_path = '/data/Text_data/corpora/original/' + DATASET + '2020.txt'
    train_path = '/data/Text_data/corpora/train_corpora/train_' + DATASET
    test_path = '/data/Text_data/corpora/train_corpora/test_' + DATASET
    vocabulary = utils.Vocabulary(
                data_path,
                max_len = MAX_LEN,
                min_len = MIN_LEN,
                word_drop = WORD_DROP
                )

    utils.split_corpus(data_path, train_path, test_path, max_len=MAX_LEN, 
                       min_len=MIN_LEN, ratio=RATIO, seed=SEED)
    train = utils.Corpus(train_path, vocabulary, max_len=MAX_LEN, 
                         min_len=MIN_LEN)
    test = utils.Corpus(test_path, vocabulary, max_len=MAX_LEN,
                        min_len=MIN_LEN)

    train_generator = utils.Generator(train.corpus, vocabulary=vocabulary)
    test_generator = utils.Generator(test.corpus, vocabulary=vocabulary)


    # ==================
    # Building model
    # ==================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    model = lm.LM(
                    vocab_size = vocabulary.vocab_size, 
                    embed_size = EMBED_SIZE,
                    hidden_dim = HIDDEN_DIM,
                    num_layers = NUM_LAYERS,
                    dropout_rate = DROPOUT_RATE
                   ) 
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total params: {:d}".format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if \
                                    p.requires_grad)
    print("Trainable params:{:d}".format(total_trainable_params))
    
    criterion = nn.NLLLoss(ignore_index = vocabulary.w2i["_PAD"])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, 
                           betas=(0.9,0.999), eps=1e-08, 
                           weight_decay=0, amsgrad=False)
    print()


    # ==============
    # Training and testing
    # ==============
    best_loss = 1000000
    step = 0
    if START_EPOCH > 0:
        model.load_state_dict(torch.load('models/' + DATASET + '-' + \
                               str(START_EPOCH) + '.pkl', map_location=device))
    
    for epoch in range(START_EPOCH + 1, EPOCH + 1):
        train_g = train_generator._build_generator(BATCH_SIZE)
        test_g = test_generator._build_generator(BATCH_SIZE)
        train_loss = []
        model.train()

        while True:
            try:
                text = train_g.__next__()
            except:
                break

            optimizer.zero_grad()
            text_in = text[:,:-1]
            text_target = text[:, 1:]
            y = model(torch.from_numpy(text_in).long().to(device))
            loss = criterion(y.reshape(-1, vocabulary.vocab_size), 
                   torch.from_numpy(text_target).reshape(-1).long().to(device))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            step += 1
            torch.cuda.empty_cache()

            if step % PRINT_EVERY == 0:
                print('step {:d} training loss {:.4f}'.format(step, loss.item()))


        test_loss = []
        model.eval()
        with torch.no_grad():
            while True:
                try:
                    text = test_g.__next__()
                except:
                    break
                text_in = text[:, :-1]
                text_target = text[:,1:]
                y = model(torch.from_numpy(text_in).long().to(device))
                loss = criterion(y.reshape(-1, vocabulary.vocab_size), 
                     torch.from_numpy(text_target).reshape(-1).long().to(device))
                test_loss.append(loss.item())
                torch.cuda.empty_cache()

        print('eoch {:d} training loss {:.4f}  test_loss {:.4f}' \
                .format(epoch, np.mean(train_loss), np.mean(test_loss)))

        if np.mean(test_loss) < best_loss:
            best_loss = np.mean(test_loss)
            print('-----------------------------------')
            print('saving parameters')
            torch.save(model.state_dict(), 'models/' + DATASET + '-' \
                        + str(epoch) + '.pkl')
            print('----------------------------------')

        if (epoch + 1) % GENERATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                #generating text
                x = torch.LongTensor([[vocabulary.w2i['_BOS']]] * 3).to(device)
                for i in range(MAX_GENERATE_LENGTH):
                    samp = model.sample(x)
                    x = torch.cat([x, samp], dim=1)
                x = x.cpu().numpy()
            print('-------------------------------')
            for i in range(x.shape[0]):
                print(' '.join([vocabulary.i2w[_] for _ in list(x[i, :]) \
                        if _ not in [vocabulary.w2i['_BOS'], 
                            vocabulary.w2i['_EOS'], 
                            vocabulary.w2i['_PAD']
                            ]]))
            print('------------------------------')



if __name__ == '__main__':
    main(args)

