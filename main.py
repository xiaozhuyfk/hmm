"""
File: main.py
Author: Hongyu Li (hongyul)
"""

import logging, globals
from util import (
    load_symbols,
    load_emission,
    load_transition,
    load_tags,
    load_priors,
    load_tweets,
    vectorize_sequence,
)
from hmm import HMM

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def train():
    config_options = globals.config
    symbol_file = config_options.get('POS', 'symbols')
    emission_file = config_options.get('POS', 'emission')
    transition_file = config_options.get('POS', 'transition')
    tag_file = config_options.get('POS', 'tags')
    prior_file = config_options.get('POS', 'prior')
    tweets_file = config_options.get('POS', 'tweets')

    symbols, word_idx = load_symbols(symbol_file)
    emission = load_emission(emission_file)
    transition = load_transition(transition_file)
    tags = load_tags(tag_file)
    priors = load_priors(prior_file)
    tweets = load_tweets(tweets_file)
    tweets_vec = [vectorize_sequence(tweet, word_idx) for tweet in tweets]

    hmm = HMM(transition=transition, emission=emission, priors=priors)
    for tweet in tweets_vec:
        print hmm.forward(tweet), hmm.backward(tweet)
        print " ".join([tags[idx] for idx in hmm.viterbi(tweet)])

    # tokens = ["@HippieOfLove", "u", "eat", "sushi", "?"]
    # sequence = [word_idx.get(t) for t in tokens]
    # print [tags[idx] for idx in hmm.viterbi(sequence)]

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Choose to learn or test AMA')

    parser.add_argument('--config',
                        default='config.cfg',
                        help='The configuration file to use')
    subparsers = parser.add_subparsers(help='command help')
    train_parser = subparsers.add_parser('train', help='Train memory network')
    train_parser.set_defaults(which='train')

    args = parser.parse_args()

    # Read global config
    globals.read_configuration(args.config)

    if args.which == 'train':
        train()


if __name__ == '__main__':
    main()