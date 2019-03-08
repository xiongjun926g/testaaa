"""Parse a corpus.

Parsing is done in parallel with StanfordNLP and CoreNLP.

TODO: spaCy
"""

import argparse
import os.path

from parse_stanford import parse_stanfordnlp, ssplit_stanfordnlp
# from parse_corenlp import parse_corenlp
from task_data import DATA_DIR, LANG_TO_LC


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', help='Corpus name ex: eng.rst.gum')
    parser.add_argument('--section', default='all', choices=('train', 'dev', 'test', 'all'), help='Selection section')
    # TODO add 'spacy'
    parser.add_argument('--parser', default='all', choices=('stanfordnlp', 'corenlp', 'all'), help='Selected parser')
    parser.add_argument('--processing', default='ssplit', choices=('ssplit', 'parsed'), help='Desired processing')
    parser.add_argument('--out_dir', default='.', help='Base dir for parser outputs')
    args = parser.parse_args()
    # locate corpus files
    corpus = args.corpus
    corpus_dir = os.path.join(DATA_DIR, corpus)
    if not os.path.isdir(corpus_dir):
        raise ValueError("Incorrect path to corpus: {}".format(corpus_dir))
    sections = (('train', 'dev', 'test') if args.section == 'all'
                else (args.section,))
    fp_toks = [os.path.join(corpus_dir, '{corpus}_{section}.tok'.format(corpus=corpus, section=section)) for section in sections]
    for fp_tok in fp_toks:
        if not os.path.exists(fp_tok):
            raise ValueError("Incorrect path to corpus file: {}".format(fp_tok))
    # TODO create folder for parses by date or version (+ store parser parameters)
    out_dir = os.path.join(args.out_dir, args.processing)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # setup and call each parser in turn
    if args.parser in ('stanfordnlp', 'all'):
        # TODO add version number? date?
        pdir = os.path.join(out_dir, 'stanfordnlp', corpus)
        if not os.path.isdir(pdir):
            os.makedirs(pdir)
        # specify the model that should be used
        treebank = None
        lang_long = corpus.split('.', 1)[0]
        lang = LANG_TO_LC[lang_long]
        if corpus == 'eng.rst.gum':
            treebank = 'en_gum'
        if args.processing == 'ssplit':
            ssplit_stanfordnlp(lang, fp_toks, pdir, treebank=treebank)
        else:
            parse_stanfordnlp(lang, fp_toks, pdir, treebank=treebank)
    #
    if args.parser in ('corenlp', 'all'):
        # TODO add version number? date?
        pdir = os.path.join(out_dir, 'corenlp', corpus)
        if not os.path.isdir(pdir):
            os.makedirs(pdir)
        for fp_tok in fp_toks:
            # parse_corenlp(fp_tok)
            pass
    #
    # evaluate parser outputs wrt the provided .conll files
    
