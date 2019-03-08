"""Parse document text using StanfordNLP.

TODO:
- [ ] compare parser prediction on en_gum with the standard 'en' model vs 'en_gum'
"""

import os.path

import numpy as np
import stanfordnlp

from conll_format import conll_texts
from tok_format import tok_texts, tok_tokens, tok_tokens_labels
from conll_format import begin_toks_sents
from task_data import DATA_DIR, open_and_fix_udv2, open_and_fix_udv2_eng_rst_gum_train, rebuild_text


def parse_stanford_neural(nlp, doc_id, doc_text, f_out):
    """Parse a doc using Stanford's fully neural pipeline from the CoNLL 2018 Shared Task.

    Parameters
    ----------
    nlp : Pipeline
        StanfordNLP Neural Pipeline.
    doc_id : str
        Doc id.
    doc_text : str
        Document text.
    f_out : File
        Output file.
    """
    # output CONLL-U file
    print('#newdoc id = ' + doc_id, file=f_out)
    ann = nlp(doc_text)
    # ann.write_conll_to_file(filename)
    return_str = ann.conll_file.conll_as_string()
    print(return_str, file=f_out, end='')


def parse_stanfordnlp(lang, fp_toks, out_dir, resource_dir=None, confirm_if_exists=True, treebank=None):
    """Parse files from a corpus with StanfordNLP.

    Parameters
    ----------
    lang_name : str
        Language name for StanfordNLP.
    fp_toks : List[str]
        File paths to the .tok files to be parsed.
    """
    # download model if necessary
    shorthand = (treebank if treebank is not None else lang)
    stanfordnlp.download(shorthand, resource_dir=resource_dir, confirm_if_exists=confirm_if_exists)  # not auto-cached ?    # ssplit .tok file
    # setup StanfordNLP neural pipeline
    processors = 'tokenize,mwt,pos,lemma,depparse'
    if resource_dir is None:
        # stanfordnlp.Pipeline will use its DEFAULT_MODEL_DIR
        nlp = stanfordnlp.Pipeline(processors=processors, lang=lang, treebank=treebank, use_gpu=False)
    else:
        nlp = stanfordnlp.Pipeline(processors=processors, lang=lang, models_dir=resource_dir, treebank=treebank, use_gpu=False)
    # parse each file in turn
    for fp_tok in fp_toks:
        fp_out = os.path.join(out_dir, os.path.basename(fp_tok))
        with open(fp_out, mode='w') as f_out:
            # parse each doc in turn
            for doc_id, doc_toks in tok_tokens(fp_tok):
                doc_text = rebuild_text(doc_toks)
                parse_stanford_neural(nlp, doc_id, doc_text, f_out)


def ssplit_stanfordnlp(lang, fp_toks, out_dir, resource_dir=None, confirm_if_exists=True, treebank=None):
    """Sentence split files from a corpus with StanfordNLP.

    Parameters
    ----------
    lang_name : str
        Language name for StanfordNLP.
    fp_toks : List[str]
        File paths to the .tok files to be parsed.
    out_dir : str
        Output folder.
    resource_dir : str
        Folder containing resources for StanfordNLP, passed to
        stanfordnlp.download().
    confirm_if_exists : boolean
        Passed to stanfordnlp.download().
    treebank : str
        Treebank name for when StanfordNLP has a specific model.
    """
    # download model if necessary
    shorthand = (treebank if treebank is not None else lang)
    stanfordnlp.download(shorthand, resource_dir=resource_dir, confirm_if_exists=confirm_if_exists)  # not auto-cached ?    # ssplit .tok file
    # setup StanfordNLP neural pipeline
    processors = 'tokenize'
    if resource_dir is None:
        # stanfordnlp.Pipeline will use its DEFAULT_MODEL_DIR
        nlp = stanfordnlp.Pipeline(processors=processors, lang=lang, treebank=treebank, use_gpu=False)
    else:
        nlp = stanfordnlp.Pipeline(processors=processors, lang=lang, models_dir=resource_dir, treebank=treebank, use_gpu=False)
    # parse each file in turn
    for fp_tok in fp_toks:
        # FIXME do both in one go
        # for each doc, get the list of tokens and labels
        tok_tok_lbls = [(doc_id, doc_toks, doc_lbls) for doc_id, doc_toks, doc_lbls in tok_tokens_labels(fp_tok)]
        # for each doc, get the character offset of tokens
        with open(fp_tok) as f_tok:
            tok_str = f_tok.read()
        tok_tok_begs = [(doc_id, doc_chars, tok_begs) for doc_id, doc_chars, tok_begs, _ in begin_toks_sents(tok_str)]
        #
        # parse
        fp_out = os.path.join(out_dir, os.path.basename(fp_tok))
        with open(fp_out, mode='w') as f_out:
            # parse each doc in turn
            for (doc_id, doc_toks, doc_lbls), (_, doc_chars, tok_begs) in zip(tok_tok_lbls, tok_tok_begs):
                doc_text = rebuild_text(doc_toks, lang=lang)
                ann = nlp(doc_text)
                conll_str = ann.conll_file.conll_as_string()
                conll_tok_begs = list(begin_toks_sents(conll_str))
                # we parse one doc at a time
                assert len(conll_tok_begs) == 1
                _, p_doc_chars, p_tok_begs, p_sent_begs = conll_tok_begs[0]
                try:
                    assert p_doc_chars == doc_chars
                except AssertionError:
                    for i, (pdc, dc) in enumerate(zip(p_doc_chars, doc_chars)):
                        if pdc != dc:
                            print(fp_tok, i, p_doc_chars[i - 10:i + 10], doc_chars[i - 10:i + 10])
                            raise
                # for each beginning of sentence (in the parser output), find the corresponding token index in the original .tok
                sent_beg_idc = np.searchsorted(tok_begs, p_sent_begs, side='left')
                sent_beg_idc = set(sent_beg_idc)
                # output CONLL-U file
                print('# newdoc id = ' + doc_id, file=f_out)
                tok_sent_idx = 1
                for tok_doc_idx, (tok, lbl) in enumerate(zip(doc_toks, doc_lbls), start=0):
                    if tok_doc_idx in sent_beg_idc:
                        if tok_doc_idx > 0:
                            # add an empty line after the previous sentence (not for the first token in doc)
                            print('', file=f_out)
                        tok_sent_idx = 1
                    else:
                        tok_sent_idx += 1
                    row = (str(tok_sent_idx), tok, '_', '_', '_', '_', '_', '_', '_', lbl)
                    print('\t'.join(row), file=f_out)
                print('', file=f_out)


if __name__ == '__main__':
    fn_tok = os.path.join(DATA_DIR, corpus_name, corpus_name + '_' + subset + '.tok')
    fn_neur = '_'.join(('out', 'neur', lang, treebank)) + '.conll'
    fn_core = '_'.join(('out', 'core')) + '.conll'
    fn_core_fix = '_'.join(('out', 'core', 'fix')) + '.conll'
    # setup StanfordNLP neural pipeline
 
    # evaluate
    fn_conll = os.path.join(DATA_DIR, corpus_name, corpus_name + '_' + subset + '.conll')
    if False:
        print('CoreNLP')
        eval_sbd(fn_conll, fn_core_fix)
    print('StanfordNLP')
    eval_sbd(fn_conll, fn_neur)
    raise ValueError("stop me now")
    # NEXT
    # WIP load tokenized text straight from the .tok and .conll files
    from stanfordnlp.models.common.conll import CoNLLFile
    input_conll = CoNLLFile(fn_conll)
    input_conll.load_all()
    print(input_conll.sents[0])
    input_tok = CoNLLFile(fn_tok)
    input_tok.load_all()
    print(input_tok.sents[0])
    raise ValueError("gne")

    # parse all corpora
    for corpus_name, corpus_props in CORPORA.items():
        # language, (discourse) formalism, corpus name
        lang, flm, cname = corpus_name.split('.')
        lc = LANG_TO_LC[lang]
        # StanfordNLP
        stanfordnlp.download(lc)  # auto-cached ?
        # TODO exclude the following processors: 'tokenize,mwt'
        # use_gpu=True makes CUDA run out of memory on my laptop
        nlp = stanfordnlp.Pipeline(lang=lc, processors='tokenize,mwt,pos,lemma,depparse', use_gpu=False)
        # loop over corpora
        for subset in ('train', 'dev'):
            fn_conll = os.path.join(DATA_DIR, corpus_name, corpus_name + '_' + subset + '.conll')
            docs_conll = conll_texts(fn_conll, use_sent_text=corpus_props.sent_text)
            if corpus_props.underscored:
                # skip because text is missing
                # TODO restore the text (use the script provided)
                print(fn_conll + '\tskip (underscored)')
                continue
            # loop over docs
            for doc_id, doc_text in docs_conll:
                doc = nlp(doc_text)
                print(doc_text)
                doc.sentences[0].print_dependencies()
                raise ValueError("check me")
