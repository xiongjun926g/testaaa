"""Information on the corpora for the DISRPT shared task.

TODO:
- [ ] check if the BOM characters <U+FEFF> present in the tok and conll files of 4 treebanks, have an impact on the parsing results. They might need to be removed (and restored for compatibility?). ( grep -RI $'\xEF' data/ |cut -d':' -f1 |sort |uniq -c )
"""

from collections import namedtuple
import io
import os.path


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data'))

Corpus = namedtuple('Corpus', ['sent_text', 'space_after', 'underscored', 'has_multitoks'])

CORPORA = {
    'deu.rst.pcc': Corpus(sent_text=False, space_after=False, underscored=False, has_multitoks=False),
    'eng.pdtb.pdtb': Corpus(sent_text=False, space_after=False, underscored=True, has_multitoks=False),
    'eng.rst.gum': Corpus(sent_text=True, space_after=True, underscored=False, has_multitoks=False),
    'eng.rst.rstdt': Corpus(sent_text=True, space_after=True, underscored=True, has_multitoks=False),
    'eng.sdrt.stac': Corpus(sent_text=False, space_after=False, underscored=False, has_multitoks=False),
    'eus.rst.ert': Corpus(sent_text=False, space_after=False, underscored=False, has_multitoks=False),
    'fra.sdrt.annodis': Corpus(sent_text=False, space_after=False, underscored=False, has_multitoks=False),
    'nld.rst.nldt': Corpus(sent_text=True, space_after=False, underscored=False, has_multitoks=False),
    'por.rst.cstn': Corpus(sent_text=True, space_after=True, underscored=False, has_multitoks=True),
    'rus.rst.rrt': Corpus(sent_text=False, space_after=False, underscored=False, has_multitoks=False),
    'spa.rst.rststb': Corpus(sent_text=False, space_after=False, underscored=False, has_multitoks=False),
    'spa.rst.sctb': Corpus(sent_text=False, space_after=False, underscored=False, has_multitoks=False),
    'zho.pdtb.cdtb': Corpus(sent_text=False, space_after=False, underscored=True, has_multitoks=False),
    'zho.rst.sctb': Corpus(sent_text=False, space_after=False, underscored=False, has_multitoks=False),
}


# map language code from DISRPT to language code (ISO 639-1)
LANG_TO_LC = {
    'deu': 'de',
    'eng': 'en',
    'eus': 'eu',
    'fra': 'fr',
    'nld': 'nl',
    'por': 'pt',
    'rus': 'ru',
    'spa': 'es',
    'tur': 'tr',
    'zho': 'zh',
}

# map corpus name in DISRPT to treebank code (StanfordNLP)
CORPUS_TO_TB = {
    'eng.rst.gum': 'en_gum',
}


def open_and_fix_udv2(fn_conll):
    """Open CONLL-U file and fix it for well-formedness in UD v2.

    This generic function merely appends to the CONLL-U file an empty newline that is expected by conll18_ud_eval.load_conllu().

    Parameters
    ----------
    fn_conll : str
        Path to the CONLL-U file.

    Returns
    -------
    f_udv2 : File
        CoNLL-U file with UD v2.
    """
    f_udv2 = io.StringIO()
    with open(fn_conll, encoding='utf-8') as f:
        f_udv2.write(f.read() + '\n')
    f_udv2.seek(0)
    return f_udv2


def open_and_fix_udv2_eng_rst_gum_train(fn_conll):
    """Open CONLL-U file for the training section of the eng.rst.gum treebank and fix it for well-formedness in UD v2.

    Parameters
    ----------
    fn_conll : str
        Path to the CONLL-U file.

    Returns
    -------
    f_udv2 : File
        CoNLL-U file with UD v2.
    """
    f_udv2 = io.StringIO()
    with open(fn_conll) as f:
        for i, line in enumerate(f):
            if line.startswith('#') or line.strip() == '':
                print(line, file=f_udv2, end='')
            else:
                fields = line.strip().split('\t')
                if i == 19891:
                    fields[6] = '12'
                    fields[7] = 'punct'
                    fields[8] = '12:punct'
                elif i == 19892:
                    fields[6] = '4'
                    fields[7] = 'discourse'
                    fields[8] = '4:discourse'
                elif i == 19893:
                    fields[6] = '4'
                    fields[7] = 'punct'
                    fields[8] = '4:punct'
                elif i == 19898:
                    fields[6] = '19'
                    fields[7] = 'punct'
                    fields[8] = '19:punct'
                elif i == 19899:
                    fields[6] = '19'
                    fields[7] = 'discourse'
                    fields[8] = '19:discourse'
                elif i == 19900:
                    fields[6] = '19'
                    fields[7] = 'punct'
                    fields[8] = '19:punct'
                print('\t'.join(fields), file=f_udv2, end='\n')
    # add an empty line
    print('', file=f_udv2, end='\n')
    # reset buffer position to the beginning, so f_udv2.read() returns something
    f_udv2.seek(0)
    return f_udv2


def rebuild_text(doc_toks, lang=None):
    """Rebuild the underlying text from a list of tokens.

    We don't assume any additional information.
    In particular, the "SpaceAfter=No" provided in some CONLL-U files is ignored.

    Parameters
    ----------
    doc_toks : List[str]
        List of tokens in the document.
    lang : str
        Language ; If None, the language is assumed to be one where tokens are
        separated with whitespaces. Currently the only interesting value is "zh"
        with no whitespace.
    """
    if lang == "zh":
        return ''.join(doc_toks)
    # default: insert whitespaces between tokens then remove extraneous ones ;
    # this heuristic is crude but a reasonable default
    doc_text = ' '.join(doc_toks)
    doc_text = (doc_text.replace(' : " ', ': "')
                .replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' :', ':')
                .replace('“ ', '“').replace(' ”', '”')
                .replace(' ;', ';')
                .replace(' ’', '’')
                .replace('( ', '(').replace(' )', ')')
                .replace('[ ', '[').replace(' ]', ']')
    )
    return doc_text
