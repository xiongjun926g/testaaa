"""Parse using Stanford CoreNLP via StanfordNLP.

"""

from stanfordnlp.server import CoreNLPClient

from task_data import rebuild_text
from tok_format import tok_tokens


CORENLP_PROPS = {
    'en_gum': {
        # https://stanfordnlp.github.io/CoreNLP/tokenize.html
        'tokenize.language': 'English',  # English PTBTokenizer
        'tokenize.options': ','.join([
            'normalizeAmpersandEntity=false',
            'normalizeCurrency=false',
            'normalizeFractions=false',
            'normalizeParentheses=false',
            'normalizeOtherBrackets=false',
            'latexQuotes=false',
            'ptb3Ellipsis=false',
            'ptb3Dashes=false',
            'strictTreebank3=true',
        ])
    },
}


def parse_stanford_corenlp(client, doc_id, doc_text, f_out):
    """Parse a doc using Stanford CoreNLP.

    Parameters
    ----------
    client : CoreNLPClient
        CoreNLP client.
    doc_id : str
        Doc id.
    doc_text : str
        Document text.
    f_out : File
        Output file name.
    """
    print('#newdoc id = ' + doc_id, file=f_out)
    ann = client.annotate(doc_text)
    print(ann, file=f_out, end='')


if __name__ == '__main__':
    # setup CoreNLP
    # don't include 1st annotator 'tokenize'
    # memory='16G'?
    # TODO: use the adequate properties file (cf. language code)
    # annotators = 'ssplit pos lemma ner depparse coref'.split()
    annotators = 'ssplit pos lemma depparse'.split()
    # TODO delete extra period from out_core.conll OR skip the sequence of replace() to keep tokens separated by whitespaces and add property 'tokenize.whitespace=true'

    with CoreNLPClient(annotators=annotators, timeout=30000, memory='8G', properties=CORENLP_PROPS['en_gum'], output_format='conllu', be_quiet=True) as client:
        # submit the request to the server
        with open(fn_core, mode='w') as f_core:
            for doc_id, doc_toks in tok_tokens(fn_tok):
                doc_text = rebuild_text(doc_toks)
                parse_stanford_corenlp(client, doc_id, doc_text, f_core)
    # undo transformation of double quotes in pair of simple quotes
    with open(fn_core_fix, mode='w') as f_core_fix:
        with open(fn_core) as f_core:
            txt_core = f_core.read()
            if "''" not in txt_core:
                raise ValueError("No pair of single quotes!")
            # TODO check the effect this line has
            txt_core = txt_core.replace("''", '"')
            # undo weird character substitution by CoreNLP !?
            txt_core = txt_core.replace("째", "°")
            if True:
                # eng.rst.gum_train only ! FIXME
                txt_core_fix = ''
                for i, line in enumerate(txt_core.split('\n')):
                    if i == 56313:  # DIRTY
                        # skip extra period added after "U.S."
                        continue
                    txt_core_fix += line + '\n'
                # drop extra final newline (yark)
                txt_core_fix = txt_core_fix[:-1]
                print(txt_core_fix, file=f_core_fix, end='')
            else:
                print(txt_core, file=f_core_fix, end='')

