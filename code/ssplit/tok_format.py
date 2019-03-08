"""Read .tok format

Known issue: the Portuguese treebank has multitokens in the conll, whose underlying tokens are also included in the tok file (but with a regular index !).
ex: [3-4, 3, 4] in the conll becomes [3, 4, 5] in the tok
"""


def tok_texts(tok_filename):
    """Retrieve the text of docs from a .tok file.

    Parameters
    ----------
    tok_filename : str
        Filename of the .tok file

    Yields
    ------
    doc_text : str
        The text of a document.
    """
    for doc_id, doc_toks in tok_tokens(tok_filename):
        doc_text = ' '.join(doc_toks)
        yield doc_id, doc_text


def tok_tokens(tok_filename):
    """Retrieve the tokens of docs from a .tok file.

    Parameters
    ----------
    tok_filename : str
        Filename of the .tok file

    Yields
    ------
    doc_toks : List[str]
        A document is read as a list of tokens.
    """
    with open(tok_filename) as f:
        doc_id = None
        doc_toks = []
        for line in f:
            if line.startswith('# newdoc'):
                if doc_toks:
                    yield (doc_id, doc_toks)
                doc_id = line.split('id = ')[1].strip()
                doc_toks = []
            elif line.strip() == '':
                continue
            else:
                tok = line.split('\t')[1]
                doc_toks.append(tok)
        else:
            # yield last doc
            yield (doc_id, doc_toks)


def tok_tokens_labels(tok_filename):
    """Retrieve the list of tokens and (target) labels for each doc in a .tok file.

    Parameters
    ----------
    tok_filename : str
        Filename of the .tok file

    Yields
    ------
    doc_toks : List[str]
        List of tokens in the document.
    doc_lbls : List[str]
        List of labels in the document (same length as doc_toks).
    """
    with open(tok_filename) as f:
        doc_id = None
        doc_toks = []
        doc_lbls = []
        for line in f:
            if line.startswith('# newdoc'):
                if doc_toks:
                    yield (doc_id, doc_toks, doc_lbls)
                doc_id = line.split('id = ')[1].strip()
                doc_toks = []
                doc_lbls = []
            elif line.strip() == '':
                continue
            else:
                fields = line.strip().split('\t')
                tok = fields[1]
                lbl = fields[9]
                doc_toks.append(tok)
                doc_lbls.append(lbl)
        else:
            # yield last doc
            yield (doc_id, doc_toks, doc_lbls)
