"""Read CONLL-U format.


"""


def _orig_sent_texts(f):
    """Read the original text of each sentence in a doc.

    Parameters
    ----------
    f : File
        Open text file.

    Yields
    ------
    doc_id : str
        Id of the document.
    doc_sents : List[str]
        List of sentence text for the doc.
    """
    doc_id = None
    doc_sents = []
    for line in f:
        if line.startswith('# newdoc'):
            if doc_sents:
                yield (doc_id, doc_sents)
            doc_id = line.split('id = ')[1].strip()
            doc_sents = []
        elif line.startswith('# sent_id = '):
            sent_id = line.split(' = ')[1].strip()
        elif line.startswith('# text = '):
            sent_text = line.split(' = ', 1)[1].strip()
            doc_sents.append(sent_text)
        elif line.strip() == '':
            continue
        else:
            continue
    else:
        yield (doc_id, doc_sents)


def _approx_sent_texts(f):
    """Approximate the original text of each sentence in a doc.

    Parameters
    ----------
    f : File
        Open text file.

    Yields
    ------
    doc_id : str
        Id of the document.
    doc_sents : List[str]
        List of sentence text for the doc.
    """
    doc_id = None
    doc_sents = []
    sent_toks = []
    for line in f:
        if line.startswith('# newdoc id = '):
            if doc_sents:
                yield (doc_id, doc_sents)
            doc_id = line.split(' = ')[1].strip()
            doc_sents = []
        elif line.startswith('# sent_id = '):
            sent_id = line.split(' = ')[1].strip()
        elif line.startswith('# text = '):
            # sent_text = line.split(' = ')[1].strip()
            continue
        elif line.strip() == '':
            # push previous sentence, start a new one
            if sent_toks:
                sent_text = ' '.join(sent_toks)
                doc_sents.append(sent_text)
            sent_toks = []
        else:
            # push token
            tok = line.split('\t')[1]
            sent_toks.append(tok)
    else:
        # append last sentence to last doc, yield
        sent_text = ' '.join(sent_toks)
        doc_sents.append(sent_text)
        yield (doc_id, doc_sents)


def conll_texts(conll_filename, use_sent_text=True):
    """Retrieve the text of docs from a CONLL-U file.

    The original text is not included, so we rebuild an
    approximation of it.

    Parameters
    ----------
    tok_filename : str
        Filename of the .tok file
    use_sent_text : boolean, defaults to True
        Use the original text of sentences if available
        (comment lines of the form "# text = ").

    Yields
    ------
    doc_id : str
        Id of the document.
    doc_text : str
        Text of the document.
    """
    with open(conll_filename) as f:
        if use_sent_text:
            # read the original text for each sentence off
            # the file
            docs_sents = _orig_sent_texts(f)
        else:
            # approximate the original text
            docs_sents = _approx_sent_texts(f)
        for doc_id, doc_sents in docs_sents:
            doc_text = ' '.join(doc_sents)
            yield doc_id, doc_text


def begin_toks_sents(conll_str):
    """Get beginning positions of tokens and sentences as offsets on the non-whitespace characters of a document text.

    Parameters
    ----------
    conll_str : str
        CONLL-U string for the file.

    Yields
    ------
    doc_id : str
        Document id.
    doc_chars : str
        Document text excluding whitespaces.
    tok_begs : List[int]
        Beginning position of each token in the doc.
        Correspond to indices in doc_chars.
    sent_begs : List[int]
        Beginning position of each sentence in the doc.
        Correspond to indices in doc_chars.
    """
    doc_id = None
    doc_chars = ''
    tok_begs = []
    sent_begs = []
    in_sent = False
    cur_idx = 0  # current (non-whitespace) character index
    for line in conll_str.split('\n'):
        if line.startswith('# newdoc id = '):
            if sent_begs:
                # yield previous doc
                yield (doc_id, doc_chars, tok_begs, sent_begs)
            # reset for a new doc
            doc_id = line.split('# newdoc id = ')[1]
            doc_chars = ''
            tok_begs = []
            sent_begs = []
            in_sent = False
            cur_idx = 0
        elif line.startswith('#'):
            continue
        elif line == '':
            # an empty line marks doc or sentence split
            in_sent = False
        else:
            # token line
            tok_begs.append(cur_idx)
            if not in_sent:
                # first token in sentence
                sent_begs.append(cur_idx)
                in_sent = True
            fields = line.split('\t')
            assert len(fields) == 10
            # delete whitespaces internal to the token
            tok_chars = fields[1].replace(' ', '').replace('\xa0', '')
            cur_idx += len(tok_chars)
            doc_chars += tok_chars
    else:
        # yield last document
        if sent_begs:
            yield (doc_id, doc_chars, tok_begs, sent_begs)
