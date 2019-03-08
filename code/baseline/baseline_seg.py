"""Baseline segmenter"""

from glob import glob
import os
import os.path
import subprocess


def sbd_seg(f_in, f_out):
    """Segment on sentence boundaries.

    Parameters
    ----------
    f_in : File
        Sentence segmented (ssplit) input file.
    f_out : File
        Discourse segmented output file.
    """
    in_sent = False
    for line in f_in:
        if line.startswith('#') or line.strip() == '':
            print(line, file=f_out, end='')
            in_sent = False
        else:
            fields = line.strip().split('\t')
            assert len(fields) == 10
            if not in_sent:
                # first token of a sentence ;
                # cannot test on fields[0] == '1' because
                # some sentences in the shared task data
                # start at token 2
                in_sent = True
                if fields[9] == '_':
                    fields[9] = 'BeginSeg=Yes'
                else:
                    fields[9] = 'BeginSeg=Yes|' + fields[9]
                print('\t'.join(fields), file=f_out)
            else:
                print(line, file=f_out, end='')


def clear_targets_beginseg(f_in, f_out):
    """Clear targets by removing BeginSeg=Yes.

    Suited for RST and SDRT files.

    Parameters
    ----------
    f_in : File
        Input file in one of the .tok or .conll formats.
    f_out : File
        Output file in the same format, all BeginSeg=Yes erased.
    """
    for line in f_in:
        if line.startswith('#') or line.strip() == '':
            print(line, file=f_out, end='')
        else:
            fields = line.strip().split('\t')
            assert len(fields) == 10
            # some files have (had) two BeginSeg=Yes for the
            # same token
            fields[9] = fields[9].replace('BeginSeg=Yes|', '').replace('BeginSeg=Yes', '')
            if fields[9] == '':
                fields[9] = '_'
            print('\t'.join(fields), file=f_out)


if __name__ == "__main__":
    # prepare input files: drop target values from both
    # .conll and sentence-split .tok
    ssplit_fps = glob(os.path.join('ssplit', 'stanfordnlp', '*', '*.tok'))
    conll_fps = glob(os.path.join('data', '*', '*.conll'))
    sent_fps = ssplit_fps + conll_fps
    #
    in_dir = 'baseline_in'
    if not os.path.isdir(in_dir):
        os.makedirs(in_dir)
    for sent_fp in sent_fps:
        with open(sent_fp) as f_sent:
            fn_in = os.path.basename(sent_fp)
            if fn_in.split('.')[1] == 'pdtb':
                continue
            cname = fn_in.rsplit('.', 1)[0].rsplit('_', 1)[0]
            in_cdir = os.path.join(in_dir, cname)
            if not os.path.isdir(in_cdir):
                os.makedirs(in_cdir)
            fp_in = os.path.join(in_cdir, fn_in)
            with open(fp_in, mode='w') as f_in:
                clear_targets_beginseg(f_sent, f_in)
    in_fps = (glob(os.path.join(in_dir, '*', '*.tok')) +
              glob(os.path.join(in_dir, '*', '*.conll')))
    # perform baseline segmentation (excluding PDTB corpora)
    out_dir = 'baseline_results'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    #
    for in_fp in in_fps:
        with open(in_fp) as f_in:
            fn_out = os.path.basename(in_fp)
            if fn_out.split('.')[1] == 'pdtb':
                # schema is more complex for connectives
                # (B-Conn, I-Conn) and mostly not
                # sentence-initial
                continue
            cname = fn_out.rsplit('.', 1)[0].rsplit('_', 1)[0]
            out_cdir = os.path.join(out_dir, cname)
            if not os.path.isdir(out_cdir):
                os.makedirs(out_cdir)
            fp_out = os.path.join(out_cdir, fn_out)
            with open(fp_out, mode='w') as f_out:
                sbd_seg(f_in, f_out)
    # evaluate and write scores to file
    pred_fps = (glob(os.path.join(out_dir, '*', '*.tok')) +
                glob(os.path.join(out_dir, '*', '*.conll')))
    for pred_fp in pred_fps:
        fname = os.path.basename(pred_fp)
        cname = fname.rsplit('.', 1)[0].rsplit('_', 1)[0]
        true_fp = os.path.join('data', cname, fname)
        if not os.path.exists(true_fp):
            raise ValueError("Unable to find reference file {}".format(true_fp))
        # store the score alongside the predictions
        score_fp = os.path.join(os.path.dirname(pred_fp), fname + '.scores')
        with open(score_fp, mode='w') as f_score:
            subprocess.call(['python', 'utils/seg_eval.py', true_fp, pred_fp], stdout=f_score)

