import argparse, json, os, joblib, sys

# TODO need to save the tag_to_ix

def main( ):
    parser = argparse.ArgumentParser(
            description='Evaluation:write prediction files')

    parser.add_argument('--fjson',
            dest='fjson',
            action='store',
            help='Expe trace file')
    parser.add_argument('--outpath',
            dest='outpath',
            action='store',
            help='Outdir')
    parser.add_argument('--epoch',
            dest='epoch',
            action='store',
            default="9",
            help='Num epoch to evaluate')
    args = parser.parse_args()

    if not os.path.isdir( args.outpath ):
        os.mkdir(args.outpath)

    write_pred_dev( args.fjson, args.outpath, num_epoch=int(args.epoch) )

    write_pred_test( args.fjson, args.outpath, num_epoch=int(args.epoch) )

def read_pred( fjson ):
    # json file gives the setting and the path to the model and pred file
    dict_ex=json.load(open(fjson))
    #(data_inpath, data_name, data_vocab_size, data_tagset_size) = dict_ex["data"]
    #(train_acc, train_p, train_r, train_f1, train_s) = dict_ex["train_scores"](train_acc, train_p, train_r, train_f1, train_s) = 
    #(dev_acc, dev_p, dev_r, dev_f1, dev_s) = dict_ex["dev_scores"]
    #float( train_loss ) = dict_ex["train_loss"]
    #num_epoch = dict_ex["epoch"]
    #dev_pred_file = dict_ex["dev_y_pred_file"]
    #model_file = dict_ex["model_state_dict"]
    #config_file = dict_ex["config_file"] = model.config_file
    return dict_ex

def write_pred_dev( fjson, outpath, num_epoch=9 ):
    dict_ex = read_pred( fjson )
    id_ex = find_expe( dict_ex, num_epoch)
    #print( dict_ex.keys())
    #print( dict_ex['Experiment'].keys())
    if id_ex == None:
        sys.exit("Expe not found for epoch", num_epoch, "in file", fjson )

    (data_inpath, data_name, data_vocab_size, data_tagset_size) = dict_ex["Experiment"][id_ex]["data"]
    if "dev_fnames" in dict_ex["Experiment"][id_ex]: # Multilingual experiment
        #print("MULTILING")
        for data_name in dict_ex["Experiment"][id_ex]["dev_fnames"]:
            dev_file = os.path.join( data_inpath, data_name, data_name+'_dev.tok' )
            pred_file = os.path.join( outpath, data_name+'_dev.scores' )
            preds = joblib.load( dict_ex["Experiment"][id_ex]["dev_y_pred_file_"+data_name] )
            if 'pdtb' in fjson:
                write_pred_pdtb( dev_file, pred_file, preds )
            else:
                write_pred( dev_file, pred_file, preds )

    else:
        dev_file = os.path.join( data_inpath, data_name, data_name+'_dev.tok' )
        pred_file = os.path.join( outpath, data_name+'_dev.scores' )
        preds = joblib.load( dict_ex["Experiment"][id_ex]["dev_y_pred_file"] )
        if 'pdtb' in fjson:
            write_pred_pdtb( dev_file, pred_file, preds )
        else:
            write_pred( dev_file, pred_file, preds )

def write_pred_test( fjson, outpath, num_epoch=9 ):
    dict_ex = read_pred( fjson )
    id_ex = find_expe( dict_ex, num_epoch)
    #print( dict_ex.keys())
    #print( dict_ex['Experiment'].keys())
    if id_ex == None:
        sys.exit("Expe not found for epoch", num_epoch, "in file", fjson )

    (data_inpath, data_name, data_vocab_size, data_tagset_size) = dict_ex["Experiment"][id_ex]["data"]
    if "test_fnames" in dict_ex["Experiment"][id_ex]: # Multilingual experiment
        #print("MULTILING")
        for data_name in dict_ex["Experiment"][id_ex]["test_fnames"]:
            test_file = os.path.join( data_inpath, data_name, data_name+'_test.tok' )
            pred_file = os.path.join( outpath, data_name+'_test.scores' )
            preds = joblib.load( dict_ex["Experiment"][id_ex]["test_y_pred_file_"+data_name] )
            if 'pdtb' in fjson:
                write_pred_pdtb( test_file, pred_file, preds )
            else:
                write_pred( test_file, pred_file, preds )

    else:
        test_file = os.path.join( data_inpath, data_name, data_name+'_test.tok' )
        pred_file = os.path.join( outpath, data_name+'_test.scores' )
        preds = joblib.load( dict_ex["Experiment"][id_ex]["test_y_pred_file"] )
        if 'pdtb' in fjson:
            write_pred_pdtb( test_file, pred_file, preds )
        else:
            write_pred( test_file, pred_file, preds )

def find_expe( dict_ex, num_epoch):
    for e in dict_ex["Experiment"]:
        if dict_ex["Experiment"][e]["epoch"] == num_epoch:
            return e
    return None

# TODO rewrite
def write_pred( infile, outfile, preds ):
    lines = open(infile).readlines()
    # TODO what s the elegant way of doing this? just tolist()? flatten? squeeze?
    #https://stackoverflow.com/questions/53903373/convert-pytorch-tensor-to-python-list
    doc_id = -1
    with open( outfile, 'w' ) as o:
        line_count = 0
        for line in lines: 
            line = line.strip()
            if line.startswith("#"):
                ##print( "LINE #:", line, doc_id, line_count )
                doc_id += 1
                ##print( "Writing preds for doc:", doc_id)
                o.write( line.strip()+'\n')
            elif line.strip()!="":
                tag = preds[line_count]
                stag = 'BeginSeg=Yes' if tag == 1 else '_'
                ##print( "LINE:", line, doc_id, line_count, tag, stag)
                o.write( '\t'.join( line.strip().split('\t')[:-1] )+'\t'+stag+'\n' )
                line_count += 1
            else:
                ##print( "LINE EMPTY:", line, doc_id, line_count)
                o.write( line.strip()+'\n')

def write_pred_pdtb( infile, outfile, preds ):
    tag_to_ix = {'_':0, 'Seg=B-Conn':1, 'Seg=I-Conn':2}
    ix_to_tag = { v:k for (k,v) in tag_to_ix.items()}
    lines = open(infile).readlines()
    # TODO what s the elegant way of doing this? just tolist()? flatten? squeeze?
    #https://stackoverflow.com/questions/53903373/convert-pytorch-tensor-to-python-list
    doc_id = -1
    with open( outfile, 'w' ) as o:
        line_count = 0
        for line in lines: 
            line = line.strip()
            if line.startswith("#"):
                ##print( "LINE #:", line, doc_id, line_count )
                doc_id += 1
                ##print( "Writing preds for doc:", doc_id)
                o.write( line.strip()+'\n')
            elif line.strip()!="":
                tag = preds[line_count]
                stag = ix_to_tag[int(tag)]
                ##print( "LINE:", line, doc_id, line_count, tag, stag)
                o.write( '\t'.join( line.strip().split('\t')[:-1] )+'\t'+stag+'\n' )
                line_count += 1
            else:
                ##print( "LINE EMPTY:", line, doc_id, line_count)
                o.write( line.strip()+'\n')

if __name__ == '__main__':
    main()