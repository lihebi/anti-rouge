"""interface:
ductac_get_docIds()
ductac_get_model_summaries(docId)
ductac_get_system_summaries(docId)
ductac_get_system_summary_scores(docId)

The input is only the path to DUC/TAC raw data folder. All temporary
files should be created under /tmp folder, with properly hashed
filename.

"""

import os
import pickle
import shutil

import numpy as np
import glob
import json
import math

from bs4 import BeautifulSoup

# from antirouge.utils import create_tokenizer_from_texts, save_tokenizer, load_tokenizer
# from antirouge.utils import sentence_split
# from antirouge.utils import dict_pickle_read, dict_pickle_read_keys, dict_pickle_write

import random
import tensorflow as tf


DATA_DIR = "/home/hebi/mnt/data/nlp/"
DUC_2002_DIR = os.path.join(DATA_DIR, 'DUC2002')
DUC_2001_DIR = os.path.join(DATA_DIR, 'DUC2001')
TAC_2009_DIR = os.path.join(DATA_DIR, 'TAC2009')


def parse_perdocs(perdocs):
    # xmlfile = "/home/hebi/mnt/data/nlp/DUC2002/data/test/summaries/summaries/d061jb/perdocs"
    res = {}
    with open(perdocs) as fp:
        soup = BeautifulSoup(fp)
        for node in soup.select('SUM'):
            docId = node.attrs['docref']
            selector = node.attrs['selector']
            summarizer = node.attrs['summarizer']
            size = node.attrs['size']
            content = node.string
            # FIXME will there be multiple model summeries for a single document?
            res[docId] = content
    return res

def parse_DUC_2002_model_summary():
    # /home/hebi/mnt/data/nlp/DUC2002/data/test/summaries/summaries/*/perdocs
    model_dir = os.path.join(DUC_2002_DIR, 'data/test/summaries/summaries')
    res = {}
    for d in os.listdir(model_dir):
        xmlfile = os.path.join(model_dir, d, 'perdocs')
        if os.path.exists(xmlfile):
            tmp = parse_perdocs(xmlfile)
            res.update(tmp)
    return res
    


def parse_DUC_2002_meta():
    """My objective is to get the article, summary, score.

    @return (docID, absID, score, doc_fname, abs_fname)
    
    1. from peer_dir, get all peer ID->fname
    2. from doc_dir, get all doc ID->fname
    3. from model_dir, get all model ID->fname
    4. read result file and parse accordingly
    """
    peer_dir = os.path.join(DUC_2002_DIR,
                            'results/abstracts/phase1/SEEpeers',
                            'SEE.abstracts.in.sentences')
    peer_baseline_dir = os.path.join(DUC_2002_DIR,
                                     'results/abstracts/phase1/SEEpeers',
                                     'SEE.baseline1.in.sentences')
    result_file = os.path.join(DUC_2002_DIR,
                               'results/abstracts/phase1/short.results.table')
    doc_dir = os.path.join(DUC_2002_DIR, 'data/test/docs.with.sentence.breaks')

    # XXX model summary is here:
    # /home/hebi/mnt/data/nlp/DUC2002/data/test/summaries/extracts_abstracts/d063jc
    # get the list of (docID, absID, score)
    res = []
    with open(result_file) as f:
        for line in f:
            if line.startswith('D'):
                splits = line.split()
                if splits[1] == 'P':
                    docsetID = splits[0]
                    docID = splits[2]  # ***
                    length = splits[3]  # seems that all lengths are 100
                    selector = splits[5]
                    # summarizer = splits[6]
                    # assessor = splits[7]
                    absID = splits[8]  # ***
                    score = splits[27]     # ***
                    # docset.type.length.[selector].peer-summarizer.docref
                    fname = '%s.%s.%s.%s.%s.%s.html' % (docsetID, 'P',
                                                        length,
                                                        selector,
                                                        absID,
                                                        docID)
                    doc_fname = os.path.join(doc_dir,
                                             docsetID.lower()+selector.lower(),
                                             docID+'.S')
                    if absID == '1':
                        abs_fname = os.path.join(peer_baseline_dir, fname)
                    else:
                        abs_fname = os.path.join(peer_dir, fname)
                    if not os.path.exists(doc_fname):
                        print('File not found: ', doc_fname)
                    elif not os.path.exists(abs_fname):
                        print('File not found: ', abs_fname)
                    else:
                        # absID is augmented with docID
                        res.append((docID, absID, float(score),
                                    doc_fname, abs_fname))
    return res

# doc file example: /home/hebi/mnt/data/nlp/DUC2002/data/test/docs.with.sentence.breaks/d061j/AP880911-0016.S
# abs file example: /home/hebi/mnt/data/nlp/DUC2002/results/abstracts/phase1/SEEpeers/SEE.baseline1.in.sentences/D061.P.100.J.1.AP880911-0016.html
def _copy_duc_text_impl(in_fname, out_fname, doc_type):
    """Parse the content of in file and output to out file."""
    # TODO should I use a rigrious html parser instead of bs4?
    assert(doc_type in ['doc', 'abs'])
    # print('copying ', in_fname, 'into', out_fname)
    selector = 'TEXT s' if doc_type == 'doc' else 'body a[href]'
    with open(in_fname) as fin, open(out_fname, 'w') as fout:
        soup = BeautifulSoup(fin)
        for s in soup.select(selector):
            sent = s.get_text()
            fout.write(sent)
            fout.write('\n\n')

def copy_duc_text(duc_meta, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    total = len(duc_meta)
    ct = 0
    for d in duc_meta:
        ct+=1
        if ct % 100 == 0:
            print(ct, '/', total)
        docID = d[0]
        # absID = d[0] + '--' + d[1]
        absID = d[1]
        doc_fname = d[3]
        abs_fname = d[4]
        doc_out = os.path.join(out_dir, docID + '.txt')
        abs_out = os.path.join(out_dir, docID + '--' + absID + '.txt')
        if not os.path.exists(doc_out):
            _copy_duc_text_impl(doc_fname, doc_out, 'doc')
        if not os.path.exists(abs_out):
            _copy_duc_text_impl(abs_fname, abs_out, 'abs')

def process_ductac(which):
    if which == 'DUC_2002':
        outdir = os.path.join('/tmp', which)
        duc_meta = parse_DUC_2002_meta()
        # copy text to folder
        text_dir = os.path.join(outdir, 'text')
        copy_duc_text(duc_meta, text_dir)
        duc_meta_simple = [d[:3] for d in duc_meta]
        doc2model = parse_DUC_2002_model_summary()
        for d in duc_meta_simple:
            docId = d[0]
            assert docId in doc2model
            outfile = os.path.join(text_dir, docId + '--X.txt')
            if not os.path.exists(outfile):
                with open(outfile, 'w') as fp:
                    fp.write(doc2model[docId])
                    fp.write('\n\n')
        with open(os.path.join(outdir, 'meta.json'),'w') as fp:
            json.dump(duc_meta_simple, fp, indent=4)
    elif which == 'DUC_2001':
        pass
    elif which == 'TAC_2009':
        pass
        

def ductac_get_docIds(which):
    "Return doc Ids for the dataset."
    meta = load_meta(which)
    doc_ids = list(set([d[0] for d in meta]))
    return doc_ids

def ductac_get_systemIds(which):
    "Return system Ids."
    meta = load_meta(which)
    res = list(set([d[1] for d in meta if is_system_abs(d[1])]))
    return res
    

def load_meta(which):
    assert which in ['DUC_2001', 'DUC_2002', 'TAC_2009']
    folder = os.path.join('/tmp', which)
    if not os.path.exists(folder):
        process_ductac(which)
    assert os.path.exists(folder)
    meta_file = os.path.join(folder, 'meta.json')
    with open(os.path.join(meta_file)) as fp:
        meta = json.load(fp)
    return meta

def is_manual_abs(ID):
    return ID.isalpha() and ID >= 'A' and ID <= 'J'

def is_baseline_abs(ID):
    return ID.isdigit() and int(ID) >=1 and int(ID) <= 3

def is_system_abs(ID):
    return ID.isdigit() and int(ID) >= 15 and int(ID) <= 31

def ductac_get_summaries(which, docId, abs_filter_func):
    meta = load_meta(which)
    meta = [d for d in meta if d[0] == docId and abs_filter_func(d[1])]
    folder = os.path.join('/tmp', which)
    abses = []
    scores = []
    for docid, absid, score in meta:
        fname = os.path.join(folder, 'text', docid + '--' + absid + '.txt')
        with open(fname) as fp:
            abses.append(fp.read())
            scores.append(score)
    return abses, scores

def ductac_get_model_summaries(which, docId):
    "Return a list of strings."
    # return ductac_get_summaries(which, docId, is_manual_abs)
    folder = os.path.join('/tmp', which)
    fname = os.path.join(folder, 'text', docId + '--X.txt')
    with open(fname) as fp:
        return [fp.read()]

def ductac_get_system_summaries(which, docId):
    "Return a list of [text], and a list of scores [float]"
    return ductac_get_summaries(which, docId, is_system_abs)

def ductac_get_system_summaries_by_system(which, systemId):
    "Return a list of [system, model] text and a list of scores [float]."
    meta = load_meta(which)
    meta = [d for d in meta if d[1] == systemId]
    folder = os.path.join('/tmp', which)
    def get_doc_content(ID):
        fname = os.path.join(folder, 'text', ID + '.txt')
        with open(fname) as fp:
            return fp.read()
    res = []
    scores = []
    for d in meta:
        model = get_doc_content(d[0] + '--X')
        system = get_doc_content(d[0] + '--' + systemId)
        res.append((model, system))
        scores.append(d[2])
    return res, scores

def __test():
    docIds = ductac_get_docIds('DUC_2002')
    process_ductac('DUC_2002')
    docIds
    docIds[0]
    ductac_get_model_summaries('DUC_2002', docIds[0])
    ductac_get_system_summaries('DUC_2002', docIds[0])
    systemIds = ductac_get_systemIds('DUC_2002')
    len(systemIds)
    texts, scores = ductac_get_system_summaries_by_system('DUC_2002', '15')
    
    len(texts)
    len(scores)
