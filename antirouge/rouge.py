"""
Testing rouge score on DUC
"""

from antirouge.data.ductac import ductac_get_docIds, ductac_get_model_summaries, ductac_get_system_summaries

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

ROUGE_PERL_DIR = "/home/hebi/github/reading/pyrouge-orig/tools/ROUGE-1.5.5/"

import pyrouge
import rouge

from pythonrouge.pythonrouge import Pythonrouge

def __test():

    # system summary(predict) & reference summary
    system = [[" Tokyo is the one of the biggest city in the world."]]
    model = [[["The capital of Japan, Tokyo, is the center of Japanese economy."]]]

    # initialize setting of ROUGE to eval ROUGE-1, 2, SU4
    # if you evaluate ROUGE by sentence list as above, set summary_file_exist=False
    # if recall_only=True, you can get recall scores of ROUGE
    r = Pythonrouge(summary_file_exist=False,
                    summary=system, reference=model,
                    n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                    recall_only=True, stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=50,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
    score = r.calc_score()
    print(score)

def myrouge(rouger, system, model):
    assert type(model) is str
    assert type(system) is str
    
    if model is '':
        print('WARNING: model is empty')
        model = ' '
    if system is '':
        print('WARNING: system is empty')
        system = ' '
        
    # very slow. Probably using the perl script in a batch manner can
    # help. But this is not the right way to right software.
    if rouger is 'pythonrouge':
        system = [[system]]
        model = [[[model]]]
        r = Pythonrouge(summary_file_exist=False,
                        summary=system, reference=model,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                        recall_only=True, stemming=True, stopwords=True,
                        word_level=True, length_limit=True, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
        score = r.calc_score()
        return {'ROUGE-1': score['ROUGE-1'],
                'ROUGE-2': score['ROUGE-2'],
                'ROUGE-SU4': score['ROUGE-SU4']}
    # fast
    elif rouger is 'rouge':
        r = rouge.Rouge()
        score = r.get_scores(system, model)
        return {'ROUGE-1': score[0]['rouge-1']['r'],
                'ROUGE-2': score[0]['rouge-2']['r'],
                # XXX no SU4 available
                'ROUGE-SU4': score[0]['rouge-l']['r']}
    assert False
    return None

def myrouge_batch(rouger, systems, models):
    return [myrouge(rouger, s, m) for s,m in zip(systems, models)]


def __test():
    r = pyrouge.Rouge155(ROUGE_PERL_DIR)
    pyrouge.test

    system = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"

    model = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"

    system = "tokyo is the one of the biggest city in the world."
    model = "the capital of japan, tokyo, is the center of japanese economy."
    
    r = rouge.Rouge()
    scores = r.get_scores(hypothesis, reference)
    scores
    myrouge('pythonrouge', system=system, model=model)
    myrouge('rouge', system=system, model=model)

def rouge_by_docId(docId, rouger):
    # print(docId)
    model_sums = ductac_get_model_summaries('DUC_2002', docId)
    system_sums, manual_scores = ductac_get_system_summaries('DUC_2002', docId)

    if not model_sums:
        print('WARNING: model summary for doc {} empty.'.format(docId))
        return [],[]

    models = [model_sums[0] for s in system_sums]
    systems = system_sums

    scores = myrouge_batch(rouger=rouger, models=models, systems=systems)
    scores = [s['ROUGE-2'] for s in scores]
    return scores, manual_scores

def correlation_by_docIds(docIds, rouger):
    all_scores = []
    total = len(docIds)
    ct = 0
    for doc in docIds:
        ct += 1
        if ct % 10 == 0:
            print('{} / {}'.format(ct, total))
        all_scores.append(rouge_by_docId(doc, rouger))
        
    scores = [s for ss in all_scores for s in ss[0]]
    manual_scores = [s for ss in all_scores for s in ss[1]]
    
    pearsonr = stats.pearsonr(manual_scores, scores)[0]
    spearmanr =  stats.spearmanr(manual_scores, scores)[0]
    kendalltau = stats.kendalltau(manual_scores, scores)[0]
    return (pearsonr, spearmanr, kendalltau)


def correlation_by_systemId(systemId, rouger):
    texts, manual_scores = ductac_get_system_summaries_by_system('DUC_2002', systemId)
    scores = []
    # total = len(texts)
    # ct = 0
    for model, system in texts:
        # ct += 1
        # if ct % 10 == 0:
        #     print('{} / {}'.format(ct, total))
        score = myrouge(rouger, model=model, system=system)
        scores.append(score)
    scores = [s['ROUGE-2'] for s in scores]
    pearsonr = stats.pearsonr(manual_scores, scores)[0]
    spearmanr =  stats.spearmanr(manual_scores, scores)[0]
    kendalltau = stats.kendalltau(manual_scores, scores)[0]
    return (pearsonr, spearmanr, kendalltau)


def __test():
    docIds = ductac_get_docIds('DUC_2002')
    
    len(docIds)
    docId = docIds[1]
    # myrouge(rouger='rouge', model='hello', system='')

    correlations = [correlation_by_docId(doc, rouger='rouge') for doc in docIds[:50]]
    pearsonr = [c[0] for c in correlations]
    spearmanr = [c[1] for c in correlations]
    kendalltau = [c[2] for c in correlations]
    np.mean(pearsonr)
    np.mean(spearmanr)
    np.mean(kendalltau)

    # (0.5346888581997241, 0.5747596060549883, 0.4096732345588499) for DUC2002
    correlations = correlation_by_docIds(docIds, rouger='rouge')
    # (0.4292313608665183, 0.47745926154809515, 0.3321880440781539) for DUC2002
    correlations = correlation_by_docIds(docIds, rouger='pythonrouge')
    correlations

    systemIds = ductac_get_systemIds('DUC_2002')
    for systemId in systemIds:
        print(systemId, end=': ')
        print(correlation_by_systemId(systemId, rouger='rouge'))
    
    plt.plot(pearsonr)
    plt.show()

    # calculate the correlation of manual score and rouge score
    plt.plot(manual_scores)
    plt.plot(rouge_1_r)
    plt.show()

    # plot
    plt.plot(manual_scores)
    plt.show()
    plt.plot(rouge_scores)


