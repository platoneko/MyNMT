import numpy as np
import torch
import torch.nn.functional as F

from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity


def accuracy(preds, targets, padding_idx=None):
    """
    preds: (batch_size, preds_len)
    targets: (batch_size, targets_len)
    """
    batch_size, targets_len = targets.size()
    preds_len = preds.size(1)
    if preds_len > targets_len:
        preds = preds[:, :targets_len]
    elif preds_len < targets_len:
        tensor = targets.new_full((batch_size, targets_len), fill_value=-1)
        tensor[:, :preds_len] = preds
        preds = tensor
    trues = (preds == targets).float()
    if padding_idx is not None:
        weights = targets.ne(padding_idx).float()
        acc = (weights * trues).sum(1) / weights.sum(1)
        acc = acc.mean()
    else:
        acc = trues.mean()
    return acc


def perplexity(logits, targets, weight=None, padding_idx=None):
    """
    :param
    logits : (batch_size, max_len, num_classes)
    targets : (batch_size, max_len)
    :return
    ppl : (1)
    """
    batch_size = logits.size(0)
    if weight is None and padding_idx is not None:
        weight = torch.ones(logits.size(-1))
        weight[padding_idx] = 0
    nll = F.cross_entropy(
        input=logits.reshape(-1, logits.size(-1)),
        target=targets.reshape(-1),
        weight=weight,
        reduction='none'
    )
    nll = nll.view(batch_size, -1).sum(dim=1)
    if padding_idx is not None:
        word_cnt = targets.ne(padding_idx).float().sum(dim=1)
        nll = nll / word_cnt
    ppl = nll.exp().mean()
    return ppl


def bleu(hyps, refs):
    """
    bleu
    """
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2


def distinct(seqs):
    """
    distinct
    """
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


def cosine(X, Y):
    """
    cosine
    """
    sim = np.sum(X * Y, axis=1) / \
        (np.sqrt((np.sum(X * X, axis=1) * np.sum(Y * Y, axis=1))) + 1e-10)
    return sim


class EmbeddingMetrics(object):
    """
    EmbeddingMetrics
    """
    def __init__(self, field):
        self.field = field
        assert field.embeddings is not None
        self.embeddings = np.array(field.embeddings)

    def texts2embeds(self, texts):
        """
        texts2embeds
        """
        texts = [self.field.numericalize(text)[1:-1] for text in texts]
        embeds = []
        for text in texts:
            vecs = self.embeddings[text]
            mask = vecs.any(axis=1)
            vecs = vecs[mask]
            if vecs.shape[0] == 0:
                vecs = np.zeros((1,) + vecs.shape[1:])
            embeds.append(vecs)
        return embeds

    def average(self, embeds):
        """
        average
        """
        avg_embeds = [embed.mean(axis=0) for embed in embeds]
        avg_embeds = np.array(avg_embeds)
        return avg_embeds

    def extrema(self, embeds):
        """
        extrema
        """
        ext_embeds = []
        for embed in embeds:
            s_max = np.max(embed, axis=0)
            s_min = np.min(embed, axis=0)
            s_plus = np.abs(s_min) <= s_max
            s = s_max * s_plus + s_min * np.logical_not(s_plus)
            ext_embeds.append(s)
        ext_embeds = np.array(ext_embeds)
        return ext_embeds

    def greedy(self, hyp_embeds, ref_embeds):
        """
        greedy
        """
        greedy_sim = []
        for hyp_embed, ref_embed in zip(hyp_embeds, ref_embeds):
            cos_sim = cosine_similarity(hyp_embed, ref_embed)
            g_sim = (cos_sim.max(axis=1).mean() +
                     cos_sim.max(axis=0).mean()) / 2
            greedy_sim.append(g_sim)
        greedy_sim = np.array(greedy_sim)
        return greedy_sim

    def embed_sim(self, hyp_texts, ref_texts):
        """
        embed_sim
        """
        assert len(hyp_texts) == len(ref_texts)
        hyp_embeds = self.texts2embeds(hyp_texts)
        ref_embeds = self.texts2embeds(ref_texts)

        ext_hyp_embeds = self.extrema(hyp_embeds)
        ext_ref_embeds = self.extrema(ref_embeds)
        ext_sim = cosine(ext_hyp_embeds, ext_ref_embeds)
        # print(len(ext_sim), (ext_sim > 0).sum())
        # print(ext_sim.sum() / (ext_sim > 0).sum())
        ext_sim_avg = ext_sim.mean()

        avg_hyp_embeds = self.average(hyp_embeds)
        avg_ref_embeds = self.average(ref_embeds)
        avg_sim = cosine(avg_hyp_embeds, avg_ref_embeds)
        # print(len(avg_sim), (avg_sim > 0).sum())
        # print(avg_sim.sum() / (avg_sim > 0).sum())
        avg_sim_avg = avg_sim.mean()

        greedy_sim = self.greedy(hyp_embeds, ref_embeds)
        # print(len(greedy_sim), (greedy_sim > 0).sum())
        # print(greedy_sim.sum() / (greedy_sim > 0).sum())
        greedy_sim_avg = greedy_sim.mean()

        return ext_sim_avg, avg_sim_avg, greedy_sim_avg
