import torch
from utils.metrics_manager import MetricsManager
from utils.metrics import accuracy, bleu, distinct
from utils.pack import Pack
import math


BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class Generator(object):
    def __init__(
            self,
            model,
            data_iter,
            src_vocab,
            tgt_vocab,
            logger,
            beam_size=4,
            per_node_beam_size=4,
            num_steps=50,
            result_path=None
    ):
        self.model = model
        assert beam_size > 0
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size
        self.num_steps = num_steps
        self.data_iter = data_iter
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.result_path = result_path
        self._new_file = True
        self.logger = logger

        self.end_index = self.tgt_vocab.stoi[EOS_TOKEN]
        self.padding_index = self.tgt_vocab.stoi[PAD_TOKEN]

    def generate(self):
        """
        generate
        """
        self.model.eval()
        mm = MetricsManager()
        with torch.no_grad():
            for inputs in self.data_iter:
                metrics = self.iterate(inputs=inputs)
                mm.update(metrics)
        self.logger.info('Generate finished!\n')
        self.logger.info(mm.report_cum())

    def iterate(self, inputs):
        """
        iterate
        """
        outputs = Pack()
        # We need ground-truth PPL
        eval_outputs = self.model(inputs, is_training=True)
        logits = eval_outputs.logits
        outputs.add(logits=logits)

        if self.beam_size == 1:
            test_outputs = self.model(inputs, is_training=False, num_steps=self.num_steps)
            prediction = test_outputs.logits.argmax(dim=2)
            outputs.add(prediction=prediction)
        else:
            test_outputs = \
                self.model.beam_forward(
                    inputs,
                    beam_size=self.beam_size,
                    per_node_beam_size=self.per_node_beam_size,
                    num_steps=self.num_steps
                )
            outputs.add(prediction=test_outputs.prediction)

        tgt_token, tgt_len = inputs.tgt
        target = tgt_token[:, 1:]
        metrics = self.collect_metrics(inputs, outputs, target)
        return metrics

    def collect_metrics(self, inputs, outputs, target):
        """
        Collect metrics and save generated results
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)

        logits = outputs.logits
        # num_tokens = target.ne(self.padding_index).sum().item()
        nll = self.model.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        mean_nll = nll
        ppx = math.exp(mean_nll.item())
        metrics.add(nll=mean_nll, ppx=ppx)

        prediction = outputs.prediction
        acc = accuracy(prediction, target,
                       padding_idx=self.padding_index,
                       end_idx=self.end_index)
        predict_sentences = tensor2str(prediction, self.tgt_vocab, sequential=True, end_index=self.end_index)
        src_sentences = tensor2str(inputs.src[0], self.src_vocab, sequential=True, end_index=self.padding_index)
        target_sentences = tensor2str(target, self.tgt_vocab, sequential=True, end_index=self.end_index)
        bleu_score = bleu(predict_sentences, target_sentences)
        dist1, dist2 = distinct(predict_sentences)
        metrics.add(acc=acc,
                    bleu_score=bleu_score,
                    dist1=dist1,
                    dist2=dist2)
        if self.result_path is not None:
            self.save_result(src_sentences, target_sentences, predict_sentences)
        return metrics

    def save_result(self, src, target, predict):
        if self._new_file:
            self._new_file = False
            with open(self.result_path, 'w') as result_file:
                for pos, tgt, pred in zip(src, target, predict):
                    result_file.write("----------------------------------------\n")
                    result_file.write("> {}\n".format(pos))
                    result_file.write("< {}\n".format(tgt))
                    result_file.write("< {}\n".format(pred))
        else:
            with open(self.result_path, 'a') as result_file:
                for pos, tgt, pred in zip(src, target, predict):
                    result_file.write("----------------------------------------\n")
                    result_file.write("> {}\n".format(pos))
                    result_file.write("< {}\n".format(tgt))
                    result_file.write("< {}\n".format(pred))


def tensor2str(tensor, vocab, sequential=True, end_index=-1):
    batch_size = tensor.size(0)
    if sequential:
        strings = []
        for idx in range(batch_size):
            array = tensor[idx].tolist()
            token_list = []
            for token_id in array:
                if token_id == end_index:
                    break
                token_list.append(vocab.itos[token_id])
            strings.append(' '.join(token_list))
    else:
        strings = [vocab.itos[token_id] for token_id in tensor.tolist()]
    return strings


