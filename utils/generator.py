import torch
from utils.metrics_manager import MetricsManager
from utils.metrics import accuracy, bleu, distinct
from utils.pack import Pack


class Generator(object):
    def __init__(
            self,
            model,
            data_iter,
            vocab,
            logger,
            beam_size=4,
            per_node_beam_size=4,
            result_path=None
    ):
        self.model = model
        assert beam_size > 0
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size
        self.data_iter = data_iter
        self.vocab = vocab
        self.result_path = result_path
        self._new_file = True
        self.logger = logger

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
        # We need greedy search to obtain PPL
        eval_outputs = self.model.forward(inputs, evaluation=True)
        logits = eval_outputs.logits
        outputs.add(logits=logits)
        if self.beam_size == 1:
            test_outputs = self.model.forward(inputs, evaluation=False)
            predictions = test_outputs.logits.argmax(dim=2)
            outputs.add(predictions=predictions)
        else:
            test_outputs = \
                self.model.beam_search(inputs, self.beam_size, self.per_node_beam_size)
            outputs.add(predictions=test_outputs.predictions)
        response_token, response_len = inputs.response
        target = response_token[:, 1:]
        metrics = self.collect_metrics(outputs, target)
        return metrics

    def collect_metrics(self, outputs, target):
        """
        Collect metrics and save generated results
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)

        logits = outputs.logits
        nll = self.model.cross_entropy(logits, target)
        ppl = nll.exp()
        metrics.add(nll=nll, ppl=ppl)

        predictions = outputs.predictions
        acc = accuracy(predictions, target, padding_idx=self.model.padding_index)
        predict_sentences = self.tensor2str(predictions)
        target_sentences = self.tensor2str(target)
        bleu_1, bleu_2 = bleu(predict_sentences, target_sentences)
        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(predict_sentences)
        metrics.add(acc=acc,
                    bleu_1=bleu_1,
                    bleu_2=bleu_2,
                    intra_dist1=intra_dist1,
                    intra_dist2=intra_dist2,
                    inter_dist1=inter_dist1,
                    inter_dist2=inter_dist2)
        if self.result_path is not None:
            self.save_result(predict_sentences)
        return metrics

    def tensor2str(self, tensor):
        batch_size = tensor.size(0)
        sentences = []
        for idx in range(batch_size):
            array = tensor[idx].tolist()
            token_list = []
            for token_id in array:
                if token_id == self.model.end_index:
                    break
                token_list.append(self.vocab.itos[token_id])
            sentences.append(' '.join(token_list))
        return sentences

    def save_result(self, sentences):
        if self._new_file:
            self._new_file = False
            with open(self.result_path, 'w') as result_file:
                result_file.writelines([s + '\n' for s in sentences])
        else:
            with open(self.result_path, 'a') as result_file:
                result_file.writelines([s + '\n' for s in sentences])
