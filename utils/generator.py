import torch
from utils.metrics_manager import MetricsManager
from utils.metrics import accuracy, bleu, distinct
from utils.pack import Pack


class Generator(object):
    def __init__(
            self,
            model,
            end_index,
            padding_index,
            data_iter,
            vocab,
            beam_size=4,
            per_node_beam_size=4,
            result_path=None
    ):
        self.model = model
        self.end_index = end_index
        self.padding_index = padding_index
        assert beam_size > 0
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size
        self.data_iter = data_iter
        self.vocab = vocab
        self.result_path = result_path
        self._new_file = True

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
        return mm

    def iterate(self, inputs):
        """
        iterate
        """
        if self.beam_size == 1:
            outputs = self.model.forward(inputs)
            logits = outputs.logits
            predictions = logits.argmax(dim=2)
            outputs.add(predictions=predictions)
        else:
            outputs = \
                self.model.beam_search(inputs, self.beam_size, self.per_node_beam_size)
        target = inputs.response[:, 1:]
        metrics = self.collect_metrics(outputs, target)
        return metrics

    def collect_metrics(self, outputs, target):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        predictions = outputs.predictions
        acc = accuracy(predictions, target, padding_idx=self.padding_index)
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
                if token_id == self.end_index:
                    break
                token_list.append(self.vocab.itos[token_id])
            sentences.append(' '.join(token_list))
        return sentences

    def save_result(self, sentences):
        if self._new_file:
            with open(self.result_path, 'w') as result_file:
                result_file.writelines([s + '\n' for s in sentences])
        else:
            with open(self.result_path, 'a') as result_file:
                result_file.writelines([s + '\n' for s in sentences])
