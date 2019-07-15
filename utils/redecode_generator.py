from utils.generator import Generator
from utils.pack import Pack
from utils.metrics import accuracy, bleu, distinct


class RedecodeGenerator(Generator):
    def iterate(self, inputs):
        """
        iterate
        """
        outputs = Pack()
        # We need greedy search to obtain PPL
        eval_outputs = self.model.forward(inputs, evaluation=True)
        logits_1 = eval_outputs.logits_1
        logits_2 = eval_outputs.logits_2
        outputs.add(logits_1=logits_1, logits_2=logits_2)
        if self.beam_size == 1:
            test_outputs = self.model.forward(inputs, evaluation=False)
            predictions_1 = test_outputs.logits_1.argmax(dim=2)
            predictions_2 = test_outputs.logits_2.argmax(dim=2)
            outputs.add(predictions_1=predictions_1, predictions_2=predictions_2)
        else:
            test_outputs = \
                self.model.beam_search(inputs, self.beam_size, self.per_node_beam_size)
            outputs.add(predictions_1=test_outputs.predictions_1,
                        predictions_2=test_outputs.predictions_2)
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

        logits_1 = outputs.logits_1
        nll_1 = self.model.cross_entropy(logits_1, target)
        ppl_1 = nll_1.exp()
        metrics.add(nll=nll_1, ppl=ppl_1)

        logits_2 = outputs.logits_2
        nll_2 = self.model.cross_entropy(logits_2, target)
        ppl_2 = nll_2.exp()
        metrics.add(nll=nll_2, ppl=ppl_2)

        predictions_1 = outputs.predictions_1
        acc_1 = accuracy(predictions_1, target, padding_idx=self.model.padding_index)
        predict_sentences_1 = self.tensor2str(predictions_1)
        target_sentences = self.tensor2str(target)
        bleu_1_1, bleu_2_1 = bleu(predict_sentences_1, target_sentences)
        intra_dist1_1, intra_dist2_1, inter_dist1_1, inter_dist2_1 = distinct(predict_sentences_1)
        metrics.add(acc_1=acc_1,
                    bleu_1_1=bleu_1_1,
                    bleu_2_1=bleu_2_1,
                    intra_dist1_1=intra_dist1_1,
                    intra_dist2_1=intra_dist2_1,
                    inter_dist1_1=inter_dist1_1,
                    inter_dist2_1=inter_dist2_1)

        predictions_2 = outputs.predictions_2
        acc_2 = accuracy(predictions_2, target, padding_idx=self.model.padding_index)
        predict_sentences_2 = self.tensor2str(predictions_2)
        target_sentences = self.tensor2str(target)
        bleu_1_2, bleu_2_2 = bleu(predict_sentences_2, target_sentences)
        intra_dist1_2, intra_dist2_2, inter_dist1_2, inter_dist2_2 = distinct(predict_sentences_2)
        metrics.add(acc_2=acc_2,
                    bleu_1_2=bleu_1_2,
                    bleu_2_2=bleu_2_2,
                    intra_dist1_2=intra_dist1_2,
                    intra_dist2_2=intra_dist2_2,
                    inter_dist1_2=inter_dist1_2,
                    inter_dist2_2=inter_dist2_2)
        if self.result_path is not None:
            self.save_result(predict_sentences_1)
            self.save_result(predict_sentences_2)

        return metrics
