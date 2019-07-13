from modules.rnn import GRUEncoder
from modules.rnn import GRUDecoder
from modules.attention import MLPAttention
from modules.utils import *
from models.base_model import BaseModel
from utils.pack import Pack
from utils.metrics import accuracy
from torch.nn.utils import clip_grad_norm_
from modules.criterions import SequenceCrossEntropy


class Seq2Seq(BaseModel):
    def __init__(
            self,
            embedding,
            embedding_size,
            hidden_size,
            start_index,
            end_index,
            padding_index,
            dropout=0.2,
            teaching_force_rate=0.5,
    ):
        super().__init__()
        self.embedding = embedding
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.start_index = start_index
        self.end_index = end_index
        self.padding_index = padding_index
        self.dropout = dropout
        self.teaching_force_rate = teaching_force_rate

        self.encoder = GRUEncoder(embedding_size, hidden_size // 2, dropout=dropout)

        decoder_attn = MLPAttention(hidden_size, hidden_size, hidden_size)
        decoder_input_size = embedding_size + hidden_size
        num_classes = embedding.weight.size(0)
        self.decoder = GRUDecoder(
            decoder_input_size,
            hidden_size,
            num_classes,
            start_index,
            end_index,
            embedding,
            attention=decoder_attn,
            dropout=dropout
        )
        weight = torch.ones(num_classes)
        weight[padding_index] = 0.0
        self.cross_entropy = SequenceCrossEntropy(padding_idx=padding_index, weight=weight)

    def forward(self, inputs, evaluation=False):
        """
        train and eval
        """
        if self.training or evaluation:
            assert inputs.response is not None
        if hasattr(inputs, 'response'):
            response_token, response_len = inputs.response

        post_token, post_len = inputs.post
        embedded_post = self.embedding(post_token)
        encoder_outputs, encoder_hidden = self.encoder((embedded_post, post_len))
        encoder_outputs_mask = post_token.ne(self.padding_index)
        if self.training:
            logits = self.decoder(
                encoder_hidden[0],
                target=response_token,
                attn_value=encoder_outputs,
                attn_mask=encoder_outputs_mask,
                teaching_force_rate=self.teaching_force_rate
            )
        elif evaluation:
            # eval, we need to obtain targets max len, so `target` is required
            logits = self.decoder(
                encoder_hidden[0],
                target=response_token,
                attn_value=encoder_outputs,
                attn_mask=encoder_outputs_mask
            )
        else:
            # test
            logits = self.decoder(
                encoder_hidden[0],
                target=None,
                attn_value=encoder_outputs,
                attn_mask=encoder_outputs_mask,
                early_stop=True
            )
        outputs = Pack(logits=logits)
        return outputs

    def beam_search(self, inputs, beam_size=4, per_node_beam_size=4):
        # designed for test or interactive mode
        post_token, post_len = inputs.post
        embedded_post = self.embedding(post_token)
        encoder_outputs, encoder_hidden = self.encoder((embedded_post, post_len))
        encoder_outputs_mask = post_token.ne(self.padding_index)
        all_top_k_predictions, log_probabilities = \
            self.decoder.forward_beam_search(encoder_hidden[0],
                                             attn_value=encoder_outputs,
                                             attn_mask=encoder_outputs_mask,
                                             beam_size=beam_size,
                                             per_node_beam_size=per_node_beam_size,
                                             early_stop=True)
        predictions = all_top_k_predictions[:, 0, :]
        outputs = Pack(predictions=predictions)
        return outputs

    def collect_metrics(self, outputs, target):
        """
        collect_metrics
        """
        num_samples = target.size(0)
        metrics = Pack(num_samples=num_samples)
        loss = 0

        logits = outputs.logits
        nll = self.cross_entropy(logits, target)
        predictions = logits.argmax(dim=2)
        acc = accuracy(predictions, target, padding_idx=self.padding_index)
        ppl = nll.exp()
        metrics.add(nll=nll, acc=acc, ppl=ppl)
        loss += nll

        metrics.add(loss=loss)
        return metrics

    def iterate(self, inputs, optimizer=None, grad_clip=None):
        """
        iterate
        """
        outputs = self.forward(inputs, evaluation=not self.training)
        response_token, response_len = inputs.response
        target = response_token[:, 1:]
        metrics = self.collect_metrics(outputs, target)

        loss = metrics.loss
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        if self.training:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(parameters=self.parameters(),
                                max_norm=grad_clip)
            optimizer.step()
        return metrics
