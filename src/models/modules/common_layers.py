from torch import nn
from omegaconf import DictConfig
import torch
import numpy
from gensim.models import KeyedVectors
from src.vocabulary import Vocabulary
from os.path import exists

def linear_after_attn(in_dim: int, out_dim: int, activation: str) -> nn.Module:
    """Linear layers after attention

        Args:
            in_dim (int): input dimension
            out_dim (int): output dimension
            activation (str): the name of activation function
        """
    # add drop out?
    return torch.nn.Sequential(
        torch.nn.Linear(2 * in_dim, 2 * in_dim),
        torch.nn.BatchNorm1d(2 * in_dim),
        get_activation(activation),
        torch.nn.Linear(2 * in_dim, out_dim),
    )


activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "lkrelu": nn.LeakyReLU(0.3)
}


def get_activation(activation_name: str) -> torch.nn.Module:
    if activation_name in activations:
        return activations[activation_name]
    raise KeyError(f"Activation {activation_name} is not supported")


class RNNLayer(torch.nn.Module):
    """

    """
    __negative_value = -numpy.inf

    def __init__(self, config: DictConfig, pad_idx: int):
        super(RNNLayer, self).__init__()
        self.__pad_idx = pad_idx
        self.__config = config
        self.__rnn = nn.LSTM(
            input_size=config.embed_size,
            hidden_size=config.rnn.hidden_size,
            num_layers=config.rnn.num_layers,
            bidirectional=config.rnn.use_bi,
            dropout=config.rnn.drop_out if config.rnn.num_layers > 1 else 0,
            batch_first=True)
        self.__dropout_rnn = nn.Dropout(config.rnn.drop_out)

    def forward(self, subtokens_embed: torch.Tensor, node_ids: torch.Tensor):
        """

        Args:
            subtokens_embed: [n nodes; max parts; embed dim]
            node_ids: [n nodes; max parts]

        Returns:

        """
        with torch.no_grad():
            is_contain_pad_id, first_pad_pos = torch.max(
                node_ids == self.__pad_idx, dim=1)
            first_pad_pos[~is_contain_pad_id] = node_ids.shape[
                1]  # if no pad token use len+1 position
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos,
                                                           descending=True)
            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
        subtokens_embed = subtokens_embed[sort_indices]
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            subtokens_embed, sorted_path_lengths, batch_first=True)
        # [2; N; rnn hidden]
        _, (node_embedding, _) = self.__rnn(packed_embeddings)
        # [N; rnn hidden]
        node_embedding = node_embedding.sum(dim=0)

        # [n nodes; max parts; rnn hidden]
        node_embedding = self.__dropout_rnn(
            node_embedding)[reverse_sort_indices]

        return node_embedding


class STEncoder(torch.nn.Module):
    """

    encoder for statement

    """

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(STEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__wd_embedding = nn.Embedding(vocabulary_size,
                                           config.embed_size,
                                           padding_idx=pad_idx)
        # Additional embedding value for masked token
        torch.nn.init.xavier_uniform_(self.__wd_embedding.weight.data)
        if exists(config.w2v_path):
            self.__add_w2v_weights(config.w2v_path, vocab)
        self.__rnn_attn = RNNLayer(config, pad_idx)

    def __add_w2v_weights(self, w2v_path: str, vocab: Vocabulary):
        """
        add pretrained word embedding to embedding layer

        Args:
            w2v_path: path to the word2vec model

        Returns:

        """
        model = KeyedVectors.load(w2v_path, mmap="r")
        w2v_weights = self.__wd_embedding.weight.data
        for wd in model.index2word:
            w2v_weights[vocab.convert_token_to_id(wd)] = torch.from_numpy(model[wd])
        self.__wd_embedding.weight.data.copy_(w2v_weights)

    def forward(self, seq: torch.Tensor):
        """

        Args:
            seq: [n nodes (seqs); max parts (seq len); embed dim]

        Returns:

        """
        # [n nodes; max parts; embed dim]
        wd_embedding = self.__wd_embedding(seq)
        # [n nodes; rnn hidden]
        node_embedding = self.__rnn_attn(wd_embedding, seq)
        return node_embedding
