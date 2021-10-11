import pickle
from dataclasses import dataclass
from os.path import exists
from typing import Dict, List
from src.utils import PAD, UNK, MASK
from gensim.models import KeyedVectors

TOKEN_TO_ID = "token_to_id"


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]

    @staticmethod
    def build_from_w2v(w2v_path: str, speicial_tokens: List[str] = [PAD, UNK, MASK]):
        """
        build vocabulary from word2vec wv

        Args:
            w2v_path: path to word2vec wv
            speicial_tokens:


        Returns:

        """
        assert exists(w2v_path), f"{w2v_path} not exists!"
        model = KeyedVectors.load(w2v_path, mmap="r")
        attr = dict()
        for idx, tk in enumerate(speicial_tokens):
            attr[tk] = idx
        for wd in model.vocab:
            attr[wd] = model.vocab[wd].index + len(speicial_tokens)
        return Vocabulary(token_to_id=attr)

    @staticmethod
    def load_vocabulary(vocabulary_path: str) -> "Vocabulary":
        if not exists(vocabulary_path):
            raise ValueError(f"Can't find vocabulary in: {vocabulary_path}")
        with open(vocabulary_path, "rb") as vocabulary_file:
            vocabulary_dicts = pickle.load(vocabulary_file)
        token_to_id = vocabulary_dicts[TOKEN_TO_ID]
        return Vocabulary(token_to_id=token_to_id)

    def dump_vocabulary(self, vocabulary_path: str):
        with open(vocabulary_path, "wb") as vocabulary_file:
            vocabulary_dicts = {
                TOKEN_TO_ID: self.token_to_id,
            }
            pickle.dump(vocabulary_dicts, vocabulary_file)

    def convert_token_to_id(self, token: str):
        return self.token_to_id.get(token, self.token_to_id[UNK])

    def convert_tokens_to_ids(self, tokens: List[str]):
        return [self.convert_token_to_id(token) for token in tokens]

    def get_vocab_size(self):
        return len(self.token_to_id)

    def get_pad_id(self):
        return self.convert_token_to_id(PAD)