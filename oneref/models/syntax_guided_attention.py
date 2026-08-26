import math
from collections import OrderedDict

import spacy
import torch


class SyntaxBiasBuilder:
    """Build a text-text syntax bias for the joint vision-language sequence."""

    def __init__(self, syntax_bias_lambda=0.1, sigma=2.0, nlp_model=None):
        self.syntax_bias_lambda = syntax_bias_lambda
        self.sigma = sigma
        self.nlp = nlp_model or self._load_spacy_model()
        self.cache = OrderedDict()
        self.max_cache_size = 4096

    @staticmethod
    def _load_spacy_model():
        try:
            return spacy.load("en_core_web_md")
        except OSError:
            try:
                return spacy.load("en_core_web_sm")
            except OSError as exc:
                raise ImportError(
                    "Please install a spaCy English model: "
                    "python -m spacy download en_core_web_md"
                ) from exc

    @staticmethod
    def _path_to_root(token):
        path = [token.i]
        current = token
        while current.head.i != current.i:
            current = current.head
            path.append(current.i)
        return path

    def _dependency_distance(self, token1, token2):
        if token1.i == token2.i:
            return 0.0

        path1 = self._path_to_root(token1)
        path2 = self._path_to_root(token2)
        path2_positions = {token_index: index for index, token_index in enumerate(path2)}
        for index1, token_index in enumerate(path1):
            if token_index in path2_positions:
                return float(index1 + path2_positions[token_index])
        return 10.0

    @staticmethod
    def _piece_to_word_indices(text, doc, pieces):
        """Map SentencePiece tokens to spaCy words using character spans."""
        normalized_text = text.lower()
        cursor = 0
        fallback_word = -1
        mapping = []

        for piece in pieces:
            if piece.startswith("▁"):
                fallback_word += 1

            surface = piece.replace("▁", " ").strip().lower()
            if not surface or surface.startswith("<"):
                mapping.append(None)
                continue

            while cursor < len(normalized_text) and normalized_text[cursor].isspace():
                cursor += 1
            start = normalized_text.find(surface, cursor)
            if start < 0:
                mapping.append(
                    min(fallback_word, len(doc) - 1) if len(doc) and fallback_word >= 0 else None
                )
                continue

            end = start + len(surface)
            word_index = next(
                (
                    token.i
                    for token in doc
                    if token.idx < end and token.idx + len(token.text) > start
                ),
                None,
            )
            mapping.append(word_index)
            cursor = end

        return mapping

    def _text_bias(self, text, pieces, max_text_tokens):
        cache_key = (text, tuple(pieces), max_text_tokens)
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]

        doc = self.nlp(text)
        pieces = pieces[: max_text_tokens - 2]
        piece_to_word = self._piece_to_word_indices(text, doc, pieces)
        word_to_positions = {}
        for piece_index, word_index in enumerate(piece_to_word, start=1):
            if word_index is not None and word_index < len(doc):
                word_to_positions.setdefault(word_index, []).append(piece_index)

        bias = torch.zeros((max_text_tokens, max_text_tokens), dtype=torch.float32)
        for source_word, source_positions in word_to_positions.items():
            for target_word, target_positions in word_to_positions.items():
                distance = self._dependency_distance(doc[source_word], doc[target_word])
                penalty = -(math.exp(distance / self.sigma) - 1.0)
                for source_position in source_positions:
                    for target_position in target_positions:
                        bias[source_position, target_position] = penalty

        self.cache[cache_key] = bias
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)
        return bias

    def build(
        self,
        text_inputs,
        token_pieces,
        sequence_length,
        text_start,
        text_length,
        num_heads,
        device,
        dtype,
    ):
        if not text_inputs or token_pieces is None:
            return None
        if len(text_inputs) != len(token_pieces):
            raise ValueError("text_inputs and token_pieces must have the same batch size")

        batch_bias = torch.zeros(
            (len(text_inputs), sequence_length, sequence_length),
            dtype=torch.float32,
        )
        for batch_index, (text, pieces) in enumerate(zip(text_inputs, token_pieces)):
            text_bias = self._text_bias(text, pieces, text_length)
            batch_bias[
                batch_index,
                text_start : text_start + text_length,
                text_start : text_start + text_length,
            ] = text_bias

        batch_bias = batch_bias.mul(self.syntax_bias_lambda).to(device=device, dtype=dtype)
        return (
            batch_bias.unsqueeze(1)
            .expand(-1, num_heads, -1, -1)
            .reshape(len(text_inputs) * num_heads, sequence_length, sequence_length)
        )
