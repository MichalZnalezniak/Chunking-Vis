import json
import re
import string

import numpy as np
import onnxruntime_genai as og
import tiktoken
from django.apps import apps
from sklearn.metrics.pairwise import cosine_similarity

from .utils import chunk_by_words


def get_splitter(splitter):
    return splitter_map[splitter]


class Splitter:
    def calculate_chunk_info(self, chunked_text):
        chunks_number = 0
        characters_number = 0

        for i, (chunk, is_overlap) in enumerate(chunked_text):
            if isinstance(chunk, str) and re.fullmatch(r"[\s]*", chunk):
                continue
            chunk_len = len(chunk)
            if not is_overlap:
                chunks_number += 1

            if is_overlap and 0 < i < len(chunked_text) - 1:
                characters_number += chunk_len * 2
            else:
                characters_number += chunk_len

        average_chunk_size = (characters_number / chunks_number) if chunks_number else 0
        return chunks_number, characters_number, average_chunk_size

    def join_tokens(self, tokens):
        if not tokens:
            return ""

        result = tokens[0]
        for tok in tokens[1:]:
            result += tok
        return result

    def split_by_separators(self, text, separators):
        if not separators:
            return [text]
        else:
            pattern = "(" + "|".join(map(re.escape, separators)) + ")"
            parts = re.split(pattern, text)
            merged_parts = [parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i] for i in range(0, len(parts), 2) if parts[i]]
            return merged_parts


class FixedLengthSplitter(Splitter):
    def process_text(self, text):
        return text

    def chunk(self, text, chunk_size, overlap, separators):
        pieces = []
        parts = self.split_by_separators(text, separators)
        for part in parts:
            start = 0
            text = self.process_text(part)
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                is_last_chunk = end >= len(text)
                if overlap > 0:
                    non_overlap_part = chunk[overlap if start > 0 else 0 : None if is_last_chunk else -overlap]
                    overlap_part_end = chunk[-overlap:]
                    pieces.append(
                        (
                            self.join_tokens(non_overlap_part),
                            False,
                        )
                    )
                    if not is_last_chunk:
                        pieces.append(
                            (
                                self.join_tokens(overlap_part_end),
                                True,
                            )
                        )
                else:
                    pieces.append((self.join_tokens(chunk), False))
                start += chunk_size - overlap

        chunks_number, characters_number, average_chunk_size = self.calculate_chunk_info(pieces)
        return pieces, chunks_number, characters_number, average_chunk_size


class WorldFixedLengthSplitter(FixedLengthSplitter):
    def process_text(self, text):
        return text.split(" ")

    def join_tokens(self, tokens):
        if not tokens:
            return ""

        result = tokens[0]
        for tok in tokens[1:]:
            if tok in string.punctuation:
                result += tok
            else:
                result += " " + tok
        if string.punctuation not in tokens[-1]:
            result += " "
        return result


class BPETokenSplitter(FixedLengthSplitter):
    def __init__(self, model_name="gpt-4o"):
        self.encoder = tiktoken.encoding_for_model(model_name)

    def process_text(self, text):
        token_ids = self.encoder.encode(text)
        tokens = [self.encoder.decode([tid]) for tid in token_ids]
        return tokens


class RecursiveSplitter(FixedLengthSplitter):
    def chunk_by_words(text, chunk_size):
        words = text.split(" ")
        chunks = []
        current_chunk = ""

        for word in words:
            add_len = len(word) + (1 if current_chunk else 0)
            if len(current_chunk) + add_len <= chunk_size:
                current_chunk += (" " if current_chunk else "") + word
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = " " + word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def split_by_separators_rec(self, text, separators, chunk_size):
        if not separators:
            return [text]
        if separators[0] == "":
            return [text]
        elif separators[0] == " ":
            return chunk_by_words(text, chunk_size)
        else:
            pattern = "(" + "|".join(map(re.escape, separators)) + ")"
            parts = re.split(pattern, text)
        merged_parts = parts
        return merged_parts

    def chunk(self, text, chunk_size, overlap, separators):
        if not separators:
            return super().chunk(text, chunk_size, 0, separators)

        parts = self.split_by_separators_rec(text, [separators[0]], chunk_size)
        all_pieces = []
        for part in parts:
            if len(part) > chunk_size and len(separators) > 1:
                pieces, _, _, _ = self.chunk(
                    part,
                    chunk_size,
                    0,
                    separators[1:],
                )
            else:
                pieces, _, _, _ = super().chunk(part, chunk_size, 0, [])

            all_pieces.extend(pieces)
        chunks_number, characters_number, average_chunk_size = self.calculate_chunk_info(all_pieces)
        return all_pieces, chunks_number, characters_number, average_chunk_size


class SemanticSplitter(Splitter):
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.tokenizer = apps.get_app_config("chunker").tokenizer_semantic
        self.session = apps.get_app_config("chunker").session_semnatic

    def encode(self, text):
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(length=512)
        encoded = self.tokenizer.encode(text)
        ids = np.array([encoded.ids], dtype=np.int64)
        att_mask = np.array([encoded.attention_mask], dtype=np.int64)
        tok_type_ids = np.zeros_like(ids, dtype=np.int64)

        ort_inputs = {
            "input_ids": ids,
            "attention_mask": att_mask,
            "token_type_ids": tok_type_ids,
        }
        outputs = self.session.run(None, ort_inputs)
        last_hidden = outputs[0]

        mask = ort_inputs["attention_mask"].astype(np.float32)
        mask = mask[:, :, None]

        summed = (last_hidden * mask).sum(axis=1)
        counts = mask.sum(axis=1)
        embeddings = summed / counts

        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings

    def chunk(self, text, chunk_size, overlap, separators):
        semantic_chunks = []
        chunks = [c for c in text.split(".") if c]
        embedded_chunks = [self.encode(chunk).reshape(-1, 1) for chunk in chunks]
        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]
            j = i + 1
            while j < len(chunks):
                sim = cosine_similarity(embedded_chunks[i], embedded_chunks[j])[0][0]
                if sim > self.threshold:
                    current_chunk += "." + chunks[j]
                    j += 1
                else:
                    break
            semantic_chunks.append((current_chunk + ".", False))
            i = j

        chunks_number, characters_number, average_chunk_size = self.calculate_chunk_info(semantic_chunks)
        return semantic_chunks, chunks_number, characters_number, average_chunk_size


class AgenticSplitter(Splitter):
    def __init__(self):
        self.tokenizer = apps.get_app_config("chunker").tokenizer_agentic
        self.model = apps.get_app_config("chunker").model_agentic

    @staticmethod
    def restore_special_chars_charwise(original, chunked_list):
        restored_chunks = []
        orig_idx = 0

        for chunk in chunked_list:
            restored_chunk = ""
            chunk_idx = 0

            while chunk_idx < len(chunk):
                if orig_idx >= len(original):
                    raise ValueError("Reached end of original text unexpectedly")
                if chunk[chunk_idx] == original[orig_idx]:
                    restored_chunk += original[orig_idx]
                    orig_idx += 1
                    chunk_idx += 1
                else:
                    restored_chunk += original[orig_idx]
                    orig_idx += 1

            while orig_idx < len(original) and original[orig_idx] in "\n\t ":
                restored_chunk += original[orig_idx]
                orig_idx += 1

            restored_chunks.append(restored_chunk)

        return restored_chunks

    def return_prompt(self, text):
        text = text.split(".")
        chat_template = """
        <|system|> \n
            You are a text assistant. You will receive a list of sentences. Your task is to combine semantically related sentences into coherent chunks while preserving the original order and text. 
            Follow these rules:
                Rules:
                    1. Merge consecutive sentences if they describe the same event, idea, topic, or concept.
                    2. Only merge sentences that are semantically related. Start a new chunk if the topic changes.
                    3. Maintain the original order of sentences; do not rearrange them.
                    4. Ensure smooth integration when combining sentences: adjust wording slightly if needed, use conjunctions or punctuation for natural flow.
                    5. Avoid overly long chunks. If a merged chunk contains multiple distinct sub-ideas, split it logically.
                    6. Return the result as a JSON list of strings. Each string should be one chunk containing one or more sentences.
        <|end|>
        <|user|>
        \n
            Combine the following sentences:
            {text}
        <|end|>
        \n
        <|assistant|>
        """
        prompt = f"{chat_template.format(text=text)}"
        return prompt

    def chunk(self, text, chunk_size, overlap, separators):
        """Split text into chunks using LLM to determine natural breakpoints based on context"""
        chunks = []
        prompt = self.return_prompt(text)
        input_tokens = self.tokenizer.encode(prompt)
        input_tokens = np.array(input_tokens)
        search_options = {}
        search_options["max_length"] = 2048
        search_options["batch_size"] = 1
        params = og.GeneratorParams(self.model)
        params.set_search_options(**search_options)
        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)

        while not generator.is_done():
            generator.generate_next_token()
        output_tokens = generator.get_sequence(0)
        del generator
        del params
        generated_tokens = output_tokens[len(input_tokens) :]
        generated_tokens = self.tokenizer.decode(generated_tokens)
        generated_tokens = generated_tokens.replace("```json", "").replace("```", "")
        chunks = json.loads(generated_tokens)
        chunks = AgenticSplitter.restore_special_chars_charwise(text, chunks)
        chunks = [(chunk, False) for chunk in chunks]
        chunks_number, characters_number, average_chunk_size = self.calculate_chunk_info(chunks)
        return chunks, chunks_number, characters_number, average_chunk_size


splitter_map = {
    "Character Splitter": FixedLengthSplitter(),
    "Word Splitter": WorldFixedLengthSplitter(),
    "Token Splitter": BPETokenSplitter(),
    "Recursive Splitter": RecursiveSplitter(),
    "Semantic Splitter": SemanticSplitter(),
    "Agentic Splitter": AgenticSplitter(),
}
