import collections
import json
import os
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

from chrisbase.util import no_space, no_replacement, no_nonprintable
from transformers import AutoTokenizer, BasicTokenizer, WordpieceTokenizer, BertTokenizer, RobertaTokenizer
from transformers.models.bert.tokenization_bert import load_vocab, whitespace_tokenize
from transformers.tokenization_utils_base import TextInput


class KorbertTokenizer(BertTokenizer):
    """
    Construct a BERT tokenizer for morpheme-analized data.
    """

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            strip_accents=None,
            **kwargs
    ):
        super(BertTokenizer, self).__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.space_tokenizer = SpaceTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = KorbertWordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    def tokenize(self, morps: TextInput, **kwargs):
        sub_tokens = []
        for token in self.space_tokenizer.tokenize(morps):
            if token not in self.all_special_tokens:
                token += '_'
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                sub_tokens.append(sub_token)
        return sub_tokens


class SpaceTokenizer(BasicTokenizer):
    """
    Constructs a BasicTokenizer that will run space splitting.
    """

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        super().__init__(
            do_lower_case=do_lower_case,
            never_split=never_split,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
        )

    def tokenize(self, text, never_split=None):
        return super().tokenize(text, never_split=never_split)

    def _run_split_on_punc(self, text, never_split=None):
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if char == ' ':
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        return ["".join(x) for x in output]


class KorbertWordpieceTokenizer(WordpieceTokenizer):
    """Runs WordPiece tokenization without '##'."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        super().__init__(
            vocab=vocab,
            unk_token=unk_token,
            max_input_chars_per_word=max_input_chars_per_word,
        )

    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


if __name__ == "__main__":
    tokenizer1A = AutoTokenizer.from_pretrained(
        "pretrained/KoELECTRA-Base-v3",
        max_len=512,
        use_fast=True,
    )
    tokenizer1B = BertTokenizer(
        vocab_file="pretrained/KoELECTRA-Base-v3/vocab.txt",
        do_lower_case=False,
        tokenize_chinese_chars=False,
    )
    tokenizer2A = KorbertTokenizer.from_pretrained(
        "pretrained/ELECTRA-morp20.05",
        max_len=512,
        use_fast=False,
        do_lower_case=False,
        tokenize_chinese_chars=False,
    )
    tokenizer2B = KorbertTokenizer(
        vocab_file="pretrained/ELECTRA-morp20.05/vocab.txt",
        do_lower_case=False,
    )
    tokenizer3A = AutoTokenizer.from_pretrained(
        "pretrained/RoBERTa-bbpe21.07-added",
        max_len=512,
        use_fast=False,
        do_lower_case=False,
    )
    tokenizer3B = RobertaTokenizer(
        "pretrained/RoBERTa-bbpe21.07-added/vocab.json",
        "pretrained/RoBERTa-bbpe21.07-added/merges.txt",
        max_len=512,
        use_fast=False,
        do_lower_case=False,
    )
    # https://github.com/huggingface/transformers/issues/1413#issuecomment-538083512
    # tokenizer3A.add_tokens(["/NN", "/NP", "/NR", "/VV", "/VA",
    #                         "/VX", "/VC", "/MM", "/MA", "/IC",
    #                         "/JK", "/JX", "/JC", "/EP", "/EF",
    #                         "/EC", "/ET", "/XP", "/XS", "/XR",
    #                         "/SF", "/SP", "/SS", "/SE", "/SO",
    #                         "/SW", "/SL", "/SH", "/SN"])
    # tokenizer3A.add_tokens(["/NNG", "/NNP", "/NNB", "/NP", "/NR",
    #                         "/VV", "/VA", "/VX", "/VCP", "/VCN",  # 10
    #                         "/MM", "/MAG", "/MAJ", "/IC",  # 14
    #                         "/JKS", "/JKC", "/JKG", "/JKO", "/JKB",  # 19
    #                         "/JKV", "/JKQ", "/JX", "/JC",  # 23
    #                         "/EP", "/EF", "/EC", "/ETN", "/ETM",  # 28
    #                         "/XPN", "/XSN", "/XSV", "/XSA", "/XR",  # 33
    #                         "/SF", "/SP", "/SS", "/SE", "/SO",  # 38
    #                         "/SW", "/SL", "/SH", "/SN"])  # 42
    bbpe_tokenizer = ByteLevelBPETokenizer("pretrained/RoBERTa-bbpe21.07-added/vocab.json", "pretrained/RoBERTa-bbpe21.07-added/merges.txt", lowercase=False)
    # bbpe_tokenizer.add_tokens(["/NN", "/NP", "/NR", "/VV", "/VA",
    #                            "/VX", "/VC", "/MM", "/MA", "/IC",
    #                            "/JK", "/JX", "/JC", "/EP", "/EF",
    #                            "/EC", "/ET", "/XP", "/XS", "/XR",
    #                            "/SF", "/SP", "/SS", "/SE", "/SO",
    #                            "/SW", "/SL", "/SH", "/SN"])
    # bbpe_tokenizer.add_tokens(["/NNG", "/NNP", "/NNB", "/NP", "/NR",
    #                            "/VV", "/VA", "/VX", "/VCP", "/VCN",  # 10
    #                            "/MM", "/MAG", "/MAJ", "/IC",  # 14
    #                            "/JKS", "/JKC", "/JKG", "/JKO", "/JKB",  # 19
    #                            "/JKV", "/JKQ", "/JX", "/JC",  # 23
    #                            "/EP", "/EF", "/EC", "/ETN", "/ETM",  # 28
    #                            "/XPN", "/XSN", "/XSV", "/XSA", "/XR",  # 33
    #                            "/SF", "/SP", "/SS", "/SE", "/SO",  # 38
    #                            "/SW", "/SL", "/SH", "/SN"])  # 42
    print(f"tokenizer3A(#={len(tokenizer3A)})({type(tokenizer3A)}) = {tokenizer3A}")
    print(f"tokenizer3B(#={len(tokenizer3B)})({type(tokenizer3B)}) = {tokenizer3B}")
    print(f"bbpe_tokenizer()({type(bbpe_tokenizer)}) = {bbpe_tokenizer}")
    vocab_items = []
    with Path("pretrained/RoBERTa-bbpe21.07-added/vocab.json").open() as inp, Path("pretrained/RoBERTa-bbpe21.07-added/vocab_items.txt").open('w') as out:
        for t, i in list(json.load(inp).items()):
            vocab_items.append({
                'id': i,
                'encoded': t,
                'decoded': no_space(no_replacement(no_nonprintable(bbpe_tokenizer.decode([i]))))
            })
        out.writelines([f"{x['id']}\t{x['encoded']}\t{x['decoded']}\n" for x in vocab_items])

    sentence1_plain = "한국어 사전학습 모델을 공유합니다."
    sentence2_plain = "지금까지 금해졌다."
    sentence1_morps = "한국어/NNP 사전/NNG 학습/NNG 모델/NNG 을/JKO 공유/NNG 하/XSV ㅂ니다/EF ./SF"
    sentence2_morps = "지금/NNG 까지/JX 금하/VV 어/EC 지/VX 었/EP 다/EF ./SF"
    sentence1_lemma = "한국어 사전 학습 모델 을 공유 합 니다 ."
    sentence2_lemma = "지금 까지 금해 졌 다 ."

    # plain = "[CLS] 한국어 사전학습 모델을 공유합니다. [SEP] 지금까지 금해졌다."
    # morps = "[CLS] 한국어/NNP 사전/NNG 학습/NNG 모델/NNG 을/JKO 공유/NNG 하/XSV ㅂ니다/EF ./SF [SEP] 지금/NNG 까지/JX 금하/VV 어/EC 지/VX 었/EP 다/EF ./SF"
    # print(f"plain={plain}")
    # print(f"morps={morps}")

    # print('tokenizer1A:', tokenizer1A.cls_token, tokenizer1A.cls_token_id, tokenizer1A.sep_token, tokenizer1A.sep_token_id, tokenizer1A.pad_token, tokenizer1A.pad_token_id)
    # print('tokenizer2A:', tokenizer2A.cls_token, tokenizer2A.cls_token_id, tokenizer2A.sep_token, tokenizer2A.sep_token_id, tokenizer2A.pad_token, tokenizer2A.pad_token_id)
    print('tokenizer3A:', tokenizer3A.cls_token, tokenizer3A.cls_token_id, tokenizer3A.sep_token, tokenizer3A.sep_token_id, tokenizer3A.pad_token, tokenizer3A.pad_token_id)


    def tokenized_ids(tokenized):
        return ' '.join(map(lambda x: f'{x:05d}',
                            tokenizer3A.convert_tokens_to_ids(tokenized)))


    def tokenized_lemmas(tokenized):
        return ' '.join(map(lambda x: no_space(no_replacement(no_nonprintable(bbpe_tokenizer.decode([x])))),
                            tokenizer3A.convert_tokens_to_ids(tokenized)))


    print(f"tokens from plain={tokenizer1A.tokenize(f'{tokenizer1A.cls_token} {sentence1_plain} {tokenizer1A.sep_token} {sentence2_plain} {tokenizer1A.pad_token}')}")
    # print(f"tokens from plain={tokenizer1B.tokenize(f'{tokenizer1B.cls_token} {sentence1_plain} {tokenizer1B.sep_token} {sentence2_plain} {tokenizer1B.pad_token}')}")

    print(f"tokens from morps={tokenizer2A.tokenize(f'{tokenizer2A.cls_token} {sentence1_morps} {tokenizer2A.sep_token} {sentence2_morps} {tokenizer2A.pad_token}')}")
    # print(f"tokens from morps={tokenizer2B.tokenize(f'{tokenizer2B.cls_token} {sentence1_morps} {tokenizer2B.sep_token} {sentence2_morps} {tokenizer2B.pad_token}')}")

    print(f"tokens from plain={tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_plain} {tokenizer3A.sep_token} {sentence2_plain} {tokenizer3A.pad_token}')}")
    print(f"tokens from lemma={tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_lemma} {tokenizer3A.sep_token} {sentence2_lemma} {tokenizer3A.pad_token}')}")
    print(f"tokens from morps={tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_morps} {tokenizer3A.sep_token} {sentence2_morps} {tokenizer3A.pad_token}')}")

    print(f"plain -> tokenize(#={len(tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_plain} {tokenizer3A.sep_token} {sentence2_plain} {tokenizer3A.pad_token}'))}) -> tokens_to_ids -> str"
          f" = {tokenized_ids(tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_plain} {tokenizer3A.sep_token} {sentence2_plain} {tokenizer3A.pad_token}'))}")
    print(f"lemma -> tokenize(#={len(tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_lemma} {tokenizer3A.sep_token} {sentence2_lemma} {tokenizer3A.pad_token}'))}) -> tokens_to_ids -> str"
          f" = {tokenized_ids(tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_lemma} {tokenizer3A.sep_token} {sentence2_lemma} {tokenizer3A.pad_token}'))}")
    print(f"morps -> tokenize(#={len(tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_morps} {tokenizer3A.sep_token} {sentence2_morps} {tokenizer3A.pad_token}'))}) -> tokens_to_ids -> str"
          f" = {tokenized_ids(tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_morps} {tokenizer3A.sep_token} {sentence2_morps} {tokenizer3A.pad_token}'))}")

    print(f"plain -> tokenize(#={len(tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_plain} {tokenizer3A.sep_token} {sentence2_plain} {tokenizer3A.pad_token}'))}) -> tokens_to_ids -> dec"
          f" = {tokenized_lemmas(tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_plain} {tokenizer3A.sep_token} {sentence2_plain} {tokenizer3A.pad_token}'))}")
    print(f"lemma -> tokenize(#={len(tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_lemma} {tokenizer3A.sep_token} {sentence2_lemma} {tokenizer3A.pad_token}'))}) -> tokens_to_ids -> dec"
          f" = {tokenized_lemmas(tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_lemma} {tokenizer3A.sep_token} {sentence2_lemma} {tokenizer3A.pad_token}'))}")
    print(f"morps -> tokenize(#={len(tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_morps} {tokenizer3A.sep_token} {sentence2_morps} {tokenizer3A.pad_token}'))}) -> tokens_to_ids -> dec"
          f" = {tokenized_lemmas(tokenizer3A.tokenize(f'{tokenizer3A.cls_token} {sentence1_morps} {tokenizer3A.sep_token} {sentence2_morps} {tokenizer3A.pad_token}'))}")
