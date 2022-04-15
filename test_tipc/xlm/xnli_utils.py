import logging
import math
import os

import numpy as np
import paddle

from xlm.utils import load_pickle

logger = logging.getLogger()

BOS_WORD = "<s>"
EOS_WORD = "</s>"
PAD_WORD = "<pad>"
UNK_WORD = "<unk>"

SPECIAL_WORD = "<special%i>"
SPECIAL_WORDS = 10

SEP_WORD = SPECIAL_WORD % 0
MASK_WORD = SPECIAL_WORD % 1
XNLI_LANGS = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]


class Dataset:
    def __init__(self, sent, pos, params):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.sent = sent
        self.pos = pos
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        # check number of sentences
        assert len(self.pos) == (self.sent == self.eos_index).sum()

        # # remove empty sentences
        # self.remove_empty_sentences()

        # sanity checks
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos)

    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert (len(self.pos) == (self.sent[self.pos[:, 1]] == eos).sum()
                )  # check sentences indices
        # assert self.lengths.min() > 0                                     # check empty sentences

    def batch_sentences(self, sentences):
        lengths = np.array([len(s) + 2 for s in sentences], dtype="int64")
        sent = (np.ones(
            shape=[lengths.max(), lengths.shape[0]],
            dtype="int64") * self.pad_index)
        sent[0] = self.eos_index
        for i, s in enumerate(sentences):
            if lengths[i] > 2:  # if sentence not empty
                sent[1:lengths[i] - 1, i] = s.astype(np.int64)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] > 0]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] <= max_len]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i too long sentences." %
                    (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        assert 0 <= a < b <= len(self.pos)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.pos = self.pos[a:b]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]

        # re-index
        min_pos = self.pos.min()
        max_pos = self.pos.max()
        self.pos -= min_pos
        self.sent = self.sent[min_pos:max_pos + 1]

        # sanity checks
        self.check()

    def get_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            pos = self.pos[sentence_ids]
            sent = [self.sent[a:b] for a, b in pos]
            sent = self.batch_sentences(sent)
            yield (sent, sentence_ids) if return_indices else sent

    def get_iterator(
            self,
            shuffle,
            group_by_size=False,
            n_sentences=-1,
            seed=None,
            return_indices=False, ):
        """
        Return a sentences iterator.
        """
        assert seed is None or shuffle is True and type(seed) is int
        rng = np.random.RandomState(seed)
        n_sentences = len(self.pos) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos)
        assert type(shuffle) is bool and type(group_by_size) is bool
        assert group_by_size is False or shuffle is True

        # sentence lengths
        lengths = self.lengths + 2

        # select sentences to iterate over
        if shuffle:
            indices = rng.permutation(len(self.pos))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind="mergesort")]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(
                indices, math.ceil(len(indices) * 1.0 / self.batch_size))
        else:
            batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [
                indices[bounds[i]:bounds[i + 1]]
                for i in range(len(bounds) - 1)
            ]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            rng.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum(
            [lengths[x].sum() for x in batches])
        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)


class ParallelDataset(Dataset):
    def __init__(self, sent1, pos1, sent2, pos2, params):

        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.batch_size = params.batch_size
        self.tokens_per_batch = params.tokens_per_batch
        self.max_batch_size = params.max_batch_size

        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

        # check number of sentences
        assert len(self.pos1) == (self.sent1 == self.eos_index).sum()
        assert len(self.pos2) == (self.sent2 == self.eos_index).sum()

        # remove empty sentences
        self.remove_empty_sentences()

        # sanity checks
        self.check()

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos1)

    def check(self):
        """
        Sanity checks.
        """
        eos = self.eos_index
        assert len(self.pos1) == len(
            self.pos2) > 0  # check number of sentences
        assert (len(self.pos1) == (self.sent1[self.pos1[:, 1]] == eos).sum()
                )  # check sentences indices
        assert (len(self.pos2) == (self.sent2[self.pos2[:, 1]] == eos).sum()
                )  # check sentences indices
        assert eos <= self.sent1.min(
        ) < self.sent1.max()  # check dictionary indices
        assert eos <= self.sent2.min(
        ) < self.sent2.max()  # check dictionary indices
        assert self.lengths1.min() > 0  # check empty sentences
        assert self.lengths2.min() > 0  # check empty sentences

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] > 0]
        indices = indices[self.lengths2[indices] > 0]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))
        self.check()

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len >= 0
        if max_len == 0:
            return
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] <= max_len]
        indices = indices[self.lengths2[indices] <= max_len]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i too long sentences." %
                    (init_size - len(indices)))
        self.check()

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        assert 0 <= a < b <= len(self.pos1)
        logger.info("Selecting sentences from %i to %i ..." % (a, b))

        # sub-select
        self.pos1 = self.pos1[a:b]
        self.pos2 = self.pos2[a:b]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]

        # re-index
        min_pos1 = self.pos1.min()
        max_pos1 = self.pos1.max()
        min_pos2 = self.pos2.min()
        max_pos2 = self.pos2.max()
        self.pos1 -= min_pos1
        self.pos2 -= min_pos2
        self.sent1 = self.sent1[min_pos1:max_pos1 + 1]
        self.sent2 = self.sent2[min_pos2:max_pos2 + 1]

        # sanity checks
        self.check()

    def get_batches_iterator(self, batches, return_indices):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        assert type(return_indices) is bool

        for sentence_ids in batches:
            if 0 < self.max_batch_size < len(sentence_ids):
                np.random.shuffle(sentence_ids)
                sentence_ids = sentence_ids[:self.max_batch_size]
            pos1 = self.pos1[sentence_ids]
            pos2 = self.pos2[sentence_ids]
            sent1 = self.batch_sentences([self.sent1[a:b] for a, b in pos1])
            sent2 = self.batch_sentences([self.sent2[a:b] for a, b in pos2])
            yield (sent1, sent2, sentence_ids) if return_indices else (sent1,
                                                                       sent2)

    def get_iterator(self,
                     shuffle,
                     group_by_size=False,
                     n_sentences=-1,
                     return_indices=False):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos1) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos1)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # sentence lengths
        lengths = self.lengths1 + self.lengths2 + 4

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos1))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(lengths[indices], kind="mergesort")]

        # create batches - either have a fixed number of sentences, or a similar number of tokens
        if self.tokens_per_batch == -1:
            batches = np.array_split(
                indices, math.ceil(len(indices) * 1.0 / self.batch_size))
        else:
            batch_ids = np.cumsum(lengths[indices]) // self.tokens_per_batch
            _, bounds = np.unique(batch_ids, return_index=True)
            batches = [
                indices[bounds[i]:bounds[i + 1]]
                for i in range(len(bounds) - 1)
            ]
            if bounds[-1] < len(indices):
                batches.append(indices[bounds[-1]:])

        # optionally shuffle batches
        if shuffle:
            np.random.shuffle(batches)

        # sanity checks
        assert n_sentences == sum([len(x) for x in batches])
        assert lengths[indices].sum() == sum(
            [lengths[x].sum() for x in batches])
        # assert set.union(*[set(x.tolist()) for x in batches]) == set(range(n_sentences))  # slow

        # return the iterator
        return self.get_batches_iterator(batches, return_indices)


def concat_batches(x1,
                   len1,
                   lang1_id,
                   x2,
                   len2,
                   lang2_id,
                   pad_idx=2,
                   eos_idx=1,
                   reset_positions=False):
    """
    Concat batches with different languages.
    """
    assert reset_positions is False or lang1_id != lang2_id
    lengths = len1 + len2
    if not reset_positions:
        lengths -= 1
    slen, bs = lengths.max().item(), lengths.shape[0]

    x = paddle.ones([slen, bs], dtype="int64") * pad_idx
    x[:len1.max().item()] = x1
    positions = paddle.tile(paddle.arange(slen)[:, None], (1, bs))

    langs = paddle.ones([slen, bs], dtype="int64") * lang1_id

    for i in range(bs):
        l1 = len1[i] if reset_positions else len1[i] - 1
        x[l1:l1 + len2[i], i] = x2[:len2[i], i]
        if reset_positions:
            positions[l1:, i] -= len1[i]
        langs[l1:, i] = lang2_id

    assert (x == eos_idx).astype("int64").sum().item() == (4 if reset_positions
                                                           else 3) * bs

    return (
        x.transpose([1, 0]),
        paddle.to_tensor(
            lengths, dtype="int64"),
        positions.transpose([1, 0]),
        langs.transpose([1, 0]), )


def truncate(x, lengths, max_len=256, eos_index=1):
    """
    Truncate long sentences.
    """
    if lengths.max().item() > max_len:
        x = x[:max_len].copy()
        lengths = lengths.copy()
        for i in range(len(lengths)):
            if lengths[i] > max_len:
                lengths[i] = max_len
                x[max_len - 1, i] = eos_index
    return x, lengths


def load_binarized(path, params):
    """
    Load a binarized dataset.
    """
    assert path.endswith(".pkl")
    assert os.path.isfile(path), path
    logger.info("Loading data from %s ..." % path)
    data = load_pickle(path)
    data = process_binarized(data, params)
    return data


def process_binarized(data, params):
    """
    Process a binarized dataset and log main statistics.
    """
    dico = data["dico"]
    assert ((data["sentences"].dtype == np.uint16) and (len(dico) < 1 << 16) or
            (data["sentences"].dtype == np.int32) and
            (1 << 16 <= len(dico) < 1 << 31))
    logger.info(
        "%i words (%i unique) in %i sentences. %i unknown words (%i unique) covering %.2f%% of the data."
        % (
            len(data["sentences"]) - len(data["positions"]),
            len(dico),
            len(data["positions"]),
            sum(data["unk_words"].values()),
            len(data["unk_words"]),
            100.0 * sum(data["unk_words"].values()) /
            (len(data["sentences"]) - len(data["positions"])), ))
    if params.max_vocab != -1:
        assert params.max_vocab > 0
        logger.info("Selecting %i most frequent words ..." % params.max_vocab)
        dico.max_vocab(params.max_vocab)
        data["sentences"][data["sentences"] >=
                          params.max_vocab] = dico.index(UNK_WORD)
        unk_count = (data["sentences"] == dico.index(UNK_WORD)).sum()
        logger.info("Now %i unknown words covering %.2f%% of the data." % (
            unk_count,
            100.0 * unk_count /
            (len(data["sentences"]) - len(data["positions"])), ))
    if params.min_count > 0:
        logger.info("Selecting words with >= %i occurrences ..." %
                    params.min_count)
        dico.min_count(params.min_count)
        data["sentences"][data["sentences"] >=
                          len(dico)] = dico.index(UNK_WORD)
        unk_count = (data["sentences"] == dico.index(UNK_WORD)).sum()
        logger.info("Now %i unknown words covering %.2f%% of the data." % (
            unk_count,
            100.0 * unk_count /
            (len(data["sentences"]) - len(data["positions"])), ))
    if (data["sentences"].dtype == np.int32) and (len(dico) < 1 << 16):
        logger.info(
            "Less than 65536 words. Moving data from int32 to uint16 ...")
        data["sentences"] = data["sentences"].astype(np.uint16)
    return data


def set_dico_parameters(params, data, dico):
    """
    Update dictionary parameters.
    """
    if "dico" in data:
        assert data["dico"] == dico
    else:
        data["dico"] = dico

    n_words = len(dico)
    bos_index = dico.index(BOS_WORD)
    eos_index = dico.index(EOS_WORD)
    pad_index = dico.index(PAD_WORD)
    unk_index = dico.index(UNK_WORD)
    mask_index = dico.index(MASK_WORD)
    if hasattr(params, "bos_index"):
        assert params.n_words == n_words
        assert params.bos_index == bos_index
        assert params.eos_index == eos_index
        assert params.pad_index == pad_index
        assert params.unk_index == unk_index
        assert params.mask_index == mask_index
    else:
        params.n_words = n_words
        params.bos_index = bos_index
        params.eos_index = eos_index
        params.pad_index = pad_index
        params.unk_index = unk_index
        params.mask_index = mask_index


class XNLIDataset:
    def __init__(self, params):
        self.params = params
        self.data = self.load_data(params)

    def load_data(self, params):
        data = {
            lang: {splt: {}
                   for splt in ["train", "valid", "test"]}
            for lang in XNLI_LANGS
        }
        label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        dpath = os.path.join(params.data_path, "XNLI")

        for splt in ["train", "valid", "test"]:
            for lang in XNLI_LANGS:
                # only English has a training set
                if splt == "train" and lang != "en":
                    del data[lang]["train"]
                    continue

                # load data and dictionary
                data1 = load_binarized(
                    os.path.join(dpath, "%s.s1.%s.pkl" % (splt, lang)), params)
                data2 = load_binarized(
                    os.path.join(dpath, "%s.s2.%s.pkl" % (splt, lang)), params)
                data["dico"] = data.get("dico", data1["dico"])

                # set dictionary parameters
                set_dico_parameters(params, data, data1["dico"])
                set_dico_parameters(params, data, data2["dico"])

                if splt == "train":
                    params.batch_size = params.train_batch_size
                else:
                    params.batch_size = params.eval_batch_size

                # create dataset
                data[lang][splt]["x"] = ParallelDataset(
                    data1["sentences"],
                    data1["positions"],
                    data2["sentences"],
                    data2["positions"],
                    params, )
                # load labels
                with open(
                        os.path.join(dpath, "%s.label.%s" % (splt, lang)),
                        "r") as f:
                    labels = [label2id[l.rstrip()] for l in f]
                data[lang][splt]["y"] = np.array(labels, dtype="int64")
                assert len(data[lang][splt]["x"]) == len(data[lang][splt]["y"])

        return data

    def get_iterator(self, splt, lang):
        assert splt in ["valid", "test"] or splt == "train" and lang == "en"
        return self.data[lang][splt]["x"].get_iterator(
            shuffle=(splt == "train"),
            group_by_size=self.params.group_by_size,
            return_indices=True, )
