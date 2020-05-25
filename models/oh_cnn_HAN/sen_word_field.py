from collections import Counter, OrderedDict
from itertools import chain
import six
import torch
from tqdm import tqdm

from torchtext.data.dataset import Dataset
from torchtext.data import Field

class SentenceWord_field(Field):

    def __init__(self, nesting_field, use_vocab=True, init_token=None, eos_token=None,
                 fix_length=None, tensor_type=torch.LongTensor, preprocessing=None,
                 postprocessing=None, tokenize=lambda s: s.split(),
                 include_lengths=False, pad_token='<pad>',
                 pad_first=False, truncate_first=False, vae_struct=False):
  
        if nesting_field.include_lengths:
            raise ValueError('nesting field cannot have include_lengths=True')

        if nesting_field.sequential:
            pad_token = nesting_field.pad_token
        super(SentenceWord_field, self).__init__(
            use_vocab=use_vocab,
            init_token=init_token,
            eos_token=eos_token,
            fix_length=fix_length,
            tensor_type=tensor_type,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            lower=nesting_field.lower,
            tokenize=tokenize,
            batch_first=True,
            pad_token=pad_token,
            unk_token=nesting_field.unk_token,
            pad_first=pad_first,
            truncate_first=truncate_first,
            include_lengths=include_lengths
        )
        self.nesting_field = nesting_field
        # in case the user forget to do that
        self.nesting_field.batch_first = True
        self.vae_struct = vae_struct

    def preprocess(self, xs):
        """Preprocess a single example.

        Firstly, tokenization and the supplied preprocessing pipeline is applied. Since
        this field is always sequential, the result is a list. Then, each element of
        the list is preprocessed using ``self.nesting_field.preprocess`` and the resulting
        list is returned.

        Arguments:
            xs (list or str): The input to preprocess.

        Returns:
            list: The preprocessed list.
        """
        return [self.nesting_field.preprocess(x)
                for x in super(SentenceWord_field, self).preprocess(xs)]

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        If ``self.nesting_field.sequential`` is ``False``, each example in the batch must
        be a list of string tokens, and pads them as if by a ``Field`` with
        ``sequential=True``. Otherwise, each example must be a list of list of tokens.
        Using ``self.nesting_field``, pads the list of tokens to
        ``self.nesting_field.fix_length`` if provided, or otherwise to the length of the
        longest list of tokens in the batch. Next, using this field, pads the result by
        filling short examples with ``self.nesting_field.pad_token``.

        Example:
            >>> import pprint
            >>> pp = pprint.PrettyPrinter(indent=4)
            >>>
            >>> nesting_field = Field(pad_token='<c>', init_token='<w>', eos_token='</w>')
            >>> field = SentenceWord_field(nesting_field, init_token='<s>', eos_token='</s>')
            >>> minibatch = [
            ...     [list('john'), list('loves'), list('mary')],
            ...     [list('mary'), list('cries')],
            ... ]
            >>> padded = field.pad(minibatch)
            >>> pp.pprint(padded)
            [   [   ['<w>', '<s>', '</w>', '<c>', '<c>', '<c>', '<c>'],
                    ['<w>', 'j', 'o', 'h', 'n', '</w>', '<c>'],
                    ['<w>', 'l', 'o', 'v', 'e', 's', '</w>'],
                    ['<w>', 'm', 'a', 'r', 'y', '</w>', '<c>'],
                    ['<w>', '</s>', '</w>', '<c>', '<c>', '<c>', '<c>']],
                [   ['<w>', '<s>', '</w>', '<c>', '<c>', '<c>', '<c>'],
                    ['<w>', 'm', 'a', 'r', 'y', '</w>', '<c>'],
                    ['<w>', 'c', 'r', 'i', 'e', 's', '</w>'],
                    ['<w>', '</s>', '</w>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>', '<c>', '<c>']]]

        Arguments:
            minibatch (list): Each element is a list of string if
                ``self.nesting_field.sequential`` is ``False``, a list of list of string
                otherwise.

        Returns:
            list: The padded minibatch. or (padded, sentence_lens, word_lengths)
        """
        minibatch = list(minibatch)
        if not self.nesting_field.sequential:
            return super(SentenceWord_field, self).pad(minibatch)

        # Save values of attributes to be monkeypatched
        old_pad_token = self.pad_token
        old_init_token = self.init_token
        old_eos_token = self.eos_token
        old_fix_len = self.nesting_field.fix_length
        # Monkeypatch the attributes
        if self.nesting_field.fix_length is None:
            max_len = max(len(xs) for ex in minibatch for xs in ex)
            fix_len = max_len + 2 - (self.nesting_field.init_token,
                                     self.nesting_field.eos_token).count(None)
            self.nesting_field.fix_length = fix_len
        self.pad_token = [self.pad_token] * self.nesting_field.fix_length
        if self.init_token is not None:
            # self.init_token = self.nesting_field.pad([[self.init_token]])[0]
            self.init_token = [self.init_token]
        if self.eos_token is not None:
            # self.eos_token = self.nesting_field.pad([[self.eos_token]])[0]
            self.eos_token = [self.eos_token]
        # Do padding
        old_include_lengths = self.include_lengths
        self.include_lengths = True
        self.nesting_field.include_lengths = True
        padded, sentence_lengths = super(SentenceWord_field, self).pad(minibatch)
        padded_with_lengths = [self.nesting_field.pad(ex) for ex in padded]
        word_lengths = []
        final_padded = []
        max_sen_len = len(padded[0])
        if self.vae_struct:
            decoder_arrs = minibatch
        for (pad, lens), sentence_len in zip(padded_with_lengths, sentence_lengths):
            if sentence_len == max_sen_len:
                lens = lens
                pad = pad
            elif self.pad_first:
                lens[:(max_sen_len - sentence_len)] = (
                    [0] * (max_sen_len - sentence_len))
                pad[:(max_sen_len - sentence_len)] = (
                    [self.pad_token] * (max_sen_len - sentence_len))
            else:
                lens[-(max_sen_len - sentence_len):] = (
                    [0] * (max_sen_len - sentence_len))
                pad[-(max_sen_len - sentence_len):] = (
                    [self.pad_token] * (max_sen_len - sentence_len))
            word_lengths.append(lens)
            final_padded.append(pad)
        padded = final_padded

        # Restore monkeypatched attributes
        self.nesting_field.fix_length = old_fix_len
        self.pad_token = old_pad_token
        self.init_token = old_init_token
        self.eos_token = old_eos_token
        self.include_lengths = old_include_lengths
        if self.include_lengths:
            return padded, sentence_lengths, word_lengths
        if self.vae_struct:
            return padded, decoder_arrs
        return padded

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for nesting field and combine it with this field's vocab.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for the nesting field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources.extend(
                    [getattr(arg, name) for name, field in arg.fields.items()
                     if field is self]
                )
            else:
                sources.append(arg)

        flattened = []
        for source in sources:
            flattened.extend(source)
        old_vectors = None
        old_unk_init = None
        old_vectors_cache = None
        if "vectors" in kwargs.keys():
            old_vectors = kwargs["vectors"]
            kwargs["vectors"] = None
        if "unk_init" in kwargs.keys():
            old_unk_init = kwargs["unk_init"]
            kwargs["unk_init"] = None
        if "vectors_cache" in kwargs.keys():
            old_vectors_cache = kwargs["vectors_cache"]
            kwargs["vectors_cache"] = None
        # just build vocab and does not load vector
        self.nesting_field.build_vocab(*flattened, **kwargs)
        super(SentenceWord_field, self).build_vocab()
        self.vocab.extend(self.nesting_field.vocab)
        self.vocab.freqs = self.nesting_field.vocab.freqs.copy()
        if old_vectors is not None:
            self.vocab.load_vectors(old_vectors,
                                    unk_init=old_unk_init, cache=old_vectors_cache)

        self.nesting_field.vocab = self.vocab

    def numericalize(self, arrs, device=None):
        """Convert a padded minibatch into a variable tensor.

        Each item in the minibatch will be numericalized independently and the resulting
        tensors will be stacked at the first dimension.

        Arguments:
            arr (List[List[str]]): List of tokenized and padded examples.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        numericalized = []
        self.nesting_field.include_lengths = False
        if self.include_lengths:
            arrs, sentence_lengths, word_lengths = arrs
        
        if self.vae_struct:
            arrs, decoder_arrs = arrs
            decoder_batch = []
            for arr in decoder_arrs:
                numericalized_ex = self.nesting_field.numericalize(arr, device=device)
                decoder_batch.append(numericalized_ex)
         

        for arr in arrs:
            numericalized_ex = self.nesting_field.numericalize(
                arr, device=device)
            numericalized.append(numericalized_ex)
        padded_batch = torch.stack(numericalized)

        self.nesting_field.include_lengths = True
        if self.include_lengths:
            sentence_lengths = \
                torch.tensor(sentence_lengths, dtype=self.dtype, device=device)
            word_lengths = torch.tensor(word_lengths, dtype=self.dtype, device=device)
            return (padded_batch, sentence_lengths, word_lengths)

        if self.vae_struct:
            return (padded_batch, decoder_batch)
        
        return padded_batch