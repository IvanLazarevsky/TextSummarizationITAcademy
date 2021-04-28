import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class GPTHeadlineDataset(Dataset):
    def __init__(self, gpt_tokenizer, summaries, contents, max_input_length=1000, max_summary_length=256):
        self.encoded = []
        self.segments = []
        vocab = gpt_tokenizer.get_vocab()
        self.eos_index = vocab['</s>']
        self.bos_index = vocab['<s>']
        self.pad_index = vocab['<pad>']
        self.summary_lengths = []
        self.content_lengths = []


        for summary, content in tqdm(zip(summaries, contents)):
            summary_bpe = [self.eos_index]
            encoded_summary = gpt_tokenizer.encode(summary)
            self.summary_lengths.append(len(encoded_summary))
            summary_bpe.extend(encoded_summary)
            summary_bpe.append(self.eos_index)

            content_bpe = [self.bos_index]
            content_bpe.extend(gpt_tokenizer.encode(content))
            self.content_lengths.append(len(content_bpe) - 1)
            content_segment = content_bpe[:max_input_length]
            text_bpe = content_segment + summary_bpe[:max_summary_length]
            segment_codes = [0] * len(content_segment) + [1] * (len(text_bpe) - len(content_segment))
            self.segments.append(segment_codes)
            self.encoded.append(torch.tensor(text_bpe, dtype=torch.long))

    def collate(self, encoded_texts_and_segments):
        encoded_texts = []
        segments = []
        for t,s in encoded_texts_and_segments:
            encoded_texts.append(t)
            segments.append(s)
        text_tensor = torch.nn.utils.rnn.pad_sequence(encoded_texts, batch_first=True, padding_value=self.pad_index)
        segments_tensor = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True, padding_value=self.pad_index)
        labels_tensor = text_tensor.clone()

        labels_tensor[labels_tensor == self.pad_index] = -100
        return text_tensor, labels_tensor, segments_tensor

    def __getitem__(self, index: int):
        return self.encoded[index], torch.tensor(self.segments[index])

    def __len__(self) -> int:
        return len(self.encoded)
