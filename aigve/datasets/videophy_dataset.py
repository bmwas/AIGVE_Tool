import os.path

from torch.utils.data import Dataset
import json
import re
import torch
from transformers import LlamaTokenizer
from core.registry import DATASETS
from functools import lru_cache

# Lazy import to avoid circular import
@lru_cache(maxsize=1)
def lazy_import_mplug_owl():
    import importlib
    extract_features = importlib.import_module(
        "aigve.metrics.multi_aspect_metrics.videophy.mplug_owl_video"
    )
    return extract_features.MplugOwlImageProcessor, extract_features.MplugOwlProcessor

@DATASETS.register_module()
class VideoPhyDataset(Dataset):
    def __init__(self, data_path, video_root_path, hf_token, tokenizer=None, processor=None, max_length=2048, media_tokens=['<image>', '<|video|>'], hf_checkpoint='videophysics/videocon_physics'):
        """
        Args:
            data_path (str): Path to the data folder, it should be a json file
            tokenizer (Tokenizer): Tokenizer object
            processor (Processor): Processor object
            max_length (int): Maximum length of the input sequence
            media_tokens (list): List of media tokens
        """
        self.dataset = json.load(open(data_path))
        self.video_root_path = video_root_path

        self.hf_token = hf_token
        self.hf_checkpoint = hf_checkpoint
        self.max_length = max_length
        self.media_tokens = {k: -int(i + 1) for i, k in enumerate(media_tokens)}
        self.media_lengths = {'<image>': 1 + 64, '<|video|>': 1 + 64}
        self.bucket = {}


        # initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.hf_checkpoint, token=self.hf_token)

        MplugOwlImageProcessor, MplugOwlProcessor = lazy_import_mplug_owl()
        self.image_processor = MplugOwlImageProcessor.from_pretrained(self.hf_checkpoint)
        # initialize processor
        if processor is not None:
            self.processor = processor
        else:
            self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)

    def __len__(self) -> int:
        """
        Returns:
            int: Length of the dataset
        """
        return self.dataset['metainfo']['length']

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the dataset
        Returns:
            dict: Dictionary containing the video, text, video path and caption
        """
        data = self.dataset['dataset_list'][idx]
        videopath = os.path.join(self.video_root_path, data['video_path_pd'])
        caption = data['prompt_gt']
        # video_input = self.processor(videos=[videopath], num_frames=16, return_tensors='pt') # video_pixel_values
        video_input = self.processor(videos=[videopath], num_frames=32, return_tensors='pt')  # video_pixel_values
        text_input = self._extract_text_token_from_conversation(caption, self.max_length, idx)
        item = {'video': video_input, 'text': text_input, 'videopath': videopath, 'caption': caption}
        return item

    def _extract_text_token_from_conversation(self, data, max_length, index):
        """
        Extracts the text tokens from the conversation
        Args:
            data (str): Conversation
            max_length (int): Maximum length of the input sequence
            index (int): Index of the dataset
        """
        # output enc_chunk
        enc_chunk = []

        if self.tokenizer.bos_token_id > 0:
            prompt_chunk = [self.tokenizer.bos_token_id]
        else:
            prompt_chunk = []

        # conversation = data["completion"]
        conversation = data

        # For Text only data
        if all([media_token not in conversation for media_token in self.media_tokens.keys()]):
            pattern = '|'.join(map(re.escape, ['AI: ', '\nHuman: ']))
            chunk_strs = re.split(f'({pattern})', conversation)
            prompt_length = -1
            stop_flag = False
            for idx, chunk_str in enumerate(chunk_strs):
                if idx == 0:
                    enc_chunk = prompt_chunk + \
                                self.tokenizer(chunk_str, add_special_tokens=False)[
                                    'input_ids']
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_strs[idx - 1] == 'AI: ':
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False)['input_ids']
                        if enc_length + len(curr_chunk) >= max_length:
                            curr_chunk = curr_chunk[:max_length - enc_length]
                            stop_flag = True
                        curr_chunk += [self.tokenizer.eos_token_id]
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [1] * len(curr_chunk)
                    else:
                        curr_chunk = self.tokenizer(
                            chunk_str, add_special_tokens=False)['input_ids']
                        if enc_length + len(curr_chunk) >= max_length + 1:
                            curr_chunk = curr_chunk[:max_length + 1 - enc_length]
                            stop_flag = True
                        enc_length += len(curr_chunk)
                        enc_chunk += curr_chunk
                        label_chunk += [0] * len(curr_chunk)
                    if stop_flag:
                        break

        # For Image-Text Data
        else:
            enc_length = 0
            prompt_length = -2
            pattern = '|'.join(
                map(re.escape, list(self.media_tokens.keys()) + ['AI: ', '\nHuman: ']))
            chunk_strs = re.split(f'({pattern})', conversation)
            chunk_strs = [x for x in chunk_strs if len(x) > 0]
            for idx, chunk_str in enumerate(chunk_strs):
                if enc_length >= max_length + 1:
                    break

                if idx == 0:
                    enc_chunk = prompt_chunk + \
                                self.tokenizer(chunk_str, add_special_tokens=False)[
                                    'input_ids']
                    enc_length = len(enc_chunk)
                    label_chunk = [0] * enc_length
                else:
                    if chunk_str in self.media_tokens:
                        # [CLS] + 256 + [EOS]
                        if enc_length + self.media_lengths[chunk_str] > max_length + 1:
                            break
                        else:
                            enc_chunk += [self.media_tokens[chunk_str]
                                          ] * self.media_lengths[chunk_str]
                            enc_length += self.media_lengths[chunk_str]
                            label_chunk += [0] * self.media_lengths[chunk_str]
                    else:

                        if chunk_strs[idx - 1] == 'AI: ':
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False)['input_ids']
                            if enc_length + len(curr_chunk) >= max_length:
                                curr_chunk = curr_chunk[:max_length - enc_length]
                            curr_chunk += [self.tokenizer.eos_token_id]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [1] * len(curr_chunk)
                        else:
                            curr_chunk = self.tokenizer(
                                chunk_str, add_special_tokens=False)['input_ids']
                            if enc_length + len(curr_chunk) >= max_length + 1:
                                curr_chunk = curr_chunk[:max_length +
                                                         1 - enc_length]
                            enc_length += len(curr_chunk)
                            enc_chunk += curr_chunk
                            label_chunk += [0] * len(curr_chunk)

        if enc_length < max_length + 1:
            padding_chunk = [self.tokenizer.pad_token_id] * \
                            (max_length + 1 - enc_length)
            padding_length = len(padding_chunk)
            label_chunk += [0] * (max_length + 1 - enc_length)
            enc_chunk = enc_chunk + padding_chunk
        else:
            padding_length = 0

        assert enc_length + padding_length == max_length + \
               1, (index, prompt_length, enc_length,
                   padding_length, max_length + 1)
        assert len(label_chunk) == max_length + \
               1, (len(label_chunk), max_length + 1)
        non_padding_mask = [1 if i < enc_length -
                                 1 else 0 for i in range(max_length)]

        enc_chunk = torch.tensor(enc_chunk).long()
        non_padding_mask = torch.tensor(non_padding_mask).long()
        prompt_mask = torch.tensor(label_chunk)[1:].long()
        prompt_length = torch.tensor([prompt_length]).long()

        # Create loss mask
        if all([media_token not in conversation for media_token in self.media_tokens.keys()]):
            non_media_mask = torch.ones_like(non_padding_mask).long()
        else:
            tmp_enc_chunk = enc_chunk.clone()
            tmp_enc_chunk[tmp_enc_chunk >= 0] = 1
            tmp_enc_chunk[tmp_enc_chunk < 0] = 0
            non_media_mask = torch.tensor(tmp_enc_chunk).long()
            non_media_mask = non_media_mask[1:].long()
        return {'input_ids': enc_chunk, "prompt_length": prompt_length, 'seq_length': enc_length,
                "non_padding_mask": non_padding_mask, 'non_media_mask': non_media_mask, 'prompt_mask': prompt_mask}



