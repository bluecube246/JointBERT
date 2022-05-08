import os
import torch
from torch.utils.data import TensorDataset


class InputExample(object):

    def __init__(self, words, intent_labels, slot_labels):
        self.words = words
        self.intent_labels = intent_labels
        self.slot_labels = slot_labels


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, intent_label_id, slot_labels_ids, token_type_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids
        self.token_type_ids = token_type_ids


class DataProcesser(object):

    def __init__(self, args):
        self.args = args
        # self.intent_labels = ['UNK'] + self.read_file(os.path.join(args.data_dir, args.task, 'vocab.intent'))
        # self.slot_labels = ['PAD', 'UNK'] + self.read_file(os.path.join(args.data_dir, args.task, 'vocab.slot'))
        self.intent_labels = self.read_file(os.path.join(args.data_dir, args.task, 'intent_label.txt'))
        self.slot_labels = self.read_file(os.path.join(args.data_dir, args.task, 'slot_label.txt'))

    @classmethod
    def read_file(cls, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def get_examples(self, intents, utterances, slots):

        examples = []
        for intent, utt, slot in zip(intents, utterances, slots):
            # utterance
            words = utt.lower().split()
            # intent
            intent_labels = self.intent_labels.index(intent) \
                if intent in self.intent_labels else self.intent_labels.index('UNK')
            # slot
            slot_labels = []
            for tag in slot.split():
                slot_labels.append(self.slot_labels.index(tag)
                                   if tag in self.slot_labels else self.slot_labels.index('UNK'))

            examples.append(InputExample(intent_labels=intent_labels,
                                         slot_labels=slot_labels,
                                         words=words))
        return examples

    def read_data(self, separator=":", lowercase=False, mode="train"):
        print('Reading source data ...')
        input_seqs = []
        tag_seqs = []
        class_labels = []
        line_num = -1

        with open(os.path.join(self.args.data_dir, self.args.task, mode), 'r') as f:
            for ind, line in enumerate(f):
                line_num += 1
                slot_tag_line, class_name = line.strip('\n\r').split(" <=> ")
                if slot_tag_line == "":
                    continue

                in_seq, tag_seq = [], []
                for item in slot_tag_line.split(' '):
                    tmp = item.split(separator)
                    assert len(tmp) >= 2
                    word, tag = separator.join(tmp[:-1]), tmp[-1]
                    in_seq.append(word)
                    tag_seq.append(tag)

                class_labels.append(class_name)
                input_seqs.append(' '.join(in_seq))
                tag_seqs.append(' '.join(tag_seq))

        return self.get_examples(class_labels, input_seqs, tag_seqs)

    def convert_to_ids(self, examples, max_seq_len, tokenizer, cls_token_segment_id=0, pad_token_segment_id=0,
                       pad_token_label_id=0, sequence_a_segment_id=0):
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        words = [example.words for example in examples]

        features = []
        for example in examples:
            tokens = []
            slot_label_ids = []
            for i, (word, slot_label) in enumerate(zip(example.words, example.slot_labels)):
                word_tokens = tokenizer.tokenize(word)
                tokens.extend(word_tokens)
                slot_label_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

            # if utt is > 48 trim to 48 2 tokens needed for [SEP] & [CLS]
            if len(tokens) > max_seq_len - 2:
                tokens = tokens[:(max_seq_len - 2)]
                slot_label_ids = slot_label_ids[:(max_seq_len - 2)]

            # Add [SEP] & [CLS]
            tokens = [cls_token] + tokens + [sep_token]
            token_type_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
            slot_label_ids = [pad_token_label_id] + slot_label_ids + [pad_token_label_id]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            attention_mask = [1] * len(input_ids)

            # Zero-pad
            zero_pad_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * zero_pad_length)
            attention_mask = attention_mask + ([0] * zero_pad_length)
            slot_label_ids = slot_label_ids + ([pad_token_label_id] * zero_pad_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * zero_pad_length)

            intent_label_id = int(example.intent_labels)

            features.append(InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                slot_labels_ids=slot_label_ids,
                intent_label_id=intent_label_id,
                token_type_ids=token_type_ids
            ))

        return features


def load_data(args, mode):
    data_processor = DataProcesser(args)

    examples = data_processor.read_data(mode=mode)
    features = data_processor.convert_to_ids(examples, args.max_seq_len, args.tokenizer)

    # print(features)

    # Convert to Tensors
    tensor_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    tensor_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    tensor_slot_label_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)
    tensor_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    tensor_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(tensor_input_ids, tensor_attention_mask, tensor_token_type_ids,
                            tensor_intent_label_ids, tensor_slot_label_ids)

    return dataset
