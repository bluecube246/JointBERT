import logging
import os
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup, BertModel
from models import Base_BERT
import torch
import torch.nn
from tqdm.auto import tqdm, trange
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from datetime import datetime
import numpy as np
from data_loader import DataProcesser
from seqeval.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_labels_list = ['UNK'] + DataProcesser.read_file(os.path.join(args.data_dir, 'vocab.intent'))
        self.slot_labels_list = ['PAD', 'UNK'] + DataProcesser.read_file(os.path.join(args.data_dir, 'vocab.slot'))

        self.pad_token_label_id = 0

        #pretrained_model = BertModel.from_pretrained(args.pretrained_model)
        self.config = BertConfig.from_pretrained(args.pretrained_model)
        self.model = Base_BERT.from_pretrained(args.pretrained_model, config=self.config, args=args)
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=32)

        t_total = len(train_dataloader)

        # named_params = list(self.model.named_parameters())
        # named_params = list(filter(lambda p: p[1].requires_grad, named_params))
        no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_group_parameters = [
        #     {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        #     {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]

        optimizer_group_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_group_parameters, lr=5e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

        # Training step

        global_step = 0
        tr_loss = 0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.epochs), desc='Epoch')

        previous_model_val = 0.0
        start = datetime.now()
        best_epoch = -1

        for epoch in train_iterator:
            intent_loss_coef = 1.0
            slot_loss_coef = 1.0
            epoch_iterator = tqdm(train_dataloader, desc='Iteration', position=0, leave=True)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'intent_label_ids': batch[2],
                    'slot_label_ids': batch[3],
                    'token_type_ids': batch[4]
                }
                outputs = self.model(**inputs)

                loss = 0

                loss_lst, logit_lst = outputs
                total_loss, slot_loss, intent_loss = loss_lst
                slot_logits, intent_logits = logit_lst

                loss = intent_loss * intent_loss_coef
                loss += slot_loss * slot_loss_coef

                loss.backward()

                tr_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # prevent exploding gradients g <- g/||g||  if g > threshold
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                global_step += 1

                print(loss)

                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    print()
                    self.evaluate("dev")
                    self.evaluate("test")
                    self.evaluate("train")

                if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self.save_model()

                epoch_iterator.set_postfix({'loss': tr_loss / global_step})

    def evaluate(self, mode):
        if mode == "dev":
            dataset = self.dev_dataset
        elif mode == 'test':
            dataset = self.test_dataset
        elif mode == 'train':
            dataset = self.train_dataset

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_label_ids = None

        self.model.eval() # Gradients will not be updated

        for batch in tqdm(eval_dataloader, desc='Evaluating', position=0, leave=True):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'intent_label_ids': batch[2],
                    'slot_label_ids': batch[3],
                    'token_type_ids': batch[4]
                }

                outputs = self.model(**inputs)

                loss_lst, logit_lst = outputs
                total_loss, slot_loss, intent_loss = loss_lst
                slot_logits, intent_logits = logit_lst

                eval_loss += total_loss.mean().item()
            nb_eval_steps += 1

            # Intent predictions
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(out_intent_label_ids,
                                                 inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot predictions
            if slot_preds is None:
                slot_preds = slot_logits.detach().cpu().numpy()
                out_slot_label_ids = inputs['slot_label_ids'].detach().cpu().numpy()
            else:
                slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                out_slot_label_ids = np.append(out_slot_label_ids,
                                               inputs['slot_label_ids'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_labels_list)}
        out_slot_label_list = [[] for _ in range(out_slot_label_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_label_ids.shape[0])]

        for i in range(out_slot_label_ids.shape[0]):
            for j in range(out_slot_label_ids.shape[1]):
                if out_slot_label_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_label_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = self.compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        results.update(total_result)

        for key in sorted(results.keys()):
            logger.info(" %s = %s", key, str(results[key]))

        return results

    def compute_metrics(self, intent_preds, intent_labels, slot_preds, slot_labels):
        assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
        results = {}
        intent_result = self.get_intent_acc(intent_preds, intent_labels)
        slot_result = self.get_slot_metrics(slot_preds, slot_labels)

        results.update(intent_result)
        results.update(slot_result)

        return results

    def get_slot_metrics(self, preds, labels, mode="slot"):
        assert len(preds) == len(labels)
        return {
            mode + "_precision": precision_score(labels, preds),
            mode + "_slot_recall": recall_score(labels, preds),
            mode + "_slot_f1": f1_score(labels, preds)
        }

    def get_intent_acc(self, preds, labels, mode="intent"):
        acc = (preds == labels).mean()
        return {
            mode + "_acc": acc
        }

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)
