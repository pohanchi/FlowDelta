# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""
#import pdb
from __future__ import absolute_import, division, print_function
import pdb
import argparse
import logging
import os
import random
import glob
import json
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME,ALBertConfig, ALBertForQuestionAnswering, ALbertTokenizer)

from transformers import AdamW, WarmupLinearSchedule, Lamb

import wandb
import run_quac
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)
from utils_quac_evaluate import EVAL_OPTS, main as evaluate_on_squad

# os.environ['WANDB_SILENT'] = "True"
# os.environ['WANDB_HOST'] = "GiveATry"
# os.environ['WANDB_MODE'] = "dryrun"

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'albert':(ALBertConfig,ALBertForQuestionAnswering,ALbertTokenizer),
}




def set_seed(args):
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if args['n_gpu'] > 0:
        torch.cuda.manual_seed_all(args['seed'])

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(args, train_dataloader, model, tokenizer,summary):
    """ Train the model """
    args['train_batch_size'] = args['per_gpu_train_batch_size'] * max(1, args['n_gpu'])

    if args['max_steps'] > 0:
        t_total = args['max_steps']
        args['num_train_epochs'] = args['max_steps'] // (len(train_dataloader) // args['gradient_accumulation_steps']) + 1
    else:
        t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = Lamb(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=t_total)
    if args['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])

    # multi-gpu training (should be after apex fp16 initialization)
    if args['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args['local_rank'] != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['local_rank']],
                                                          output_device=args['local_rank'],
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Instantaneous batch size per GPU = %d", args['per_gpu_train_batch_size'])
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args['train_batch_size'] * args['gradient_accumulation_steps'] * (torch.distributed.get_world_size() if args['local_rank'] != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch", disable=args['local_rank'] not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
    target = json.load(open(args['predict_file']))['data']
    target_dict = {}
    for p in target:
        for par in p['paragraphs']:
            p_id = par['id']
            qa_list = par['qas']
            for qa in qa_list:
                q_idx = qa['id']
                val_spans = [anss['text'] for anss in qa['answers']]
                target_dict[q_idx] = val_spans

    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args['local_rank'] not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # pdb.set_trace()
            batch = tuple(t.to(args['device']) for t in batch)
            inputs = {'input_ids'      : batch[0],
                      'attention_mask' : batch[1],
                      'token_type_ids' : batch[2],
                      'context_feature': batch[3],
                      'start_positions': batch[4],
                      'end_positions'  : batch[5],
                      'is_impossible':   batch[6]
                      }   

            loss = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            if args['n_gpu'] > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                if args['fp16']:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

 
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args['local_rank'] in [-1, 0] and args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    # Log metrics
                    if args['local_rank'] == -1 and args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        score = evaluate(args, model, tokenizer,target_dict)
                        tr_loss/=args['logging_steps']s
                        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
                        tr_loss = 0
                        # wandb.log({'training_loss':tr_loss},step=global_step)
                        # wandb.log({'eval_f1':score},step=global_step)
                        summary.add_scalar('eval_F1',score,global_step)  
                        logging.info("Evaluate F1 score {} on global_step {}".format(score,global_step))

                if args['local_rank'] in [-1, 0] and args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args['max_steps'] > 0 and global_step > args['max_steps']:
                epoch_iterator.close()
                break
        if args['max_steps'] > 0 and global_step > args['max_steps']:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, evaluator,prefix=""):
    eval_examples,eval_features = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(args['output_dir']) and args['local_rank'] in [-1, 0]:
        os.makedirs(args['output_dir'])

    args['eval_batch_size'] = args['per_gpu_eval_batch_size'] * max(1, args['n_gpu'])
    # Note that DistributedSampler samples randomly
    eval_dataloader = run_quac.make_dialog_tensors(eval_features, is_eval=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    all_results = []
    device = args['device']
    for input_ids, input_mask, segment_ids, context_feature, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        context_feature = context_feature.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits, batch_class_logits = model(input_ids, segment_ids, input_mask, context_feature)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            class_logits = batch_class_logits[i].detach().cpu().tolist()[0]
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(run_quac.RawResult(unique_id=unique_id,
                                            start_logits=start_logits,
                                            end_logits=end_logits,
                                            class_logits=class_logits))
    
    output_prediction_file = os.path.join(args['output_dir'], "predictions.json")
    output_nbest_file = os.path.join(args['output_dir'], "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args['output_dir'], "null_odds.json")
    pred, nbest_pred = run_quac.write_predictions(eval_examples, eval_features, all_results,
                                            args['n_best_size'], args['max_answer_length'],
                                            args['do_lower_case'], None,
                                            None, None, 
                                            args['verbose_logging'], True, args['null_score_diff_threshold'],ignore_write=True)
    if args['output_file'] is not None:
        # we hand craft the thrshold
        run_quac.write_quac(pred, nbest_pred, args['predict_file'], args['output_file'])
        f1 = run_quac.quac_performance(pred, nbest_pred, evaluator)

    return f1


def load_and_cache_examples(args, tokenizer, evaluate=False):
    if args['local_rank'] not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args['predict_file'] if evaluate else args['train_file']
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, "albert")).pop(),
        str(args['max_seq_length']),str(args['doc_stride']),str(args['max_query_length'])))
    if os.path.exists(cached_features_file) and not args['overwrite_cache']:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        examples = run_quac.read_quac_examples(input_file=input_file,
                                                is_training=True)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = run_quac.read_quac_examples(input_file=input_file,
                                                is_training=True)

        features = run_quac.convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args['max_seq_length'],
                                                doc_stride=args['doc_stride'],
                                                max_query_length=args['max_query_length'],
                                                is_training=True)
        if args['local_rank'] in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args['local_rank'] == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset

    return examples,features


def main():
    # os.environ['WANDB_DIR'] = args.output_dir
    # os.environ['WANDB_CONFIG_DIR'] = "../"
    # wandb.init()
    # wandb.config.update(args)

    args = yaml.full_load(open("config/argument.yaml","r"))
    config = args['ModelConfig']
    TrainArgs = args['TrainingArguments']
    TrainArgs['output_file'] = TrainArgs['output_dir']+"/"+TrainArgs['output_file']

    if os.path.exists(TrainArgs['output_dir']) and os.listdir(TrainArgs['output_dir']) and TrainArgs['do_train'] and not TrainArgs['overwrite_output_dir']:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(TrainArgs['output_dir']))

    # Setup CUDA, GPU & distributed training
    if TrainArgs['local_rank'] == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TrainArgs['n_gpu'] = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(TrainArgs['local_rank'])
        device = torch.device("cuda", TrainArgs['local_rank'])
        torch.distributed.init_process_group(backend='nccl')
        TrainArgs['n_gpu'] = 1
    TrainArgs['device'] = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if TrainArgs['local_rank'] in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    TrainArgs['local_rank'], device, TrainArgs['n_gpu'], bool(TrainArgs['local_rank'] != -1), TrainArgs['fp16'])

    # Set seed
    set_seed(TrainArgs)

    # Load pretrained model and tokenizer
    if TrainArgs['local_rank'] not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    TrainArgs['model_type'] = TrainArgs['model_type'].lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[TrainArgs['model_type']]
    config = config_class.from_json_file(TrainArgs['config_pretrain'])
    tokenizer = tokenizer_class(vocab_file="spm_model/30k-clean.model",do_lower_case=TrainArgs['do_lower_case'])
    
    #Load Model from Pretrained
    model = model_class(config=config)
    model.load_state_dict(torch.load(TrainArgs['model_dict_pretrain']),strict=False)

    if TrainArgs['local_rank'] == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(TrainArgs['device'])
    #pdb.set_trace()
    logger.info("Training/evaluation parameters %s", TrainArgs)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if TrainArgs['fp16'] is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if TrainArgs['fp16']:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if TrainArgs['do_train']:
        _,features = load_and_cache_examples(TrainArgs, tokenizer, evaluate=False)
        train_dataloader = run_quac.make_dialog_tensors(features,is_eval=False)
        writer = SummaryWriter(log_dir=TrainArgs['output_dir']+'/')
        global_step, tr_loss = train(TrainArgs, train_dataloader, model, tokenizer,writer)

    # Save the trained model and the tokenizer
    if TrainArgs['do_train'] and (TrainArgs['local_rank'] == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(TrainArgs['output_dir']) and TrainArgs['local_rank'] in [-1, 0]:
            os.makedirs(TrainArgs['output_dir'])

        logger.info("Saving model checkpoint to %s", TrainArgs['output_dir'])
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(TrainArgs['output_dir'])
        tokenizer.save_pretrained(TrainArgs['output_dir'])
        # wandb.save(os.path.join(output_dir, "training_args.bin"))
        
        # Good practice: save your training arguments together with the trained model
        torch.save(TrainArgs, os.path.join(TrainArgs['output_dir'], 'training_args.bin'))
        
        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(TrainArgs['output_dir'])
        tokenizer = tokenizer_class.from_pretrained(TrainArgs['output_dir'], do_lower_case=TrainArgs['do_lower_case'])
        model.to(TrainArgs['device'])

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    target = json.load(open(TrainArgs['predict_file']))['data']
    target_dict = {}
    for p in target:
        for par in p['paragraphs']:
            p_id = par['id']
            qa_list = par['qas']
            for qa in qa_list:
                q_idx = qa['id']
                val_spans = [anss['text'] for anss in qa['answers']]
                target_dict[q_idx] = val_spans

    if TrainArgs['do_eval'] and TrainArgs['local_rank'] in [-1, 0]:
        checkpoints = [TrainArgs['output_dir']]
        if TrainArgs['eval_all_checkpoints']:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(TrainArgs['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(TrainArgs['device'])

            # Evaluate
            score=evaluate(TrainArgs, model, tokenizer, target_dict,prefix=global_step)
            logger.info("Results F1: {} on step {}".format(score,global_step))

    return 


if __name__ == "__main__":
    main()
