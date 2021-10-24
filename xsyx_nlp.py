#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:03:52 2019

@author: fisher
"""
from bert_classifier import *

class Bert_Classifier(object):
    def __init__(self, init_checkpoint, vocab_file, bert_config_file, data_dir=None, output_dir='output',
                 do_lower_case=True, max_seq_length=128, save_checkpoints_steps=1000):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.init_checkpoint = init_checkpoint
        tokenization.validate_case_matches_checkpoint(do_lower_case, init_checkpoint)
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        self.run_config = tf.estimator.RunConfig(model_dir=output_dir, save_checkpoints_steps=save_checkpoints_steps)
        self.bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        tf.gfile.MakeDirs(output_dir)
        self.processor = DataFrameProcessor()
        self.label_list, self.label_map = self.processor.get_labels(data_dir)
        self.estimator = None
        
    def train(self, X=None, y=None, num_train_epochs=3, train_batch_size=32, learning_rate=5e-5, warmup_proportion=0.1, seed=None):
        examples = self.processor.get_train_examples(self.data_dir, data_col = 'productName', label_col = 'cateName')
        features = convert_examples_to_features(examples, self.label_map, self.max_seq_length, self.tokenizer)
        train_input_fn = input_fn_builder(features, seq_length=self.max_seq_length, 
                                          is_training=True, drop_remainder=True, batch_size=train_batch_size, seed=seed)

        num_train_steps = int(len(examples) / train_batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        model_fn = model_fn_builder(self.bert_config, len(self.label_list), self.init_checkpoint, learning_rate,
                                    num_train_steps, num_warmup_steps)
        self.estimator = tf.estimator.Estimator(model_fn=model_fn, config=self.run_config)
        self.estimator.train(train_input_fn, max_steps=num_train_steps)
    
    def evaluate(self, X=None, y=None, eval_batch_size=8):
        if self.estimator is None:
            model_fn = model_fn_builder(self.bert_config, len(self.label_list), self.init_checkpoint)
            self.estimator = tf.estimator.Estimator(model_fn=model_fn, config=self.run_config)
        if X is None or y is None:
            examples = self.processor.get_dev_examples(self.data_dir, 'val.csv', data_col = 'productName', label_col = 'cateName')
        else:
            examples = self.processor.create_examples(X, y)
        features = convert_examples_to_features(examples, self.label_map, self.max_seq_length, self.tokenizer)
        eval_input_fn = input_fn_builder(features, seq_length=self.max_seq_length, 
                                          is_training=False, drop_remainder=False, batch_size=eval_batch_size)
        result = self.estimator.evaluate(eval_input_fn)
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print("  %s = %s", key, str(result[key]))
        return result
    
    def predict(self, X=None, predict_batch_size=8):
        if X is None:
            #X = ['牛肉','猪肉']
            examples = self.processor.get_test_examples(self.data_dir, data_col = 'productName')
        else:
            examples = self.processor.create_examples(X)
        features = convert_examples_to_features(examples, self.label_map, self.max_seq_length, self.tokenizer)
        predict_input_fn = input_fn_builder(features, seq_length=self.max_seq_length, 
                                          is_training=False, drop_remainder=False, batch_size=predict_batch_size)
        if self.estimator is None:
            model_fn = model_fn_builder(self.bert_config, len(self.label_list), self.init_checkpoint)
            self.estimator = tf.estimator.Estimator(model_fn=model_fn, config=self.run_config)
        result = self.estimator.predict(predict_input_fn)
        print("***** Predict results *****")
        rs = []
        for (i, prediction) in enumerate(result):
            probabilities = prediction["probabilities"]
            if i >= len(examples):
                break
            print(probabilities)
            print(prediction["predicted_labels"])
            print(self.label_list[prediction["predicted_labels"]])
            rs.append(self.label_list[prediction["predicted_labels"]])
        return rs
