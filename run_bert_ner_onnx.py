#《Fast Automated Text Anonymization with ONNX & HuggingFace》
import json
import os
import os.path

from dataclasses import dataclass

import numpy as np

from typing import Dict, Tuple, List, Any, Union

from time import perf_counter
#import nltk
from onnxruntime import (
    InferenceSession,
    get_all_providers,
    SessionOptions,
    GraphOptimizationLevel,
)
import torch 
from transformers import BertTokenizerFast


@dataclass
class NerEntity:
	entity_group: str
	word: str
	score: float
	start: int
	end: int


class NerModel:
	def __init__(
	    self,
	    tokenizer,
	    model_file_path: str,
	    ids_to_labels_path: str,
	    provider: str = "CPUExecutionProvider",
	):
	    """This Nodel class is used to make NER (named-entity-recognition)
	    predictions on a text string.

	    The text string can consist of an arbitrary number of sentences.
	    The text is split into sentences using a statics
	    based sentence segmention method provided by the nltk package.
	    This allows the model to make predictions on very
	    long text strings without running into constraints.
	    This model class does the following steps:
	    1. split input text string into sentences
	    2. tokenize the text with a pretrained tokenizer specified for the given onnx model
	    3. use the tokenizer output to map original text to the output tokens (one word can be split into multiple
	    tokens)
	    4. prepare tokenized input for onnx model --> create dictionary with needed input keys
	    (information is stored in onnx model object itself)
	    5. make a prediction
	    6. get labels & confidence score per label
	    7. iterate over each sentence of the input to identify entites of interest -> stored them in a list of
	    NerEntity objects
	    Attributes:
	        model_file_path: local path to the model.onnx file
	        token_path: local path to the tokenizer_config.json and tokenizer.json file
	        provider: defines how the onnx model is executed (using COU or GPU --> CudaExecutionProvider)
	    """
	    # prepare onnx model for runtime
	    start_initializing = perf_counter()
	    self.model = self._initialize_model_for_runtime_(
	        model_path=model_file_path, provider=provider
	    )
	    print(
	        f"time to run model initialization: {perf_counter() - start_initializing}s"
	    )
	    
	    # get the dictionary keys that are requested by the model and outputed the tokenizer        
	    self.model_input_feed_keys = [node.name for node in self.model.get_inputs()]

	    # initialize tokenizer
	    self.tokenizer = tokenizer
	    
	    # download punkt package to use the nltk tokenizer, which splits sentences on a statistical basis        
	    #nltk.download("punkt")
	    
	    #self.sentencizer = nltk.load("tokenizers/punkt/german.pickle")

	    # get model config --> model output classes
	    self.id2label, self.zero_class = self.get_model_config(
	        ids_to_labels_path = ids_to_labels_path
	    )

	@staticmethod
	def get_model_config(ids_to_labels_path: str) -> Tuple[Dict[int, str], int]:
		id2label = torch.load(ids_to_labels_path)
		if "O" in id2label.values():
			zero_class = int(list(id2label.keys())[list(id2label.values()).index("O")])
		else:
			raise ValueError("model config is incorrect")

		return id2label, zero_class

	@staticmethod
	def _initialize_model_for_runtime_(
	    model_path: str, provider: str
	) -> InferenceSession:
	    """For optimal inference performance we are not using onnx itself but
	    onnx runtime.

	    This library allows for some optimizations specified for
	    the given hardware. This includes the number of threads to be used.
	    Attributes:
	        model_path: local path to the model.onnx file
	        provider: information about the hardware that will being used for inference
	        a session object, which is basically the optimized representation of the onnx model
	    """
	    assert (
	        provider in get_all_providers()
	    ), f"provider {provider} not found, {get_all_providers()}"

	    # Few properties that might have an impact on performances (provided by MS)
	    options = SessionOptions()
	    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

	    # Load the model as a graph and prepare the CPU backend
	    session = InferenceSession(model_path, options, providers=[provider])
	    session.disable_fallback()

	    return session

	def predict(self, input: str, sample_size: int = 2) -> Tuple[List[Union[List[NerEntity], List[Any]]], Any]:
	    """
	    This function receives a string input. The input is split into sentences and reorganised into subsamples 
	    to balance speed and performance. This behaviour is controlled by the sample_size parameter, which defines 
	    how many sentences are merged back together. A value of 2 has shown to keep the same accuracy by speeding 
	    up the process for longer texts at the same time. This parameter should be be tuned based on your data
	    Attributes:
	        input: text string 
	        sample_size: integer that defines how many sentences are merged back together
	    """
	    # 1. split input into sentences/sequences
	    input_split = [input] #self.sentencizer.tokenize(input)
	    
	    # merge sentence bag together to speed up process while retaining accuracy. This is a process that 
	    # needs to be optimised through experimentation    
	    input_split_merge = []
	    for i in range(0, len(input_split), sample_size):
	        if i + sample_size < len(input_split):
	            input_split_merge.append(" ".join(sent for sent in input_split[i: i + sample_size]))
	        else:
	            input_split_merge.append(" ".join(sent for sent in input_split[i:]))

	    num_samples = len(input_split_merge)

	    if num_samples == 0:
	        return [], input_split_merge

	    # 2. tokenize input
	    token_output = self.tokenizer(input_split_merge, return_offsets_mapping=True)

	    # 3. get word2token & word_ids
	    word2token = self._map_token_output_to_sentence_(
	        token_output, input_split_merge
	    )
	    word_ids = self._map_word_id_to_token_id_(token_output)

	    # 4. prepare model_inputs for onnx --> make it compatible
	    inputs_onnx, sentence_lengths = self._prepare_onnx_input_(
	        token_output, num_samples, self.model_input_feed_keys
	    )

	    # 5. make prediction
	    prediction = np.zeros(
	        (num_samples, np.max(sentence_lengths), len(self.id2label))
	    )

	    # iterate over each sentence is faster than batch for some reason
	    for i, inputs in enumerate(inputs_onnx):
	        prediction[i, : sentence_lengths[i], :] = np.squeeze(
	            self.model.run(None, inputs)[-1]
	        )

	    # 6. get labels
	    labels = np.argmax(prediction, axis=-1)
	    labels = [
	        labels[sentence, : sentence_lengths[sentence]]
	        for sentence in range(labels.shape[0])
	    ]

	    # 7. get confidence score
	    conf_scores = [
	        self._softmax_(prediction[sentence, : sentence_lengths[sentence]])
	        for sentence in range(prediction.shape[0])
	    ]

	    # 8. create list of entities
	    # keep only words of interest --> not predicted as None class 8
	    entities = []

	    for w, word_id in enumerate(word_ids):
	        label = labels[w]
	        conf_score = conf_scores[w]
	        w2t = word2token[w]
	        # 7. create list of entities
	        # keep only words of interest --> not predicted as None class 8
	        word_ids_of_interest = np.asarray([np.mean(label[idx]) for idx in word_id])
	        word_ids_of_interest = np.argwhere(word_ids_of_interest != self.zero_class)

	        # assign entities to word with prediction other than None class
	        if len(word_ids_of_interest) > 0:
	            entities.append(
	                self._assign_entities_(
	                    word_ids_of_interest=word_ids_of_interest,
	                    labels=label,
	                    conf_score=conf_score,
	                    word_ids=word_id,
	                    word2token=w2t,
	                    zero_class=self.zero_class,
	                )
	            )
	        else:
	            entities.append([])

	    return entities, input_split_merge

	@staticmethod
	def _map_token_output_to_sentence_(
	    token_output, input: List[str]
	) -> List[Dict[int, Tuple[str, Tuple[Any, Any]]]]:
	    """
	    This function will map the token index to the string of the original
	    text.
	    Attributes:
	        token_output: Encoding object --> output of the pretrained tokenizer
	        input: raw text input that was fed to the model to predict entities on

	    return dictionary with word2token mapping
	    """
	    word2token = []
	    for e, encoding in enumerate(token_output.encodings):
	        word2token.append(
	            {
	                t: (input[e][start:stop], (start, stop))
	                for t, (start, stop) in enumerate(encoding.offsets)
	                if encoding.special_tokens_mask[t] == 0
	            }
	        )
	        
	    """
	    the input text 'Manni Meier geht zu Obi in Köln Mühlheim.' is split into tokens as follows:
	    [{1: ('Mann', (0, 4)),
	      2: ('i', (4, 5)),
	      3: ('Meier', (6, 11)),
	      4: ('geht', (12, 16)),
	      5: ('zu', (17, 19)),
	      6: ('Ob', (20, 22)),
	      7: ('i', (22, 23)),
	      8: ('in', (24, 26)),
	      9: ('Köln', (27, 31)),
	      10: ('Mühl', (32, 36)),
	      11: ('heim', (36, 40))}]
	      
	    """
	    return word2token

	@staticmethod
	def _map_word_id_to_token_id_(token_output) -> List[np.ndarray]:
	    """This function will map the token id to the word id --> a single word
	    of the original input is split into multiple tokens, e.g.
	    Badheizkoerper --> 'Bad', 'he', 'iz', 'koerper'.

	    In the end we want to
	    make classifications based on the entire word not just token based.
	    Attributes:
	        token_output: list of integers, where each unique integer refers to a word in the raw input in the correct
	    order
	    return: a numpy array (beter for indexing) with a numpy array for each word in the raw input containing
	    integers referring to the token
	    """
	    out = []
	    for encoding in token_output.encodings:
	        word_ids = np.asarray(
	            [w if isinstance(w, int) else np.nan for w in encoding.word_ids]
	        )
	        word_ids = [
	            np.argwhere(word_ids == w).flatten()
	            for w in np.unique(word_ids)
	            if ~np.isnan(w)
	        ]
	        out_list = np.empty(len(word_ids), dtype=object)
	        out_list[:] = word_ids
	        out.append(out_list)
	        
	    """
	    The words of the input text 'Manni Meier geht zu Obi in Köln Mühlheim' are split in tokens like the following
	    [array([array([1, 2]), --> Manni is split into Mann & i, which are tokens Nr. 1 & 2
	            array([3]), 
	            array([4]), 
	            array([5]), 
	            array([6, 7]),
	            array([8]), 
	            array([9]), 
	            array([10, 11])], 
	            dtype=object)]
	    """
	    return out

	@staticmethod
	def _prepare_onnx_input_(
	    token_output, num_samples: int, model_input_feed_keys: List[str]
	) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
	    """This function will create the input for the onnx model to work with
	    batch input.
	    Therefore we create list of dictionaries, where each dictionary represents the input to the onnx model 
	    of a single sample.
	    We will also store the number of tokens for each sample. We will merge the model output of each sample 
	    in a single numpy array later for easier computation. To only use token classifications that are part of 
	    the sample itself we need to know the length, because the size of the array will be the size of the biggest sample
	    Attributes:
	        token_output: output of the pretrained tokenizer
	        num_samples: number of sentences/sequences found in the input by the nltk sentence splitter
	        dictionary with the correct input structure, length of each found sequence
	    """

	    lengths = []
	    for i in range(num_samples):
	        lengths.append(len(token_output["input_ids"][i]))

	    inputs_onnx = []

	    for i in range(num_samples):
	        inputs_onnx.append(
	            {
	                k: [np.asarray(token_output[k][i], dtype=int)]
	                for k in model_input_feed_keys
	            }
	        )

	    return inputs_onnx, lengths

	@staticmethod
	def _get_majority_voting_(pred: np.ndarray, zero_class: int) -> int:
	    """In the future, this function should output the majority class of all
	    predictions made for each token corresponding to the same word.

	    Right now, it seems to be valid to exclude all None classes (8 for germaner, 0 for germeval_14 --> look at
	    id2label)
	    Attributes:
	        pred: array with a label prediction (int) for each token of the same word
	    """
	    pred_left = pred[pred != zero_class]
	    if len(pred_left) >= 0.5 * len(pred):
	        if len(pred_left) > 1:
	            # TODO only keep the majority class
	            return pred[0]
	        else:
	            return pred[0]
	    else:
	        return zero_class

	@staticmethod
	def _softmax_(z: np.ndarray) -> np.ndarray:
	    """convert the raw model output to a softmax representation to use the
	    score as a confidence score.

	    Attributes:
	        z: raw model output as a 2d array (n, c) with n being the number of tokens and c the number of
	    output classes
	    return: softmax representation of the scoring input
	    """
	    s = np.max(z, axis=1)
	    s = s[:, np.newaxis]  # necessary step to do broadcasting
	    e_x = np.exp(z - s)
	    div = np.sum(e_x, axis=1)
	    div = div[:, np.newaxis]  # dito
	    return e_x / div
	
	def _assign_entities_(
	  self,
	  word_ids_of_interest: List[int],
	  labels: np.ndarray,
	  conf_score: np.ndarray,
	  word_ids: np.ndarray,
	  word2token: Dict[int, Tuple[str, Tuple[Any, Any]]],
	  zero_class: int,
	  ) -> List['NerEntity']:
	  """This function will assign the predicted entity to the corresponding
	  word by aggregating the predictions for each token of the word.

	  Attributes:
	  word_ids_of_interest: numpy array with integers pointing at the words that are not predicted as NONE classes
	  labels: numpy array with predicted labels per token
	  conf_score: softmax representation of raw output score --> is used as confidence score
	  word_ids: mapping of tokens and single words
	  word2token: mapping of the string corresponding to a token
	  return: list of predicted entities if found
	  """
	  # differentiate between a single word classification and multiple word
	  # classification
	  if len(word_ids_of_interest) == 1:
		  word_ids = word_ids[word_ids_of_interest][0][0]
		  if len(word_ids) > 1:
		      # word is based on more than one token
		      word_pred = self._get_majority_voting_(labels[word_ids], zero_class)
		      if word_pred == zero_class:
		          return []
		      word = "".join([word2token[i][0] for i in word_ids])
		      start_end = np.asarray(
		          [[word2token[i][1][0], word2token[i][1][1]] for i in word_ids]
		      )
		      score = np.mean(conf_score[word_ids, labels[word_ids]])
		      entity_group = self.id2label[word_pred].split("-")[-1]
		      start = np.min(start_end)
		      end = np.max(start_end)
		  else:
		      entity_group = self.id2label[labels[word_ids[0]]].split("-")[-1]
		      word = word2token[word_ids[0]][0]
		      start, end = word2token[word_ids[0]][1]
		      score = conf_score[word_ids[0], labels[word_ids[0]]]

		  if not word.isdigit():
		      return [
		          NerEntity(
		              entity_group=entity_group,
		              word=word,
		              score=score,
		              start=start,
		              end=end,
		          )
		      ]
		  else:
		      return []
	  else:
		  entities = []
		  word_ids_of_interest = np.squeeze(word_ids_of_interest)
		  word_ids = word_ids[word_ids_of_interest]
		  for idx_list in word_ids:
		      if len(idx_list) > 1:
		          # word is based on more than one token
		          word_pred = self._get_majority_voting_(labels[idx_list], zero_class)
		          if word_pred == self.zero_class:
		              continue
		          word = "".join([word2token[i][0] for i in idx_list])
		          start_end = np.asarray(
		              [[word2token[i][1][0], word2token[i][1][1]] for i in idx_list]
		          )
		          score = np.mean(conf_score[idx_list, labels[idx_list]])
		          entity_group = self.id2label[word_pred].split("-")[-1]
		          start = np.min(start_end)
		          end = np.max(start_end)
		      else:
		          entity_group = self.id2label[labels[idx_list[0]]].split("-")[-1]
		          word = word2token[idx_list[0]][0]
		          score = conf_score[idx_list[0], labels[idx_list[0]]]
		          start, end = word2token[idx_list[0]][1]

		      if not word.isdigit():
		          entities.append(
		              NerEntity(
		                  entity_group=entity_group,
		                  word=word,
		                  score=score,
		                  start=start,
		                  end=end,
		              )
		          )

		  return entities

# initialize NER Model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = NerModel(tokenizer, model_file_path="bert_ner_ft.onnx", ids_to_labels_path="ids_to_labels.pt")

# sample text
single_text = "As a surveyor, I want to be able to log into a system and enter information about the geospatial coordinates, building type, and characteristics for each survey I perform in a day, so that all of my survey data is stored in one central location and is easily accessible for analysis and reporting. Acceptance Criteria: The surveyor is able to log into the system using their unique credentials. The surveyor is able to enter the geospatial coordinates (latitude and longitude) of the building being surveyed."

entities, input_split = model.predict(single_text, sample_size=1)

print(input_split)

print(entities)

"""
[[
NerEntity(entity_group='Subject', word='surveyor', score=0.9897148007961543, start=5, end=13), 
NerEntity(entity_group='Processing', word='to', score=0.7579462088468345, start=33, end=35), 
NerEntity(entity_group='Processing', word='to', score=0.7793675146582706, start=340, end=342), 
NerEntity(entity_group='PII', word='their', score=0.5561912109946814, start=369, end=374), 
NerEntity(entity_group='PII', word='unique', score=0.5470437128623181, start=375, end=381), 
NerEntity(entity_group='Processing', word='to', score=0.6037882219821065, start=416, end=418)
]]
"""