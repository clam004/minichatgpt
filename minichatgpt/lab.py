import os

import torch
from transformers import pipeline, AutoTokenizer

from .core import LengthSampler
from .languagemodels.modeling_value_head import AutoModelForCausalLMWithValueHead
from .trainer import PPOTrainer
from .processdata.build_dataset import build_dataset


class Lab():

    def __init__(self,
        config=None,
    ):
        self.config = config

    def set_generation_config(self,
        do_sample=True,
        output_min_length = 4,
        output_max_length = 16,
        pad_token_id=50256,
    ):
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

        self.generation_kwargs = {
            "min_length":-1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id, 
        }

        self.output_length_sampler = LengthSampler(output_min_length, output_max_length)

    def build_dataset(self,
        config = None,
        dataset_name="imdb", 
        input_min_text_length=2, 
        input_max_text_length=8,
    ):

        if config is None and self.config:
            config = self.config

        self.dataset = build_dataset(
            config,
            dataset_name, 
            input_min_text_length, 
            input_max_text_length,
        )

        return self.dataset 

    def init_policies_tokenizer(self, model_name=None):

        if model_name is None and self.config.model_name:
            model_name = self.config.model_name

        self.new_policy = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        self.old_policy = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.new_policy, self.old_policy, self.tokenizer

    def num_params_million(self, model):

        return sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6

    def init_ppo_trainer(self,
        config, 
        new_policy, 
        old_policy, 
        tokenizer, 
        dataset, 
        dataloader_collator,
    ):

        self.ppo_trainer = PPOTrainer(
            config, 
            new_policy, 
            old_policy,
            tokenizer, 
            dataset, 
            dataloader_collator=dataloader_collator,
        )

        self.batches_per_epoch = len(self.ppo_trainer.dataloader)

        self.device = self.ppo_trainer.accelerator.device
        if self.ppo_trainer.accelerator.num_processes == 1:
            # to avoid a `pipeline` bug
            self.device = 0 if torch.cuda.is_available() else "cpu"

        return self.ppo_trainer

    def init_reward_model(self,):

        if self.device:
            device = self.device
        else:
            device = 0 if torch.cuda.is_available() else "cpu"

        self.reward_model = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)




