from ..trainer import PPOConfig

config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    learning_rate=1.41e-5,
)

sent_kwargs = {
    "return_all_scores": True, # need to do this for output[1]["score"] to be the positive score
    #"top_k":None, # dont do this or the index of the score will not always be positive second
    "function_to_apply": "none",
    "batch_size": config.mini_batch_size
}

