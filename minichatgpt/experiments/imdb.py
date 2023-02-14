from ..trainer import PPOConfig

config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    learning_rate=1.41e-5,
)

sent_kwargs = {
    #"return_all_scores": True,
    "top_k":None,
    "function_to_apply": "none",
    "batch_size": config.forward_batch_size
}

