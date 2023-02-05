from ..trainer import PPOConfig

imdb_config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    learning_rate=1.41e-5,
    log_with="wandb",
)

imdb_sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": imdb_config.forward_batch_size
}

ABC = "a"