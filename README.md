# minichatgpt

Focused dissection of the implementation details of a small simplified self contained toy project demonstrating reinforcement learning from human feedback (RLHF) with special emphasis on connecting the equations describing proximal policy optimization to the lines of pytorch code that apply PPO to work with sequences, such as completing sentences so they end with a positive sentiment. We do this not by self-supervised or supervised learning, but rather, by generating text and learning from scores assigned to that text after it is generated, this is analogous to the way ChatGPT was trained using human scores of model generated answers to instructions. 
## Building Development Environments

### python virtual environment for data science

```console
you@you chat-api % python3 -m venv venv
you@you chat-api % source venv/bin/activate
(venv) you@you chat-api % pip install --upgrade pip
(venv) you@you chat-api % pip install -r requirements.txt
```

### install package using a setup.py and pip

To install package for development, from inside the top-level or main minichatgpt directory (the one where if you `ls` you see `setup.py`, `requirements.txt` and `README.md` in the same folder as you)
run the below at the command line or terminal:

```console
pip install -e .
```

leave out the `-e` for production `pip install .`, for other development packages like jupyter notebook and matplotlib, run:

```console
pip install -e ".[interactive]"
```

you should see something like

```
Obtaining file:///Users/.../minichatgpt
  Preparing metadata (setup.py) ... done
Installing collected packages: minichatgpt
  Running setup.py develop for minichatgpt
Successfully installed minichatgpt-0....
```

Now from directories other than the top-level or main minichatgpt directory you can

```python
import minichatgpt
from minichatgpt.example_script import example_class_function
```

and the changes you make to example_class_function will be available to you with your next `import minichatgpt`, no `pip install -e .` required

## Tutorial

**Forward batching**: Since the models can be fairly big and we want to rollout large PPO batches this can lead to out-of-memory errors when doing the forward passes for text generation and sentiment analysis. We introduce the parameter `forward_batch_size` to split the forward passes into smaller batches. Although this hurts performance a little this is neglectible compared to the computations of the backward passes when optimizing the model. The same parameter is used in the `PPOTrainer` when doing forward passes. The `batch_size` should multiple of `forward_batch_size`.

# References and Credits 

@misc{vonwerra2022trl,
  author = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert},
  title = {TRL: Transformer Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lvwerra/trl}}
}

