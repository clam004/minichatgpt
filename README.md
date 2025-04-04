# minichatgpt

Focused dissection of the implementation details of a small simplified self contained toy project demonstrating reinforcement learning from human feedback (RLHF) with special emphasis on connecting the equations describing proximal policy optimization to the lines of pytorch code that apply PPO to work with sequences, such as completing sentences so they end with a positive sentiment. We do this not by self-supervised or supervised learning, but rather, by generating text and learning from scores assigned to that text after it is generated, this is analogous to the way ChatGPT was trained using human scores of model generated answers to instructions. 
## Building Development Environments

### python virtual environment for data science

```console
you@you chat-api % python3 -m venv venv
you@you chat-api % source venv/bin/activate
(venv) you@you chat-api % pip install --upgrade pip
```

### install package using a setup.py and pip

To install package for development, from inside the top-level or 
main minichatgpt directory (the one where if you `ls` you see `setup.py`, `requirements.txt` and `README.md` in the same folder as you)
run `pip install -e .` at the command line or terminal. Leave out the `-e` for production `pip install .`, for other development packages like jupyter notebook and matplotlib, run:

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

- [Tutorial Slide Deck](https://docs.google.com/presentation/d/12yG8yuNl3JM8lAr3YCB_y2_3EHndZJaMvii7inT0WH0/edit?usp=sharing)
- [YouTube Video from Silicon Valley Code Camp](https://www.youtube.com/live/WnGFR-bSNWM)

# References and Credits 

@misc{vonwerra2022trl,
  author = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert},
  title = {TRL: Transformer Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lvwerra/trl}}
}

