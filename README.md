# minichatgpt

minimal implementation of reinforcement learning from human feedback


## Building Development Environments

### python virtual environment for data science

```console
you@you chat-api % python3 -m venv venv
you@you chat-api % source venv/bin/activate
(venv) you@you chat-api % pip install --upgrade pip
(venv) you@you chat-api % pip install -r requirements.txt
```

### install package using a setup.py and pip

To install package, from inside the top-level or main minichatgpt directory (the one where if you `ls` you see `setup.py`, `requirements.txt` and `README.md` in the same folder as you)
run the below at the command line or terminal:

```console
pip install -e .
```

the above is meant for production, for development packages like jupyter notebook and matplotlib, run:

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

