# Assignment 4: Running Shakespearean GPT2 inference w/ Flash Attention
![images](https://github.com/user-attachments/assets/90ed33d5-968b-4785-8944-3737d5d95f93)
<img src="https://github.com/user-attachments/assets/5ea688cb-7203-4b1e-82d4-f5af42f54ec5" width="630" alt="flashattention">

Following your successful work on the softmax kernel, your lab has given you a new project to tackle: implementing flash attention. 
Fortunately, your co-worker already implemented one but it doesn't work for running inference due to the changing sequence length. 
Your goal is to improve upon the work and develop a robust flash attention algorithm to run GPT2 inference that has been trained on Shakespearean text.

## To get started:
Run `python3 gpt149.py part4` to pass the test case for fixed sequence length (N=32)\
Run `python3 gpt149.py part4 --inference -m shakes128` to run GPT2 inference with flash attention on the 128M parameter model.

You can also run the inference on larger sequence lengths:
- `python3 gpt149.py part4  --inference -m shakes256` for the seq_len=256 model
- `python3 gpt149.py part4 --inference -m shakes1024` for the seq_len=1024 model
- `python3 gpt149.py part4  --inference -m shakes2048` for the seq_len=2048 model

`usage: gpt149.py [-h] [-m MODEL] [--inference] [-N N] testname` (testname is fixed to `part4`)

Modify the kernel in `attention.cu` to make the kernel work for various sequence lengths

## Some hints
Understand what the current kernel is doing, and how it implements the flash attention algorithm\
Play around with various seq. length values in `gpt149.py (line 178)` and identify which values the kernel works for (you'll notice a trend for the values)\
The fix is quite simple, and only a few lines of code.

## For the ambitious:
The current flash attention code is quite slow (It can only outperform pytorch w/ small parameter values), with all the topics we've covered so far think of a ways to improve the current implementation! \
Maybe making the matmul's run on tensor cores? Applying reduction for the running sum `lij` and for finding the maximum `mij`?  \
Think of other ways to improve the kernel! :)
