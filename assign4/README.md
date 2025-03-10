# Assignment 4: Running GPT2 inference w/ Flash Attention
![images](https://github.com/user-attachments/assets/90ed33d5-968b-4785-8944-3737d5d95f93)
<img src="https://github.com/user-attachments/assets/5ea688cb-7203-4b1e-82d4-f5af42f54ec5" width="630" alt="flashattention">

Following your successful work on the softmax kernel, your lab has given you a new project to tackle: implementing flash attention. 
Fortunately, your lab has already implemented a flash attention kernel but it doesn't work for running inference due to the changing sequence length. 
Your goal is to improve the current kernel and develop a robust flash attention algorithm to run GPT2 inference that has been trained on Shakespearean text.

## To get started:
- Install the necessary packages in your environment

- Run `python3 gpt149.py part4` to pass the test case for fixed sequence length (N=32)
- Run `python3 gpt149.py part4 --inference -m shakes128` to run GPT2 inference with the flash attention kernel.

You can also run the inference on larger sequence lengths:
- `python3 gpt149.py part4  --inference -m shakes256` for the seq_len=256 model
- `python3 gpt149.py part4 --inference -m shakes1024` for the seq_len=1024 model
- `python3 gpt149.py part4  --inference -m shakes2048` for the seq_len=2048 model

`usage: gpt149.py [-h] [-m MODEL] [--inference] [-N N] testname` (testname is fixed to `part4`)

Modify the kernel in `attention.cu` to make the kernel work for various sequence lengths

## Some hints
- Understand what the current kernel is doing, and how it implements the flash attention algorithm
- Play around with various seq. length values in `gpt149.py (line 178)` and identify which values the kernel works for (you'll notice a trend for the values). Think about why this is the case...
- The fix is quite simple, and only a few lines of code.

## For the ambitious:
The current flash attention code is quite slow (if you look at the inference block times the naive pytorch attention implementation outperforms the kernel)\
With all the topics we've covered so far, can you think of some ways to improve the current implementation? \
Maybe making the matmuls run on tensor cores? Applying reduction for the running sum `lij` and for finding the maximum `mij`?  \
Try to improve the kernel even more! :)
