## Hello! This is a repo that uses supervised fine tuning/reinforcement learning with human feedback to make the base GPT-2 model more polite!

# Step 1: Supervised Fine-Tuning
In order to fine tune our model, we need DATA. I used Stanford's Politeness Corpus, which is a database of around 5000 utterances of polite/unpolite behavior. Our goal is for the model to behave more like these positive examples. Fine tuning it on strictly positive results yielded these results: 

{'loss': 2.0365, 'grad_norm': 11.7273, 'learning_rate': 4.4266e-05, 'epoch': 0.23} 
{'loss': 1.8060, 'grad_norm': 10.9722, 'learning_rate': 3.8532e-05, 'epoch': 0.46} 
{'loss': 1.9268, 'grad_norm': 10.7645, 'learning_rate': 3.2798e-05, 'epoch': 0.69} 
{'loss': 1.7529, 'grad_norm': 6.3128, 'learning_rate': 2.7064e-05, 'epoch': 0.92} 

{'eval_loss': 1.6979, 'eval_runtime': 2.9053, 'eval_samples_per_second': 75.034, 'eval_steps_per_second': 9.637, 'epoch': 1.0} 

{'loss': 1.6426, 'grad_norm': 8.9161, 'learning_rate': 2.1330e-05, 'epoch': 1.15} 
{'loss': 1.5255, 'grad_norm': 11.4214, 'learning_rate': 1.5596e-05, 'epoch': 1.38} 
{'loss': 1.6770, 'grad_norm': 7.1712, 'learning_rate': 9.8623e-06, 'epoch': 1.61} 
{'loss': 1.5274, 'grad_norm': 11.9370, 'learning_rate': 4.1284e-06, 'epoch': 1.83} 

{'eval_loss': 1.7035, 'eval_runtime': 2.5211, 'eval_samples_per_second': 86.469, 'eval_steps_per_second': 11.106, 'epoch': 2.0}  

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 872/872 [03:39<00:00, 4.22it/s]  

We can see the loss decreasing (which is good!), but our eval loss stagnates. This is most likely due to the limited data set, as there is only around 1000 positive examples. As we get into RLHF, I will show more concrete examples of before and after our training. 

# Step 2: Reinforcement Learning with Human Feedback!