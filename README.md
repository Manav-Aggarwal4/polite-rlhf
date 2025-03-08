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
  i. Train the BERT Model:
      First, we need to train a BERT model on how to properly score our model's responses. It uses regression to output a single float "score", where the higher it is, the more polite. I trained it on around 5000         
      utterances from a annotated MIT dataset. Here are the results:  
    {'loss': 0.04, 'grad_norm': 2.7226834297180176, 'learning_rate': 4.546279491833031e-05, 'epoch': 0.18}                                                           
    {'loss': 0.0256, 'grad_norm': 0.752590537071228, 'learning_rate': 4.092558983666062e-05, 'epoch': 0.36}                                                          
    {'loss': 0.0192, 'grad_norm': 1.339168906211853, 'learning_rate': 3.638838475499093e-05, 'epoch': 0.54}                                                          
    {'loss': 0.0132, 'grad_norm': 0.9183265566825867, 'learning_rate': 3.1851179673321235e-05, 'epoch': 0.73}                                                        
    {'loss': 0.0149, 'grad_norm': 1.658274531364441, 'learning_rate': 2.7313974591651543e-05, 'epoch': 0.91}                                                         
    {'eval_loss': 0.009353662841022015, 'eval_runtime': 9.9578, 'eval_samples_per_second': 49.107, 'eval_steps_per_second': 6.226, 'epoch': 1.0}                     
    {'loss': 0.0126, 'grad_norm': 2.82747745513916, 'learning_rate': 2.277676950998185e-05, 'epoch': 1.09}                                                           
    {'loss': 0.0078, 'grad_norm': 0.7475993037223816, 'learning_rate': 1.8239564428312163e-05, 'epoch': 1.27}                                                        
    {'loss': 0.0075, 'grad_norm': 1.3442684412002563, 'learning_rate': 1.3702359346642468e-05, 'epoch': 1.45}                                                        
    {'loss': 0.0062, 'grad_norm': 0.5093813538551331, 'learning_rate': 9.165154264972778e-06, 'epoch': 1.63}                                                         
    {'loss': 0.0069, 'grad_norm': 0.3205867409706116, 'learning_rate': 4.627949183303086e-06, 'epoch': 1.81}                                                         
    {'loss': 0.0066, 'grad_norm': 0.4753980040550232, 'learning_rate': 9.074410163339383e-08, 'epoch': 2.0}                                                          
    {'eval_loss': 0.0069235265254974365, 'eval_runtime': 8.3446, 'eval_samples_per_second': 58.601, 'eval_steps_per_second': 7.43, 'epoch': 2.0} 
      
  Again, the loss has consistenly decreased. Progress!
      
