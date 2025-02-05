#Discuss the implications and advantages of each 
# scenario and explain your rationale as to how 
#the model should be trained given the following: 
#1. If the entire network should be frozen. 

# Implications: Inability to update parameters. 
# Classification/NER will not learn anything new, as there
# aren't any updates to the weights
# So it can't really capture anything beyond what it knows 
#from pre=training

# Advantages: Very fast training.


#2. If only the transformer backbone should be frozen.
# Implications: The task-specific heads are still trainable, 
# but the transformer layer remains fronze. So no domain-specific
#fine-tuning
# Advantages: The transformer pretraining could still
# be somehow beneficial. The training is faster (less parameters
# updated at every step) 
#useful for tasks where adaptation of the backbone is not necessary


#3. If only one of the task-specific heads (either for Task A or Task B) should be frozen. 
# Implications: The forzen head acts as a fixed feature extractor. One main disadvantage
# is that if tasks are related, you lose the chance to benefit from the shared knowledge
# Advantages: Preserving pre-training capabilities for the frozen task and adaptiing for the 
#other task. So the main advantage is preserving the performance of the well-trained and fine-tuned
#head without messing with the other task's head.


# Consider a scenario where transfer learning can be beneficial. 
# Explain how you would approach 
# the transfer learning process, including: 

# 1. The choice of a pre-trained model. 
# 2. The layers you would freeze/unfreeze. 
# 3. The rationale behind these choices.

#1) I would select a domain like bert-base-uncased for transfer learning. 
# 2) I would fine-tune the layers that are more task-specific, 
# and freeze the layers that are more general. So we might freeze the first couple
#of layers in BERT to keep them as feature extractors. 
# 3) Lower transformer layers often learn unversal linguistic features, and higher
#layers learn more task-specific features.
# For example, freeze layers 0-4 and fine-tune layers 5-11 in a 12-layer BERT model.


