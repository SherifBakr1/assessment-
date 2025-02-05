#Task 4: Training Loop Implementation (BONUS) 
# If not already done, code the training loop for 
# the Multi-Task Learning Expansion in Task 2. 
# Explain any assumptions or decisions made paying special
#  attention to how training within a 
# MTL framework operates. Please note you need not actually 
# train the model. 

# Things to focus on: 
# ● Handling of hypothetical data 
# ● Forward pass 
# ● Metrics

# Assumptions: All samples (data or sentences in my case) contain labels
#for both tasks (classification and NER). I also used 
# align_ner_labels to map entity spans to tokenized input to align NER labels.
#I also used padding=True, truncation=True in the tokenizer to ensureconsistency
#I used -100 for ignored token, which I included in the loss function.
#I used a shared backbone where I process the input once for both tasks.
#Classification uses the [cls] token embedding and NER uses the token embeddings.
#Combine the losses by adding losses with different weights.
#For metrics, precision, recall, and f1-score could be calculated. Could be calculated
#after each epoch in a validation set, although I did not use a valiation set as the training
#data is very small.