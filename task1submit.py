from transformers import AutoTokenizer, AutoModel
import torch 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModel.from_pretrained(model_name) 

sentences = [
    "Interstellar is the best movie ever",
    "I would rather play soccer than watch a movie",
    "Denver is in Colorado"
]

inputs = tokenizer(
    sentences, padding=True, truncation=True, return_tensors="pt"
)  

with torch.no_grad(): 
    outputs = model(**inputs) 

token_embeddings = outputs.last_hidden_state

attention_mask = inputs["attention_mask"]  


mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

masked_embeddings = token_embeddings * mask 

summed = torch.sum(masked_embeddings, dim=1)  

counts = torch.clamp(mask.sum(1), min=1e-9) 

mean_pooled = summed / counts  

normalized_embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

embeddings = normalized_embeddings.numpy()
print("Embedding shape:", embeddings.shape)

similarity_matrix = cosine_similarity(embeddings)
print("\nCosine Similarity Matrix:\n", similarity_matrix)