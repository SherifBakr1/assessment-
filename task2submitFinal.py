import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class EnhancedMultiTaskTransformer(nn.Module):
    def __init__(self, model_name, num_classes, num_ner_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)        ##

        self.classifier = nn.Linear(hidden_size, num_classes)
        self.ner_classifier = nn.Linear(hidden_size, num_ner_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        hidden_state = self.dropout(outputs.last_hidden_state)

        cls_embedding = hidden_state[:, 0, :]
        sentence_logits = self.classifier(cls_embedding)
        ner_logits = self.ner_classifier(hidden_state)

        return sentence_logits, ner_logits

# Initialize model
num_classes = 3
num_ner_labels = 5
model = EnhancedMultiTaskTransformer(model_name, num_classes, num_ner_labels)

sentences = [
    ("Interstellar is the best movie ever", 0, []),
    ("Inception received critical acclaim", 0, []),
    ("The Dark Knight is a cinematic masterpiece", 0, [(4, 15, 1)]),
    ("Christopher Nolan directed Inception", 0, [(0, 17, 1)]),
    ("The football match ended in a draw", 1, []),
    ("Basketball requires great teamwork", 1, []),
    ("Soccer is the world's most popular sport", 1, []),
    ("World Cup matches attract millions", 1, [(4, 13, 3)]),
    ("Denver is in Colorado", 2, [(0, 6, 2), (10, 18, 2)]),
    ("The Amazon River flows through Brazil", 2, [(4, 16, 2), (29, 35, 2)]),
    ("Mount Everest is in the Himalayas", 2, [(0, 13, 2), (22, 31, 2)]),
    ("Tokyo will host the 2025 Summit", 2, [(0, 5, 2), (22, 28, 3)]),
    ("Microsoft announced new AI features", 0, [(0, 9, 3)]),
    ("Einstein revolutionized physics", 2, [(0, 9, 4)]),
    ("Google opened a Tokyo office", 2, [(0, 6, 3), (17, 22, 2)]),
    ("John moved from Paris to London", 2, [(0, 4, 1), (10, 15, 2), (19, 25, 2)]),
    ("The meeting is in Berlin", 2, [(16, 22, 2)]),
]

def align_ner_labels(text, entities, tokenizer):
    tokenized = tokenizer(text, return_offsets_mapping=True)
    labels = [-100] * len(tokenized["input_ids"])
    
    for (start, end, label) in entities:
        for i, (tok_start, tok_end) in enumerate(tokenized.offset_mapping):
            # Check for any overlap with entity span
            if (tok_start < end) and (tok_end > start) and (tok_start != 0):
                if labels[i] == -100:  # Only label if not already marked
                    labels[i] = label
    return labels

# Data processing
texts = [s[0] for s in sentences]
labels = torch.tensor([s[1] for s in sentences])
ner_annotations = [s[2] for s in sentences]

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", return_offsets_mapping=True)

ner_labels = torch.full((len(texts), inputs["input_ids"].shape[1]), -100)
for i, (text, entities) in enumerate(zip(texts, ner_annotations)):
    aligned = align_ner_labels(text, entities, tokenizer)
    ner_labels[i, :len(aligned)] = torch.tensor(aligned)

# Training setup
classification_criterion = nn.CrossEntropyLoss()
ner_criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.AdamW(model.parameters(), lr=3e-5)
epochs = 30

def calculate_loss_weights(cls_loss, ner_loss):
    ner_weight = min(0.8, 0.5 + (ner_loss.item()/3))  
    return (1 - ner_weight) * cls_loss + ner_weight * ner_loss

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    sentence_logits, ner_logits = model(inputs["input_ids"], inputs["attention_mask"])
    
    cls_loss = classification_criterion(sentence_logits, labels)
    ner_loss = ner_criterion(ner_logits.view(-1, num_ner_labels), ner_labels.view(-1))
    total_loss = calculate_loss_weights(cls_loss, ner_loss)
    
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss.item():.4f} [CLS: {cls_loss.item():.4f}, NER: {ner_loss.item():.4f}]")

def print_predictions(sentences, model, tokenizer):
    model.eval()
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        sentence_logits, ner_logits = model(inputs["input_ids"], inputs["attention_mask"])
        pred_labels = torch.argmax(sentence_logits, dim=1)
        ner_preds = torch.argmax(ner_logits, dim=-1)

    ner_label_map = {0: "O", 1: "PER", 2: "LOC", 3: "ORG", 4: "MISC"}
    cls_label_map = {0: "Movies", 1: "Sports", 2: "Geo"}
    
    for i, sentence in enumerate(sentences):
        print(f"\nSentence: {sentence}")
        print(f"Predicted Category: {cls_label_map[pred_labels[i].item()]}")
        
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][i])
        predictions = ner_preds[i].tolist()
        
        current_entity = []
        current_label = None
        for token, pred in zip(tokens, predictions):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            
            label = ner_label_map.get(pred, "O")
            
            if label != "O":
                if current_label is None:
                    current_label = label
                    current_entity = [token]
                elif label == current_label:
                    current_entity.append(token)
                else:
                    print(f"Entity: {' '.join(current_entity)} ({current_label})")
                    current_label = label
                    current_entity = [token]
            else:
                if current_entity:
                    print(f"Entity: {' '.join(current_entity)} ({current_label})")
                    current_entity = []
                    current_label = None
        
        if current_entity:
            print(f"Entity: {' '.join(current_entity)} ({current_label})")

# Test sentences
test_sentences = [
    "Christopher Nolan's Batman trilogy was produced by Warner Bros",
    "The World Cup final will be in Qatar next year",
    "Albert Einstein worked at the Princeton Institute",
    "Google announced new cloud features in Tokyo"
]

print("\n Predictions ")
print_predictions(test_sentences, model, tokenizer)