import torch
from torch import nn
from transformers import BartForConditionalGeneration, BartConfig
from dnabart_config import get_dnabart_config

dnabartcfg = get_dnabart_config()
model = BartForConditionalGeneration(config=dnabartcfg)


# class DNABART(BartForConditionalGeneration):
#     def __init__(self, config: BartConfig, num_classes: int):
#         """
#         Custom BART model for DNA sequence generation and classification.

#         Args:
#             config (BartConfig): Configuration for the BART model.
#             num_classes (int): Number of classes for the classification task.
#         """
#         super().__init__(config)

#         # Add a classification head
#         self.classification_head = nn.Sequential(
#             nn.Linear(config.d_model, config.d_model // 2),
#             nn.ReLU(),
#             nn.Linear(config.d_model // 2, num_classes)
#         )

#     def forward(
#         self, 
#         input_ids, 
#         attention_mask=None, 
#         labels=None, 
#         classification_labels=None, 
#         task="generation"
#     ):
#         """
#         Forward pass for DNABART.

#         Args:
#             input_ids (torch.Tensor): Input token IDs.
#             attention_mask (torch.Tensor, optional): Attention mask for input IDs.
#             labels (torch.Tensor, optional): Target token IDs for sequence generation.
#             classification_labels (torch.Tensor, optional): Labels for classification task.
#             task (str): Task to perform ("generation" or "classification").

#         Returns:
#             dict: Outputs containing logits for generation or classification.
#         """
#         if task == "generation":
#             # Perform sequence generation
#             output = super().forward(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )
#             return output

#         elif task == "classification":
#             # Extract encoder outputs
#             encoder_outputs = self.model.encoder(
#                 input_ids=input_ids, attention_mask=attention_mask
#             )
#             # Pool the encoder outputs (e.g., mean pooling)
#             pooled_output = encoder_outputs.last_hidden_state.mean(dim=1)
#             # Pass through the classification head
#             logits = self.classification_head(pooled_output)

#             outputs = {"logits": logits}
#             if classification_labels is not None:
#                 loss_fn = nn.CrossEntropyLoss()
#                 loss = loss_fn(logits, classification_labels)
#                 outputs["loss"] = loss

#             return outputs

#         else:
#             raise ValueError("Task must be 'generation' or 'classification'")


import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os

class DNABARTForClassification(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        """
        Modify BART for classification task
        
        :param pretrained_model_path: Path to pretrained DNABART model
        :param num_classes: Number of classification classes
        """
        super().__init__()
        
        # Load pretrained BART model
        self.bart = BartForConditionalGeneration.from_pretrained(pretrained_model_path)
        
        # Freeze BART encoder layers (optional)
        for param in self.bart.model.encoder.parameters():
            param.requires_grad = False
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.bart.config.d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get BART encoder output
        bart_outputs = self.bart.model.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Use the pooled output (first token representation)
        pooled_output = bart_outputs.last_hidden_state[:, 0, :]
        
        # Classify
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return loss, logits

