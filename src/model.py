# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from tasks import GLUE_TASKS


class MTLModel(nn.Module):
    def __init__(self, vocab_size, tasks, hidden_size=768, dropout_rate=0.1, device=None, latent_dim=30):
        super(MTLModel, self).__init__()

        self.tasks = tasks
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim # 30 assumed 3 layers in Quantum circuit with 10 qubits each

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # BERT encoder only
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # No ELMo - BERT only for stability
        self.elmo_available = False
        print("Using BERT-only model (ELMo disabled)")

        # Sentence encoder (BERT only input)
        encoder_input_dim = hidden_size  # Only BERT embeddings
        self.sentence_encoder = nn.LSTM(
            input_size=encoder_input_dim,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )

        self.single_proj = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, 512),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.latent_dim)
        )
        self.pair_proj = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size * 4, 512),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.latent_dim)
        )
        # Task-specific prediction layers (now expect input_dim = latent_dim)
        for task_name, task_config in tasks.items():
            num_classes = task_config['num_classes']
            pair_input = task_config.get('pair_input', False)

            input_dim = self.latent_dim
            # if pair_input:
            #     input_dim = hidden_size * 4
            # else:
            #     input_dim = hidden_size
            if task_name != 'mnli':
                out_dim = 1
                pred_layer = nn.Sequential(
                    nn.Linear(input_dim, out_dim)
                )
            else:
                out_dim = 3
                pred_layer = nn.Sequential(
                    nn.Linear(input_dim, out_dim)
                )
            # pred_layer = nn.Sequential(
            #     nn.Dropout(dropout_rate),
            #     nn.Linear(input_dim, 512),
            #     nn.Tanh(),
            #     nn.Dropout(dropout_rate),
            #     nn.Linear(512, num_classes)
            # )

            setattr(self, f'{task_name}_pred_layer', pred_layer)

        # Move model to device
        self.to(self.device)

    def encode_sentence(self, input_ids, attention_mask, raw_texts=None):
        """Encode sentence with BERT + BiLSTM + Max Pooling"""
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # BERT embeddings - use gradient computation during training
        if self.training:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            combined_embs = bert_outputs.last_hidden_state
        else:
            with torch.no_grad():
                bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                combined_embs = bert_outputs.last_hidden_state

        # BiLSTM encoding
        lstm_mask = attention_mask.float()
        seq_lengths = lstm_mask.sum(dim=1).cpu().int()
        seq_lengths = torch.clamp(seq_lengths, min=1)

        # Pack sequences for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            combined_embs, seq_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.sentence_encoder(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Handle sequence length alignment for max pooling
        batch_size, actual_seq_len, hidden_dim = lstm_output.shape

        # Create attention mask that matches LSTM output length
        if attention_mask.size(1) != actual_seq_len:
            if attention_mask.size(1) > actual_seq_len:
                # Truncate mask
                pooling_mask = attention_mask[:, :actual_seq_len]
            else:
                # Pad mask with zeros
                padding_length = actual_seq_len - attention_mask.size(1)
                padding = torch.zeros(batch_size, padding_length, device=self.device)
                pooling_mask = torch.cat([attention_mask, padding], dim=1)
        else:
            pooling_mask = attention_mask

        # Max pooling with proper masking
        expanded_mask = pooling_mask.unsqueeze(-1).expand(batch_size, actual_seq_len, hidden_dim)
        lstm_output = lstm_output.masked_fill(~expanded_mask.bool(), -float('inf'))
        sentence_repr, _ = lstm_output.max(dim=1)

        return sentence_repr

    def forward(self, task=None, input1=None, input2=None, attention_mask1=None,
                attention_mask2=None, raw_texts1=None, raw_texts2=None, label=None):
        """Forward pass"""
        if task not in self.tasks:
            raise ValueError(f"Task '{task}' not found. Available: {list(self.tasks.keys())}")

        task_config = self.tasks[task]
        pair_input = task_config.get('pair_input', False)
        pred_layer = getattr(self, f'{task}_pred_layer')

        # Move label to device if provided
        if label is not None:
            label = label.to(self.device)

        # Set BERT to training mode if model is training
        if self.training:
            self.bert.train()
        else:
            self.bert.eval()

        if pair_input:
            # Encode both sentences for pair tasks
            s1_enc = self.encode_sentence(input1, attention_mask1, raw_texts1)
            s2_enc = self.encode_sentence(input2, attention_mask2, raw_texts2)

            # Combine pair features: [s1, s2, |s1-s2|, s1*s2]
            pair_features = torch.cat([
                s1_enc, s2_enc,
                torch.abs(s1_enc - s2_enc),
                s1_enc * s2_enc
            ], dim=1)

            latent = self.pair_proj(pair_features)
            # logits = pred_layer(pair_features)
        else:
            # Single sentence tasks
            sent_enc = self.encode_sentence(input1, attention_mask1, raw_texts1)
            # logits = pred_layer(sent_enc)
            latent = self.single_proj(sent_enc)
        logits = pred_layer(latent)
        out = {'logits': logits}

        # Compute loss if labels provided
        if label is not None:
            if task in ['stsb']:  # Regression task
                loss = F.mse_loss(logits.squeeze(-1), label.float())
            elif task in ['cola', 'sst2', 'mrpc', 'qnli', 'qqp', 'rte', 'wnli']:
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), label.float())
            else:  # Classification tasks
                loss = F.cross_entropy(logits, label.long())
            out['loss'] = loss

        return out