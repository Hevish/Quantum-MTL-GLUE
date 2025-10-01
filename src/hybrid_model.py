# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from tasks import GLUE_TASKS
from my_quantum_circuit import QuantumCircuit


class MTLModel(nn.Module):
    def __init__(self, vocab_size, tasks, hidden_size=768, dropout_rate=0.1,
                 device=None, latent_dim=30, QUANTUM=False, quantum_params=None):

        super(MTLModel, self).__init__()

        self.tasks = tasks
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        self.use_quantum = bool(QUANTUM)

        # device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # BERT encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.elmo_available = False
        print("Using BERT-only model (ELMo disabled)")

        # BiLSTM sentence encoder
        encoder_input_dim = hidden_size
        self.sentence_encoder = nn.LSTM(
            input_size=encoder_input_dim,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )

        # Classical projection layers reduce to latent_dim
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

        # Classical heads (used if not quantum)
        if not self.use_quantum:
            for task_name, task_config in tasks.items():
                num_classes = task_config['num_classes']
                pred_layer = nn.Sequential(
                    nn.Linear(self.latent_dim, num_classes)
                )
                setattr(self, f'{task_name}_pred_layer', pred_layer)

        # Quantum integration
        else:
            if quantum_params is None:
                raise ValueError("quantum_params must be provided when QUANTUM=True.")

            # Extract quantum parameters
            tasks_qubits = quantum_params.get('tasks_qubits')
            layers_per_task = quantum_params.get('layers_per_task')
            task_observables = quantum_params.get('task_observables')
            encoding_layers = quantum_params.get('encoding_layers')
            # task_order defines the ordering of outputs returned by the QC (defaults to tasks.keys())
            task_order = quantum_params.get('task_order', list(self.tasks.keys()))

            if tasks_qubits is None or layers_per_task is None or task_observables is None or encoding_layers is None:
                raise ValueError("quantum_params must include 'tasks_qubits', 'layers_per_task', 'task_observables', and 'encoding_layers'.")

            if len(tasks_qubits) != len(layers_per_task) or len(tasks_qubits) != len(task_observables):
                raise ValueError("Length mismatch: tasks_qubits, layers_per_task and task_observables must align.")

            # compute expected encoding dim for the QC
            total_qubits = sum(tasks_qubits)
            enc_dim = encoding_layers * total_qubits

            # if latent_dim != enc_dim create a linear mapping
            if enc_dim != self.latent_dim:
                self.fc_to_quantum = nn.Linear(self.latent_dim, enc_dim)
            else:
                self.fc_to_quantum = None

            # instantiate the QuantumCircuit (expects to accept (B, enc_dim) and return per-task outputs)
            self.qc = QuantumCircuit(
                tasks_qubits=tasks_qubits,
                layers_per_task=layers_per_task,
                task_observables=task_observables,
                encoding_layers=encoding_layers
            )

            # record ordering and enc dim for forward checks
            self._quantum_task_order = task_order
            self._quantum_enc_dim = enc_dim

        # Move model (and submodules/parameters) to device
        self.to(self.device)

    def encode_sentence(self, input_ids, attention_mask, raw_texts=None):
        """Encode sentence with BERT + BiLSTM + Max Pooling"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if self.training:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            combined_embs = bert_outputs.last_hidden_state
        else:
            with torch.no_grad():
                bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                combined_embs = bert_outputs.last_hidden_state

        lstm_mask = attention_mask.float()
        seq_lengths = lstm_mask.sum(dim=1).cpu().int()
        seq_lengths = torch.clamp(seq_lengths, min=1)

        packed_input = nn.utils.rnn.pack_padded_sequence(
            combined_embs, seq_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.sentence_encoder(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        batch_size, actual_seq_len, hidden_dim = lstm_output.shape

        if attention_mask.size(1) != actual_seq_len:
            if attention_mask.size(1) > actual_seq_len:
                pooling_mask = attention_mask[:, :actual_seq_len]
            else:
                padding_length = actual_seq_len - attention_mask.size(1)
                padding = torch.zeros(batch_size, padding_length, device=self.device)
                pooling_mask = torch.cat([attention_mask, padding], dim=1)
        else:
            pooling_mask = attention_mask

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
        num_classes = task_config['num_classes']

        # move label
        if label is not None:
            label = label.to(self.device)

        # set bert mode
        if self.training:
            self.bert.train()
        else:
            self.bert.eval()

        # compute latent (B, latent_dim)
        if pair_input:
            s1_enc = self.encode_sentence(input1, attention_mask1, raw_texts1)
            s2_enc = self.encode_sentence(input2, attention_mask2, raw_texts2)
            pair_features = torch.cat([
                s1_enc, s2_enc,
                torch.abs(s1_enc - s2_enc),
                s1_enc * s2_enc
            ], dim=1)  # (B, hidden_size*4)
            latent = self.pair_proj(pair_features)
        else:
            sent_enc = self.encode_sentence(input1, attention_mask1, raw_texts1)
            latent = self.single_proj(sent_enc)

        # QUANTUM branch: QC outputs are final logits
        if self.use_quantum:
            # map to enc_dim if needed
            if self.fc_to_quantum is not None:
                xq = self.fc_to_quantum(latent)  # (B, enc_dim)
            else:
                xq = latent  # assume matches enc_dim

            # ensure proper device/dtype for QC call
            xq = xq.to(self.device)

            # call QC; expect a tuple/list/dict of per-task outputs in the order self._quantum_task_order
            qouts = self.qc(xq)

            # support dict or list/tuple outputs
            if isinstance(qouts, dict):
                if task not in qouts:
                    raise RuntimeError("QuantumCircuit returned a dict without the requested task's output.")
                qout = qouts[task]
            else:
                if not isinstance(qouts, (list, tuple)):
                    raise RuntimeError("QuantumCircuit must return a list/tuple or dict of per-task tensors.")
                if len(qouts) != len(self._quantum_task_order):
                    raise RuntimeError("QuantumCircuit returned a different number of outputs than expected.")
                idx = self._quantum_task_order.index(task)
                qout = qouts[idx]

            qout = qout.to(self.device).float()
            # scale stsb outputs to [0,1] if regression
            if task == 'stsb':
                qout = 0.5 + 0.5 * torch.tanh(qout)
            # validate output shape matches expected task requirements
            # if task in ['stsb']:  # regression task
            #     # for regression, ensure output is (B,) or (B,1)
            #     if qout.dim() == 2 and qout.size(1) == 1:
            #         qout = qout.squeeze(-1)  # (B,1) -> (B,)
            #     elif qout.dim() == 1:
            #         pass  # already (B,)
            #     else:
            #         raise RuntimeError(
            #             f"Regression task '{task}' quantum output has incompatible shape {qout.shape}. Expected (B,) or (B,1)")
            # else:  # classification task
            #     expected_classes = num_classes
            #     if qout.dim() == 2:
            #         if qout.size(1) != expected_classes:
            #             raise RuntimeError(
            #                 f"Classification task '{task}' quantum output shape {qout.shape} doesn't match expected classes {expected_classes}. Quantum circuit must output exactly {expected_classes} observables.")
            #     elif qout.dim() == 1:
            #         if expected_classes != 1:
            #             raise RuntimeError(
            #                 f"Classification task '{task}' quantum output is 1D but expected {expected_classes} classes. Quantum circuit configuration error.")
            #     else:
            #         raise RuntimeError(f"Classification task '{task}' quantum output has unexpected shape {qout.shape}")

            out = {'logits': qout}

        # Classical branch
        else:
            pred_layer = getattr(self, f'{task}_pred_layer')
            out_logits = pred_layer(latent)
            out = {'logits': out_logits}

        # compute loss if label provided
        if label is not None:
            if task in ['stsb']:  # regression
                # out_logits may be (B,1) or (B,), handle both
                out_logits = out.get('logits').squeeze(-1)
                # scaled_logits = 0.5 + 0.5 * torch.tanh(out_logits)
                loss = F.mse_loss(out_logits, label.float())
            elif num_classes == 2:
                loss = F.binary_cross_entropy_with_logits(out.get('logits').squeeze(-1), label.float())
            else:
                # classification
                loss = F.cross_entropy(out.get('logits'), label.long())
            out['loss'] = loss

        return out
