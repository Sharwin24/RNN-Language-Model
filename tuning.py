import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from rnn import RNN, HyperParams
import time
import os
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import math


class RNNLLM:
    def __init__(self, train_valid_test_files: tuple[str, str, str], hp: HyperParams):
        self.train_file, self.valid_file, self.test_file = train_valid_test_files
        self.HP = hp
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using Torch device: {self.device}')

    def train_models(self, hyperparams: list[HyperParams]):
        hp_to_loss: dict[HyperParams, tuple[float, float, float]] = {}
        # Each value in the dict is a tuple of (valid loss, test_loss, perplexity)
        print(f'Evaluating {len(hyperparams)} hyperparameter configurations')
        for i, hp in enumerate(hyperparams):
            print(f'Evaluating HP {i+1}/{len(hyperparams)}')
            self.HP = hp
            self.train(debug=False, exp_id=i)
            valid_loss, valid_perplexity = self.evaluate(self.valid_loader)
            test_loss, test_perplexity = self.evaluate(self.test_loader)
            hp_to_loss[hp] = (valid_loss, test_loss, valid_perplexity)
        return hp_to_loss

    def evaluate(self, data_loader):
        self.model.eval()
        hidden = self.init_hidden_layer(
            self.HP.num_layers, self.HP.batch_size, self.HP.hidden_dim
        )
        total_loss = 0.0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                actual_batch_size = x.size(0)
                if hidden.size(1) != actual_batch_size:
                    hidden = self.init_hidden_layer(
                        self.HP.num_layers, actual_batch_size, self.HP.hidden_dim
                    )
                hidden = hidden.detach()
                output, hidden = self.model(x, hidden)
                loss = self.loss_func(
                    output.view(-1, self.HP.vocab_size), y.view(-1))
                total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        perplexity = math.exp(avg_loss)
        return avg_loss, perplexity

    def load_model(self, exp_dir: str):
        model_weights_path = os.path.join(exp_dir, 'model_weights.pth')
        hp_pickle_path = os.path.join(exp_dir, f'HP_{self.HP.__hash__()}.pkl')
        if os.path.exists(hp_pickle_path):
            with open(hp_pickle_path, 'rb') as f:
                stored_hp = pickle.load(f)
                if stored_hp == self.HP:
                    print(
                        f'Found existing model with the same hyperparameters: {self.HP}')
                return True
        return False

    def train(self, debug=True, exp_id: int = -1):
        # Before training, if we've already trained a model
        # with the exact same hyperparameters, we can load it instead of training
        self.setup_training_data()
        self.setup_training_model()
        if exp_id == -1:
            exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Create experiment folder
        experiment_folder = f'Experiment {exp_id}'
        os.makedirs(experiment_folder, exist_ok=True)
        if (self.load_model(experiment_folder)):
            print(f'Skipping training for model: {self.HP}')
            return
        train_losses = []
        train_perplexities = []
        valid_losses = []
        valid_perplexities = []
        start_time = time.time()
        for e in range(self.HP.num_epochs):
            print(f"Epoch {e+1}/{self.HP.num_epochs}") if debug else None
            # Initialize hidden layers on every epoch
            hidden = self.init_hidden_layer(
                self.HP.num_layers, self.HP.batch_size, self.HP.hidden_dim
            )
            epoch_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.train_loader):
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)
                actual_batch_size = x.size(0)
                # Adjust the hidden state to match the actual batch size
                if hidden.size(1) != actual_batch_size:
                    hidden = self.init_hidden_layer(
                        self.HP.num_layers, actual_batch_size, self.HP.hidden_dim
                    )
                print(
                    f"Processing batch {batch_idx + 1}/{len(self.train_loader)}"
                ) if debug else None
                self.optimizer.zero_grad()
                output, hidden = self.model(x, hidden)
                loss = self.loss_func(
                    output.view(-1, self.HP.vocab_size), y.view(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5)  # Clipping
                self.optimizer.step()
                # Prevent backprop through time?
                hidden = hidden.detach()
                epoch_loss += loss.item()
                print(
                    f"Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item()}"
                ) if debug else None
                # End of batch loop
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            epoch_perplexity = math.exp(avg_epoch_loss)
            train_perplexities.append(epoch_perplexity)
            train_losses.append(avg_epoch_loss)

            # Validation loss
            valid_loss, valid_perplexity = self.evaluate(self.valid_loader)
            valid_losses.append(valid_loss)
            valid_perplexities.append(valid_perplexity)
            print(
                f"Epoch {e+1}/{self.HP.num_epochs} Train Loss: {avg_epoch_loss}, Valid Loss: {valid_loss} Train Perplexity: {epoch_perplexity} Valid Perplexity: {valid_perplexity}"
            )
            # End of epoch loop
        end_time = time.time()
        train_time_seconds = end_time - start_time
        train_time_minutes = train_time_seconds / 60
        train_time_hours = int(train_time_seconds // 3600)
        train_time_minutes = int((train_time_seconds % 3600) // 60)
        train_time_seconds = int(train_time_seconds % 60)
        train_time_str = f'{train_time_hours:02d}:{train_time_minutes:02d}:{train_time_seconds:02d}'
        print(
            f'Training took {train_time_str} (HH:MM:SS)'
        )

        # Save the model weights and hyperparameters
        torch.save(self.model.state_dict(), os.path.join(
            experiment_folder, 'model_weights.pth'))

        # Write hyperparameters to a pickle file
        with open(os.path.join(experiment_folder, f'HP_{self.HP.__hash__()}.pkl'), 'wb') as f:
            pickle.dump(self.HP, f)

        # Plot and save loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.HP.num_epochs + 1), train_losses,
                 label='Training Loss', marker='o', color='b')
        plt.plot(range(1, self.HP.num_epochs + 1), valid_losses,
                 label='Validation Loss', marker='o', color='r')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.figtext(
            0.15, 0.85, f'Training Time: {train_time_str} (HH::MM::SS)', fontsize=10, ha='left'
        )
        plt.savefig(os.path.join(experiment_folder, f'loss_curve.png'))

        # Plot and save perplexity
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.HP.num_epochs + 1), train_perplexities,
                 label='Training Perplexity', marker='o', color='b')
        plt.plot(range(1, self.HP.num_epochs + 1), valid_perplexities,
                 label='Validation Perplexity', marker='o', color='r')
        plt.xlabel('Epochs')
        plt.ylabel('Perplexity')
        plt.title('Perplexity Curve')
        plt.legend()
        plt.figtext(
            0.15, 0.85, f'Training Time: {train_time_str} (HH::MM::SS)', fontsize=10, ha='left'
        )
        plt.savefig(os.path.join(experiment_folder,
                    f'perplexity_curve.png'))

    def setup_training_model(self):
        print(f'Setting up training model with {self.HP}')
        self.model = RNN(self.HP)
        self.model = self.model.to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.HP.learning_rate)

    def setup_training_data(self):
        # Load the data and create a reduced vocabulary
        print(f'Loading data and creating reduced vocabulary')
        self.train_indices, self.train_vocab = self.load_data(self.train_file)
        self.valid_indices, self.valid_vocab = self.load_data(self.valid_file)
        self.test_indices, self.test_vocab = self.load_data(self.test_file)

        print(f'Train vocab size: {self.train_vocab}')
        print(f'Valid vocab size: {self.valid_vocab}')
        print(f'Test vocab size: {self.test_vocab}')

        print(f'Creating input-output pairs for the dataset')
        # Create input-output pairs for the dataset
        self.train_inputs, self.train_targets = self.create_sequences(
            self.train_indices, self.HP.seq_length)
        self.valid_inputs, self.valid_targets = self.create_sequences(
            self.valid_indices, self.HP.seq_length)
        self.test_inputs, self.test_targets = self.create_sequences(
            self.test_indices, self.HP.seq_length)

        print(f"Number of training sequences: {len(self.train_inputs)}")  # LOG: Added
        print(f"Sequence length: {self.HP.seq_length}")  # LOG: Added

        print(f'Creating TensorDataset and DataLoader objects')
        # Create the TensorDataset objects
        self.train_dataset = TensorDataset(
            self.train_inputs, self.train_targets)
        self.valid_dataset = TensorDataset(
            self.valid_inputs, self.valid_targets)
        self.test_dataset = TensorDataset(self.test_inputs, self.test_targets)

        # Create the DataLoader objects
        self.train_loader = DataLoader(
            self.train_dataset, self.HP.batch_size, shuffle=True)
        self.valid_loader = DataLoader(
            self.valid_dataset, self.HP.batch_size, shuffle=False)
        self.test_loader = DataLoader(
            self.test_dataset, self.HP.batch_size, shuffle=False)

        print(f"Number of batches in train_loader: {len(self.train_loader)}")  # LOG: Added
        print(f"Batch size in train_loader: {self.HP.batch_size}")  # LOG: Added

    def init_hidden_layer(self, num_layers, batch_size, hidden_dim):
        return torch.zeros(num_layers, batch_size, hidden_dim).to(self.device)

    # def create_sequences(self, data, seq_length):
    #     inputs = []
    #     targets = []
    #     for i in range(len(data) - seq_length):
    #         inputs.append(data[i:i + seq_length])  # input sequence
    #         # target sequence, shifted by one
    #         targets.append(data[i + 1:i + seq_length + 1])
    #     return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def create_sequences(self, data, seq_length):
        inputs = []
        targets = []
        for i in range(0, len(data) - seq_length, seq_length):  # Slide by seq_length
            inputs.append(data[i:i + seq_length])  # input sequence
            targets.append(data[i + 1:i + seq_length + 1])  # target sequence, shifted by one
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    def load_wikitext(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text

    def tokenize(self, text):
        return text.split(' ')

    def create_vocab_mapping(self, tokens):
        counter = Counter(tokens)
        vocab = counter.most_common(self.HP.vocab_size - 1)
        word_to_idx = {}
        for idx, (word, _) in enumerate(vocab):
            word_to_idx[word] = idx
        word_to_idx['<unk>'] = 0
        return word_to_idx

    def tokens_to_indices(self, tokens, word_to_idx):
        # return a list of indices (based on the dict mapping words to tokens)
        return [word_to_idx.get(token, word_to_idx['<unk>']) for token in tokens]

    def load_data(self, text):
        data_text = self.load_wikitext(text)
        data_tokens = self.tokenize(data_text)
        data_word_to_idx = self.create_vocab_mapping(data_tokens)
        vocab_size = len(data_word_to_idx)
        data_indices = self.tokens_to_indices(data_tokens, data_word_to_idx)
        return data_indices, vocab_size


if __name__ == '__main__':
    hp = HyperParams(
        vocab_size=10000,
        batch_size=32,
        seq_length=25,
        learning_rate=0.001,
        num_epochs=2,
        hidden_dim=256,
        num_layers=1,
        embedding_dim=100,
        dropout=0.5
    )
    multi_layer_rnn = HyperParams(
        vocab_size=10000,
        batch_size=32,
        seq_length=25,
        learning_rate=0.001,
        num_epochs=2,
        hidden_dim=256,
        num_layers=2,
        embedding_dim=100,
        dropout=0.5
    )
    rnn_llm = RNNLLM(
        train_valid_test_files=(
            'wiki2.train.txt', 'wiki2.valid.txt', 'wiki2.test.txt'
        ),
        hp=hp
    )
    # rnn_llm.train(debug=False)
    hps = [hp, multi_layer_rnn]
    hp_to_loss_map = rnn_llm.train_models(hps)
    for hp, (valid_loss, test_loss, valid_perplexity) in hp_to_loss_map.items():
        print('--------------------')
        print(
            f'{hp}\nValidation Loss: {valid_loss}, Test Loss: {test_loss}, Validation Perplexity: {valid_perplexity}'
        )
        print('--------------------')
