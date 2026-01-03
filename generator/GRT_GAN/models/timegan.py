# -*- coding: UTF-8 -*-
import torch
import numpy as np
import torch.nn as nn

class EmbeddingNetwork(torch.nn.Module):
    """The embedding network (encoder) for TimeGAN
    """
    def __init__(self, args):
        super(EmbeddingNetwork, self).__init__()
        self.feature_dim = args.feature_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Embedder Architecture
        self.emb_rnn = torch.nn.LSTM(
            input_size=self.feature_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.emb_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.emb_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.emb_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.emb_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, X, T):
        """Forward pass for embedding features from original space into latent space
        Args:
            - X: input time-series features (B x S x F)
            - T: input temporal information (B)
        Returns:
            - H: latent space embeddings (B x S x H)
        """
        # Dynamic RNN input for ignoring paddings
        # print(f"X shape : {X.shape}")
        X_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=X, 
            lengths=T.cpu().to(torch.int64), 
            batch_first=True, 
            enforce_sorted=False
        )
        # print(f"X_packed: {X_packed}")
        # print(f"X_packed shape : {X_packed.data.shape}")
        # print(f"feature_dim: {self.feature_dim}")
        

        H_o, H_t = self.emb_rnn(X_packed)
        # print(f"H_o: {H_o}")
        
        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o, 
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )
        # print(f"H_o2: {H_o}")
        
        # 128 x 100 x 10
        logits = self.emb_linear(H_o)
        # print(f"logits: {logits}")
        # exit()
        # 128 x 100 x 10
        H = self.emb_sigmoid(logits)
        return H

class RecoveryNetwork(torch.nn.Module):
    """The recovery network (decoder) for TimeGAN
    """
    def __init__(self, args):
        super(RecoveryNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.feature_dim = args.feature_dim
        self.Z_dim = args.Z_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Recovery Architecture
        self.rec_rnn = torch.nn.LSTM(
            input_size=self.hidden_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        # only recover the original feature without condition
        self.rec_linear = torch.nn.Linear(self.hidden_dim, self.Z_dim)

        # # add a relu activation function to force the output to be positive
        # self.rec_relu = torch.nn.ReLU()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.rec_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.rec_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for the recovering features from latent space to original space
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information (B)
        Returns:
            - X_tilde: recovered data (B x S x F)
        """
        # Dynamic RNN input for ignoring paddings
        # H_packed = torch.nn.utils.rnn.pack_padded_sequence(
        #     input=H, 
        #     lengths=T.cpu().to(torch.int64), 
        #     batch_first=True, 
        #     enforce_sorted=False
        # )
        H_packed = H
        
        # 128 x 100 x 10
        H_o, H_t = self.rec_rnn(H_packed)
        
        # Pad RNN output back to sequence length
        # H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
        #     sequence=H_o, 
        #     batch_first=True,
        #     padding_value=self.padding_value,
        #     total_length=self.max_seq_len
        # )

        # 128 x 100 x 71
        X_tilde = self.rec_linear(H_o)

        # # apply the relu activation function to force the output to be positive
        # X_tilde = self.rec_relu(X_tilde)

        return X_tilde

class SupervisorNetwork(torch.nn.Module):
    """The Supervisor network (decoder) for TimeGAN
    """
    def __init__(self, args):
        super(SupervisorNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Supervisor Architecture
        self.sup_rnn = torch.nn.LSTM(
            input_size=self.hidden_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers-1,
            batch_first=True
        )
        self.sup_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.sup_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.sup_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.sup_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for the supervisor for predicting next step
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information (B)
        Returns:
            - H_hat: predicted next step data (B x S x E)
        """
        # Dynamic RNN input for ignoring paddings
        # H_packed = torch.nn.utils.rnn.pack_padded_sequence(
        #     input=H, 
        #     lengths=T.cpu().to(torch.int64), 
        #     batch_first=True, 
        #     enforce_sorted=False
        # )
        H_packed = H
        
        # 128 x 100 x 10
        H_o, H_t = self.sup_rnn(H_packed)
        
        # Pad RNN output back to sequence length
        # print(f"H_o shape: {H_o.shape}")
        # H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
        #     sequence=H_o, 
        #     batch_first=True,
        #     padding_value=self.padding_value,
        #     total_length=self.max_seq_len
        # )

        # 128 x 100 x 10
        logits = self.sup_linear(H_o)
        # 128 x 100 x 10
        H_hat = self.sup_sigmoid(logits)
        return H_hat


# class ConditionalGeneratorNetwork(torch.nn.Module):
#     """The conditional generator network (encoder) for TimeGAN
#     """
#     def __init__(self, args):
#         super(ConditionalGeneratorNetwork, self).__init__()
#         self.Z_dim = args.Z_dim
#         self.feature_dim = args.feature_dim
#         self.hidden_dim = args.hidden_dim
#         self.num_layers = args.num_layers
#         self.padding_value = args.padding_value
#         self.max_seq_len = args.max_seq_len

#         # Generator Architecture
#         # LSTM for condition processing
#         self.cond_rnn = torch.nn.LSTM(
#             input_size=self.feature_dim,
#             hidden_size=self.hidden_dim,
#             num_layers=self.num_layers,
#             batch_first=True
#         )
        
#         # LSTM for generation
#         self.gen_rnn = torch.nn.LSTM(
#             input_size=self.hidden_dim + self.feature_dim, 
#             hidden_size=self.hidden_dim, 
#             num_layers=self.num_layers, 
#             batch_first=True
#         )
        
#         self.gen_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.gen_sigmoid = torch.nn.Sigmoid()

#         # Init weights as before
#         with torch.no_grad():
#             for name, param in self.cond_rnn.named_parameters():
#                 if 'weight_ih' in name or 'weight_hh' in name:
#                     torch.nn.init.xavier_uniform_(param.data)
#                 elif 'bias' in name:
#                     param.data.fill_(1 if 'ih' in name else 0)
#             for name, param in self.gen_rnn.named_parameters():
#                 if 'weight_ih' in name or 'weight_hh' in name:
#                     torch.nn.init.xavier_uniform_(param.data)
#                 elif 'bias' in name:
#                     param.data.fill_(1 if 'ih' in name else 0)
#             for name, param in self.gen_linear.named_parameters():
#                 if 'weight' in name:
#                     torch.nn.init.xavier_uniform_(param)
#                 elif 'bias' in name:
#                     param.data.fill_(0)

#     def forward(self, Z, T):
#         """Takes in random noise (features) and generates synthetic features within the latent space
#         Args:
#             - Z: input random noise (B x S x Z)
#             - T: input temporal information
#         Returns:
#             - H: embeddings (B x S x E)
#         """
#         # Split Z into condition and generation inputs
#         condition = Z[:, :int(0.5 * self.max_seq_len), :]
#         gen_input = Z[:, int(0.5 * self.max_seq_len):, :]

#         # Process the condition through its own RNN
#         # cond_packed = torch.nn.utils.rnn.pack_padded_sequence(
#         #     input=condition, 
#         #     lengths=T.cpu().to(torch.int64), 
#         #     batch_first=True, 
#         #     enforce_sorted=False
#         # )
#         cond_out, cond_hidden = self.cond_rnn(condition)

#         # # Adjust total_length to match the condition sequence length
#         # cond_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
#         #     sequence=cond_out, 
#         #     batch_first=True,
#         #     padding_value=self.padding_value,
#         #     total_length=condition.size(1)  # Update this to reflect the correct length
#         # )

#         # Concatenate condition output with the rest of Z for generation
#         combined_input = torch.cat([cond_out, gen_input], dim=-1)

#         # Dynamic RNN input for ignoring paddings
#         # combined_packed = torch.nn.utils.rnn.pack_padded_sequence(
#         #     input=combined_input, 
#         #     lengths=T.cpu().to(torch.int64), 
#         #     batch_first=True, 
#         #     enforce_sorted=False
#         # )

#         H_o, H_t = self.gen_rnn(combined_input)
        
#         # Pad RNN output back to sequence length
#         # H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
#         #     sequence=H_o, 
#         #     batch_first=True,
#         #     padding_value=self.padding_value,
#         #     total_length=self.max_seq_len
#         # )

#         logits = self.gen_linear(H_o)
#         H = self.gen_sigmoid(logits)
#         return H


# class ConditionalGeneratorNetwork(torch.nn.Module):
#     """The conditional generator network (encoder) for TimeGAN"""
#     def __init__(self, args):
#         super(ConditionalGeneratorNetwork, self).__init__()
#         self.Z_dim = args.Z_dim
#         self.feature_dim = args.feature_dim
#         self.hidden_dim = args.hidden_dim
#         self.num_layers = args.num_layers
#         self.padding_value = args.padding_value
#         self.max_seq_len = args.max_seq_len

#         # Generator Architecture
#         self.cond_rnn = torch.nn.LSTM(
#             input_size=self.feature_dim,
#             hidden_size=self.hidden_dim,
#             num_layers=self.num_layers,
#             batch_first=True
#         )
        
#         self.macro_rnn = torch.nn.LSTM(
#             input_size=args.macro_data_dim,
#             hidden_size=self.hidden_dim,
#             num_layers=self.num_layers,
#             batch_first=True
#         )

#         self.noise_projection = torch.nn.Linear(self.Z_dim, self.hidden_dim)

#         self.gen_rnn_first_half = torch.nn.LSTM(
#             input_size=self.hidden_dim,  # Combining history_out and macro_out_first_half
#             hidden_size=self.hidden_dim, 
#             num_layers=self.num_layers, 
#             batch_first=True
#         )

#         self.gen_rnn_second_half = torch.nn.LSTM(
#             input_size=self.hidden_dim * 3,  # Combining generated first half, macro_out_second_half, and noise_input_projected
#             hidden_size=self.hidden_dim, 
#             num_layers=self.num_layers, 
#             batch_first=True
#         )
        
#         self.gen_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.gen_sigmoid = torch.nn.Sigmoid()

#         # Init weights as before
#         with torch.no_grad():
#             for name, param in self.cond_rnn.named_parameters():
#                 if 'weight_ih' in name or 'weight_hh' in name:
#                     torch.nn.init.xavier_uniform_(param.data)
#                 elif 'bias' in name:
#                     param.data.fill_(1 if 'ih' in name else 0)
#             for name, param in self.macro_rnn.named_parameters():
#                 if 'weight_ih' in name or 'weight_hh' in name:
#                     torch.nn.init.xavier_uniform_(param.data)
#                 elif 'bias' in name:
#                     param.data.fill_(1 if 'ih' in name else 0)
#             for name, param in self.gen_rnn_first_half.named_parameters():
#                 if 'weight_ih' in name or 'weight_hh' in name:
#                     torch.nn.init.xavier_uniform_(param.data)
#                 elif 'bias' in name:
#                     param.data.fill_(1 if 'ih' in name else 0)
#             for name, param in self.gen_rnn_second_half.named_parameters():
#                 if 'weight_ih' in name or 'weight_hh' in name:
#                     torch.nn.init.xavier_uniform_(param.data)
#                 elif 'bias' in name:
#                     param.data.fill_(1 if 'ih' in name else 0)
#             for name, param in self.gen_linear.named_parameters():
#                 if 'weight' in name:
#                     torch.nn.init.xavier_uniform_(param)
#                 elif 'bias' in name:
#                     param.data.fill_(0)

#     def forward(self, Z, T):
#         """Takes in random noise and both historical and macro conditions to generate future data.
#         Args:
#             - Z: input random noise (B x S x Z)
#             - T: input temporal information (B x S)
#         Returns:
#             - H: generated data (B x S x E)
#         """
#         # Split Z into historical condition, macro condition, and noise
#         history_condition = Z[:, :int(0.5 * self.max_seq_len), :]
#         macro_condition_first_half = Z[:, :int(0.5 * self.max_seq_len), self.Z_dim:]
#         macro_condition_second_half = Z[:, int(0.5 * self.max_seq_len):, self.Z_dim:]
#         noise_input = Z[:, int(0.5 * self.max_seq_len):, :self.Z_dim]

#         # Process the historical condition (x_{0,t/2}) through its own RNN
#         history_out, _ = self.cond_rnn(history_condition)

#         # Process both halves of the macro condition through the same RNN
#         macro_out_first_half, _ = self.macro_rnn(macro_condition_first_half)
#         macro_out_second_half, _ = self.macro_rnn(macro_condition_second_half)

#         # Combine the first half outputs for context
#         # combined_first_half_input = torch.cat([history_out, macro_out_first_half], dim=-1)
#         combined_first_half_input = history_out

#         # Generate the first half of the sequence
#         first_half_output, _ = self.gen_rnn_first_half(combined_first_half_input)

#         # Project noise_input to match the hidden dimension
#         noise_input_projected = self.noise_projection(noise_input)

#         # Combine generated first half, second half macro condition, and noise for second half generation
#         combined_second_half_input = torch.cat([first_half_output, macro_out_second_half, noise_input_projected], dim=-1)

#         # Generate the second half of the sequence
#         second_half_output, _ = self.gen_rnn_second_half(combined_second_half_input)

#         # Concatenate first half and second half outputs
#         full_sequence_output = torch.cat([first_half_output, second_half_output], dim=1)

#         logits = self.gen_linear(full_sequence_output)
#         H = self.gen_sigmoid(logits)
#         return H


class ConditionalGeneratorNetwork(torch.nn.Module):
    """The conditional generator network (encoder) for TimeGAN using Transformers"""
    def __init__(self, args):
        super(ConditionalGeneratorNetwork, self).__init__()
        self.Z_dim = args.Z_dim
        self.feature_dim = args.feature_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len
        self.num_heads = args.num_heads  # Number of attention heads in the Transformer

        # Generator Architecture
        self.history_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=self.num_heads),
            num_layers=self.num_layers
        )

        self.macro_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=self.num_heads),
            num_layers=self.num_layers
        )

        self.noise_projection = nn.Linear(self.Z_dim, self.hidden_dim)

        self.gen_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_dim * 3, nhead=self.num_heads),
            num_layers=self.num_layers
        )

        self.gen_linear = nn.Linear(self.hidden_dim * 3, self.feature_dim)
        self.gen_sigmoid = nn.Sigmoid()

    def forward(self, Z, T):
        """Takes in random noise and both historical and macro conditions to generate future data.
        Args:
            - Z: input random noise (B x S x Z)
            - T: input temporal information (B x S)
        Returns:
            - H: generated data (B x S x E)
        """
        # Split Z into historical condition, macro condition, and noise
        history_condition = Z[:, :int(0.5 * self.max_seq_len), :]
        macro_condition_first_half = Z[:, :int(0.5 * self.max_seq_len), self.Z_dim:]
        macro_condition_second_half = Z[:, int(0.5 * self.max_seq_len):, self.Z_dim:]
        noise_input = Z[:, int(0.5 * self.max_seq_len):, :self.Z_dim]

        # Process the historical condition through the Transformer
        history_out = self.history_transformer(history_condition.permute(1, 0, 2))  # Shape: (S/2, B, feature_dim)
        history_out = history_out.permute(1, 0, 2)  # Back to (B, S/2, feature_dim)

        # Process both halves of the macro condition through the same Transformer
        macro_out_first_half = self.macro_transformer(macro_condition_first_half.permute(1, 0, 2))
        macro_out_first_half = macro_out_first_half.permute(1, 0, 2)

        macro_out_second_half = self.macro_transformer(macro_condition_second_half.permute(1, 0, 2))
        macro_out_second_half = macro_out_second_half.permute(1, 0, 2)

        # Combine the first half outputs for context
        combined_first_half_input = torch.cat([history_out, macro_out_first_half], dim=-1)

        # Generate the first half of the sequence
        first_half_output = self.gen_transformer(combined_first_half_input.permute(1, 0, 2))
        first_half_output = first_half_output.permute(1, 0, 2)

        # Project noise_input to match the hidden dimension
        noise_input_projected = self.noise_projection(noise_input)

        # Combine generated first half, second half macro condition, and noise for second half generation
        combined_second_half_input = torch.cat([first_half_output, macro_out_second_half, noise_input_projected], dim=-1)

        # Generate the second half of the sequence
        second_half_output = self.gen_transformer(combined_second_half_input.permute(1, 0, 2))
        second_half_output = second_half_output.permute(1, 0, 2)

        # Concatenate first half and second half outputs
        full_sequence_output = torch.cat([first_half_output, second_half_output], dim=1)

        logits = self.gen_linear(full_sequence_output)
        H = self.gen_sigmoid(logits)
        return H



class GeneratorNetwork(torch.nn.Module):
    """The generator network (encoder) for TimeGAN
    """
    def __init__(self, args):
        super(GeneratorNetwork, self).__init__()
        self.Z_dim = args.Z_dim
        self.feature_dim= args.feature_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Generator Architecture
        self.gen_rnn = torch.nn.LSTM(
            input_size=self.feature_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.gen_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gen_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.gen_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.gen_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, Z, T):
        """Takes in random noise (features) and generates synthetic features within the latent space
        Args:
            - Z: input random noise (B x S x Z)
            - T: input temporal information
        Returns:
            - H: embeddings (B x S x E)
        """
        # Dynamic RNN input for ignoring paddings
        # print(f"Z shape : {Z.shape}")
        # print("T: ", T)
        # print("T shape: ", T.shape)
        Z_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=Z, 
            lengths=T.cpu().to(torch.int64), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 128 x 100 x 71
        H_o, H_t = self.gen_rnn(Z_packed)
        
        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o, 
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # 128 x 100 x 10
        logits = self.gen_linear(H_o)
        # B x S
        H = self.gen_sigmoid(logits)
        return H

class DiscriminatorNetwork(torch.nn.Module):
    """The Discriminator network (decoder) for TimeGAN
    """
    def __init__(self, args):
        super(DiscriminatorNetwork, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Discriminator Architecture
        self.dis_rnn = torch.nn.LSTM(
            input_size=self.hidden_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.dis_linear = torch.nn.Linear(self.hidden_dim, 1)

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.dis_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.dis_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, H, T):
        """Forward pass for predicting if the data is real or synthetic
        Args:
            - H: latent representation (B x S x E)
            - T: input temporal information
        Returns:
            - logits: predicted logits (B x S x 1)
        """
        # Dynamic RNN input for ignoring paddings
        # H_packed = torch.nn.utils.rnn.pack_padded_sequence(
        #     input=H, 
        #     lengths=T.cpu().to(torch.int64), 
        #     batch_first=True, 
        #     enforce_sorted=False
        # )
        H_packed = H
        
        # 128 x 100 x 10
        H_o, H_t = self.dis_rnn(H_packed)
        
        # # Pad RNN output back to sequence length
        # H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
        #     sequence=H_o, 
        #     batch_first=True,
        #     padding_value=self.padding_value,
        #     total_length=self.max_seq_len
        # )

        # 128 x 100
        logits = self.dis_linear(H_o).squeeze(-1)
        return logits

class TimeGAN(torch.nn.Module):
    """Implementation of TimeGAN (Yoon et al., 2019) using PyTorch
    Reference:
    - https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html
    - https://github.com/jsyoon0823/TimeGAN
    """
    def __init__(self, args):
        super(TimeGAN, self).__init__()
        self.device = args.device
        self.feature_dim = args.feature_dim
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size

        self.embedder = EmbeddingNetwork(args)
        self.recovery = RecoveryNetwork(args)
        self.generator = GeneratorNetwork(args)
        # self.generator = ConditionalGeneratorNetwork(args)
        self.supervisor = SupervisorNetwork(args)
        self.discriminator = DiscriminatorNetwork(args)
        self.alpha_kl = 1
        self.alpha_mmd = 1
        self.alpha_ms = 1
        self.alpha_tv = 1
        self.alpha_entropy = 1

    def compute_mmd(self, X, X_hat, kernel='rbf'):
        if kernel == 'rbf':
            sigma = torch.mean(torch.pdist(X)).item()
            K = lambda X1, X2: torch.exp(-torch.cdist(X1, X2)**2 / (2 * sigma ** 2))
        else:
            raise ValueError('Unsupported kernel type')

        X_kernel = K(X, X)
        X_hat_kernel = K(X_hat, X_hat)
        cross_kernel = K(X, X_hat)

        mmd_loss = X_kernel.mean() + X_hat_kernel.mean() - 2 * cross_kernel.mean()
        return mmd_loss
    
    def masked_mean(self, X, mask, dim, eps=1e-8, keepdim=False):
        """
        Compute the mean of X over `dim`, ignoring positions where mask == 0.
        X and mask must be the same shape.
        """
        X_masked = X * mask
        sum_ = X_masked.sum(dim=dim, keepdim=keepdim)
        count = mask.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
        return sum_ / count


    def masked_var(self, X, mask, dim, eps=1e-8, keepdim=False, unbiased=False):
        """
        Compute the variance of X over `dim`, ignoring positions where mask == 0.
        Uses var(X) = E[X^2] - (E[X])^2 approach.
        """
        mean_ = self.masked_mean(X, mask, dim=dim, eps=eps, keepdim=True)
        mean_sq = self.masked_mean(X**2, mask, dim=dim, eps=eps, keepdim=True)
        var_ = mean_sq - mean_**2

        if unbiased:
            # Apply the (N / (N-1)) correction if needed
            # and N > 1 unmasked entries. This is optional based on your needs.
            count = mask.sum(dim=dim, keepdim=True).clamp_min(2)
            var_ = var_ * (count / (count - 1))

        if not keepdim:
            var_ = var_.squeeze(dim=dim)  # remove dimension if keepdim=False

        return var_.clamp_min(0.0)  # avoid negative float rounding

    def mode_seeking_loss(self, Z1, Z2, X_hat1, X_hat2, mask):
        Lz = self.masked_mean(torch.abs(Z1 - Z2), mask, dim=[1, 2])
        Lx = self.masked_mean(torch.abs(X_hat1 - X_hat2), mask, dim=[1, 2])

        # print("Z1 shape",Z1.shape)
        # print("Z2 shape",Z2.shape)
        # print("X_hat1 shape",X_hat1.shape)
        # print("X_hat2 shape",X_hat2.shape)
        # print(f"Lz shape: {Lz.shape}")
        # print(f"Lx shape: {Lx.shape}")
        # print("LZ preview",Lz[:5])
        # print("LX preview",Lx[:5])

        # Compute the ratio of losses and take the masked mean
        return torch.mean(Lx / (Lz + 1e-6)).item()

    def tv_loss(self, X_hat):
        return torch.mean(torch.abs(X_hat[:, :, 1:] - X_hat[:, :, :-1]))

    def entropy_loss(self, X_hat):
        return -torch.mean(torch.sum(torch.softmax(X_hat, dim=1) * torch.log_softmax(X_hat, dim=1), dim=1))


    def _recovery_forward(self, X, T, M):
        """The embedding network forward pass and the embedder network loss
        Args:
            - X: the original input features
            - T: the temporal information
        Returns:
            - E_loss: the reconstruction loss
            - X_tilde: the reconstructed features
        """
        # Forward Pass
        # print(f"X preview on GPU {torch.cuda.current_device()} : {X[0,0,:5]}")
        H = self.embedder(X, T)
        # print(f"H preview on GPU {torch.cuda.current_device()} : {H[0,:5,1]}")
        X_tilde = self.recovery(H, T)
        # print(f"X_tilde preview on GPU {torch.cuda.current_device()} : {X_tilde[0,0,:5]}")

        # For Joint training
        H_hat_supervise = self.supervisor(H, T)
        # G_loss_S = torch.nn.functional.mse_loss(
        #     H_hat_supervise[:,:-1,:], 
        #     H[:,1:,:]
        # ) # Teacher forcing next output

        # # Reconstruction Loss

        # # preview X_tilde,X,H,H_hat_supervise elements
        # # print(f"X_tilde: {X_tilde[0,0,:]}")
        # # print(f"X: {X[0,0,:]}")
        # # print(f"H: {H[0,0,:]}")
        # # print(f"H_hat_supervise: {H_hat_supervise[0,0,:]}")

        # E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, X[:,:,:self.Z_dim])
        # E_loss0 = 10 * torch.sqrt(E_loss_T0)
        # E_loss = E_loss0 + 0.1 * G_loss_S
        # Compute the supervised loss (G_loss_S) with teacher forcing

        # given the Mask M
        # mask is get from 
        # mask = np.isnan(data)
        # self.mask = torch.BoolTensor(mask) if mask is not None else None
        # calculate the loss only on the non-nan values

        # cant not mask in latent space
        # H_hat_supervise_masked = H_hat_supervise
        # H_masked = H

        # G_loss_S_masked = torch.nn.functional.mse_loss(
        #     H_hat_supervise_masked[:,:-1,:],
        #     H_masked[:,1:,:]
        # )




        G_loss_S = torch.nn.functional.mse_loss(
            H_hat_supervise[:,:-1,:], 
            H[:,1:,:]
        )  # This operation will be distributed across GPUs by DataParallel

        # Invert the mask (so True => we keep that row)
        mask = ~M  

        # Use the mask along the first dimension (the batch dimension)
        # print(f"X_tilde shape: {X_tilde.shape}")
        # print(f"X shape: {X.shape}")
        # print(f"mask shape: {mask.shape}")
        # X_masked = X[mask]               # shape (num_unmasked, feature_num)
        # X_tilde_masked = X_tilde[mask]   # shape (num_unmasked, feature_num)
        # E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, X[:,:,:self.Z_dim])  # Ensure X_tilde and X are on the correct device
        # print("X_tile preview",X_tilde[:2,:5,0])
        # print("X preview",X[:,:,:self.Z_dim][:2,:5,0])
        E_loss_T0_masked = torch.nn.functional.mse_loss(X_tilde[mask], X[:,:,:self.Z_dim][mask])  # Ensure X_tilde and X are on the correct device
        # print("X_tile shape",X_tilde.shape)
        # print("X shape",X.shape)
        # # masked shape
        # print("mask shape",mask.shape)
        # print("X_tile_mased shape",X_tilde_masked.shape)
        # print("X_tile_masked_preview",X_tilde_masked[:2,:5,0])
        # print("X_masked_preview",X[:,:,:self.Z_dim][mask][:2,:5,0])
        # E_loss0 = 10 * torch.sqrt(E_loss_T0)  # E_loss_T0 should be a scalar tensor
        # E_loss0 = E_loss_T0  # E_loss_T0 should be a scalar tensor
        E_loss0_masked = E_loss_T0_masked  # E_loss_T0 should be a scalar tensor
        # E_loss = E_loss0 + 0.1 * G_loss_S
        E_loss_masked = E_loss0_masked + 0.1 * G_loss_S
        # return E_loss, E_loss0, E_loss_T0
        return E_loss_masked, E_loss0_masked, E_loss_T0_masked

    def _supervisor_forward(self, X, T, M):
        """The supervisor training forward pass
        Args:
            - X: the original feature input
        Returns:
            - S_loss: the supervisor's loss
        """
        # Supervision Forward Pass
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)

        # Supervised loss
        # can not mask the latent space
        # H_hat_supervise_masked = H_hat_supervise
        # H_masked = H
        S_loss = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])        # Teacher forcing next output\
        # S_loss_masked = torch.nn.functional.mse_loss(H_hat_supervise_masked[:,:-1,:], H_masked[:,1:,:])        # Teacher forcing next output\
        return S_loss
        # return S_loss_masked

    def _discriminator_forward(self, X, T, Z, M, gamma=1):
        """The discriminator forward pass and adversarial loss
        Args:
            - X: the input features
            - T: the temporal information
            - Z: the input noise
        Returns:
            - D_loss: the adversarial loss
        """
        # Real
        H = self.embedder(X, T).detach()
        
        # Generator
        E_hat = self.generator(Z, T).detach()
        H_hat = self.supervisor(E_hat, T).detach()

        # Forward Pass
        Y_real = self.discriminator(H, T)            # Encoded original data
        Y_fake = self.discriminator(H_hat, T)        # Output of generator + supervisor
        Y_fake_e = self.discriminator(E_hat, T)      # Output of generator


        # can not mask the discriminator
        # Y_real_masked = Y_real
        # Y_fake_masked = Y_fake
        # Y_fake_e_masked = Y_fake_e

        D_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(Y_real, torch.ones_like(Y_real))
        D_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.zeros_like(Y_fake))
        D_loss_fake_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.zeros_like(Y_fake_e))

        # D_loss_real_masked = torch.nn.functional.binary_cross_entropy_with_logits(Y_real_masked, torch.ones_like(Y_real_masked))
        # D_loss_fake_masked = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_masked, torch.zeros_like(Y_fake_masked))
        # D_loss_fake_e_masked = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e_masked, torch.zeros_like(Y_fake_e_masked))

        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
        # D_loss_masked= D_loss_real_masked + D_loss_fake_masked + gamma * D_loss_fake_e_masked

        return D_loss
        # return D_loss_masked

    def _generator_forward(self, X, T, Z, M, gamma=1):
        """The generator forward pass
        Args:
            - X: the original feature input
            - T: the temporal information
            - Z: the noise for generator input
        Returns:
            - G_loss: the generator's loss
        """
        assert not torch.isnan(X).any(), "Input contains NaN values"
        assert not torch.isinf(X).any(), "Input contains Inf values"

        # Supervisor Forward Pass
        H = self.embedder(X, T)
        H_hat_supervise = self.supervisor(H, T)


        # Generator Forward Pass
        E_hat = self.generator(Z, T)

        
        # print("E_hat shape",E_hat.shape)
        H_hat = self.supervisor(E_hat, T)
        
        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)



        log_res = False
        if log_res:
            print("X,preview",X.detach().cpu().numpy()[:2,:5,0])
            print("E_hat,preview",E_hat.detach().cpu().numpy()[:2,:5,0])
            print("H_hat,preview",H_hat.detach().cpu().numpy()[:2,:5,0])

            print("X_hat,preview",X_hat.detach().cpu().numpy()[:2,:5,0])

        # Generator Loss
        # 1. Adversarial loss
        Y_fake = self.discriminator(H_hat, T)        # Output of supervisor
        Y_fake_e = self.discriminator(E_hat, T)      # Output of generator


        G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
        G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e))



        # 2. Supervised loss
        G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])        # Teacher forcing next output

        # 3. Two Momments
        G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(X[:,:,:self.Z_dim].var(dim=0, unbiased=False) + 1e-6)))
        G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X[:,:,:self.Z_dim].mean(dim=0))))


        G_loss_V = G_loss_V1 + G_loss_V2

        X_real = X[:, :, :self.Z_dim]
        X_fake = X_hat[:, :, :self.Z_dim]
        mask_3d = ~M[:, :, :self.Z_dim]


        # 2) Compute masked means/vars
        mu_real = self.masked_mean(X_real, mask_3d, dim=(0,1))
        mu_fake = self.masked_mean(X_fake, mask_3d, dim=(0,1))

        var_real = self.masked_var(X_real, mask_3d, dim=(0,1), unbiased=False) + 1e-6
        var_fake = self.masked_var(X_fake, mask_3d, dim=(0,1), unbiased=False) + 1e-6

        # 3) Two Moments
        G_loss_V1 = torch.mean(torch.abs(torch.sqrt(var_fake) - torch.sqrt(var_real)))
        G_loss_V2 = torch.mean(torch.abs(mu_fake - mu_real))
        G_loss_V  = G_loss_V1 + G_loss_V2

        # 4) Standard deviation difference
        G_loss_std = torch.mean(torch.abs(var_fake - var_real) / var_real)

        # 5) KL Divergence (Gaussian)
        KL_term = 0.5 * (
            torch.log(var_fake / var_real)
            + (var_real + (mu_real - mu_fake)**2) / var_fake
            - 1.0
        )
        KL_loss = KL_term.mean()  # or sum(...) then .mean(), your choice





        # X_masked = X[~M]
        # only maske the [:,:self.Z_dim] part of X
        # X_masked = X
        # X_hat_masked = X_hat[~M]
        # X_masked[:, :, :self.Z_dim]= X[:,:,:self.Z_dim][~M]


        # 3.5 std of the generated data and the real data
        # generated_std= torch.sqrt(X_hat.std(dim=0, unbiased=False) + 1e-6)
        # real_std = torch.sqrt(X[:,:,:self.Z_dim].std(dim=0, unbiased=False) + 1e-6)
        # G_loss_std = torch.mean( torch.abs((generated_std - real_std))/ (real_std + 1e-6))

        # generated_variance = X_hat.var(dim=0, unbiased=False) + 1e-6
        # real_variance = X[:, :, :self.Z_dim].var(dim=0, unbiased=False) + 1e-6

        # # generated_variance_masked = X_hat_masked.var(dim=0, unbiased=False) + 1e-6
        # # real_variance_masked = X_masked[:, :, :self.Z_dim].var(dim=0, unbiased=False) + 1e-6

        # G_loss_std = torch.mean(torch.abs((generated_variance - real_variance)) / (real_variance + 1e-6))
        # # G_loss_std_masked = torch.mean(torch.abs((generated_variance_masked - real_variance_masked)) / (real_variance_masked + 1e-6))

        # # print('std:',torch.sqrt(X_hat.std(dim=0, unbiased=False)),torch.sqrt(X[:,:,:self.Z_dim].std(dim=0, unbiased=False)))
        # # 4. Summation
        # # G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 1*G_loss_V


        # # KL Divergence Loss
        # # Assuming X and X_hat represent the mean and variance of your real and fake distributions
        # epsilon = 1e-6
        # mu_real = X[:, :, :self.Z_dim].mean(dim=0)
        # sigma_real = X[:, :, :self.Z_dim].var(dim=0) + epsilon

        # mu_fake = X_hat.mean(dim=0)
        # sigma_fake = X_hat.var(dim=0) + epsilon

        # # Gaussian KL Divergence
        # KL_loss = 0.5 * torch.sum(
        #     torch.log(sigma_fake / sigma_real) +
        #     (sigma_real + (mu_real - mu_fake).pow(2)) / sigma_fake - 1,
        #     dim=0
        # ).mean()



        # MMD Loss
        # MMD_loss = self.compute_mmd(X[:, :, :self.Z_dim], X_hat)


                    # Z_mb = torch.rand((X_mb.size()[0], args.max_seq_len//2, args.Z_dim))*0.01
                    # # repace the second half of the real data with noise and keep the second half as it 
                    # Z_= np.concatenate((X_mb[:,:args.max_seq_len//2,:args.Z_dim],Z_mb),axis=1)
                    # # to tensor
                    # Z_= torch.tensor(Z_).to(X_mb.device)
                    # Z_mb_ = torch.cat((Z_,X_mb[:,:,args.Z_dim:]),dim=2)

        # Mode-Seeking Loss same 
        # Z1_mb = torch.rand((X.size()[0], self.max_seq_len//2, self.Z_dim))*0.01
        # Z2_mb = torch.rand((X.size()[0], self.max_seq_len//2, self.Z_dim))*0.01
        # Z1_mb = Z1_mb.to(self.device)
        # Z2_mb = Z2_mb.to(self.device)
        # Z1 = torch.cat((X[:,:self.max_seq_len//2,:self.Z_dim], Z1_mb), dim=1)
        # Z2 = torch.cat((X[:,:self.max_seq_len//2,:self.Z_dim], Z2_mb), dim=1)
        # Z1 = torch.cat((Z1, X[:,:,self.Z_dim:]), dim=2)
        # Z2 = torch.cat((Z2, X[:,:,self.Z_dim:]), dim=2)
        # X_hat1, X_hat2 = self.generator(Z1,T), self.generator(Z2,T)
        # MS_loss = self.mode_seeking_loss(Z1, Z2, X_hat1, X_hat2)



        Z1_mb = torch.rand((X.size()[0], self.max_seq_len//2, self.Z_dim))*0.01
        Z2_mb = torch.rand((X.size()[0], self.max_seq_len//2, self.Z_dim))*0.01
        Z1_mb = Z1_mb.to(self.device)
        Z2_mb = Z2_mb.to(self.device)
        Z1 = torch.cat((X[:,:self.max_seq_len//2,:self.Z_dim], Z1_mb), dim=1)
        Z2 = torch.cat((X[:,:self.max_seq_len//2,:self.Z_dim], Z2_mb), dim=1)
        Z1 = torch.cat((Z1, X[:,:,self.Z_dim:]), dim=2)
        Z2 = torch.cat((Z2, X[:,:,self.Z_dim:]), dim=2)

        E_hat_1 = self.generator(Z1, T)
        H_hat_1 = self.supervisor(E_hat_1, T)
        X_hat_1 = self.recovery(H_hat_1, T)

        # do the same to Z2
        E_hat_2 = self.generator(Z2, T)
        H_hat_2 = self.supervisor(E_hat_2, T)
        X_hat_2 = self.recovery(H_hat_2, T)


        MS_loss_masked = self.mode_seeking_loss(Z1[:,:,:self.Z_dim], Z2[:,:,:self.Z_dim], X_hat_1[:,:,:self.Z_dim], X_hat_2[:,:,:self.Z_dim], mask_3d)

        # Total Variation Loss
        # TV_loss = self.tv_loss(X_hat)

        # Entropy Loss
        # Entropy_loss = self.entropy_loss(X_hat)

        # Summation of all losses
        G_loss = (
            G_loss_U + 
            gamma * G_loss_U_e + 
            100 * torch.sqrt(G_loss_S) + 
            1 * G_loss_V+
            3 * G_loss_std  
            # MS_loss_masked
            # self.alpha_kl * KL_loss
            # self.alpha_mmd * MMD_loss +
            # self.alpha_ms * MS_loss 
            # self.alpha_tv * TV_loss +
            # self.alpha_entropy * Entropy_loss
        )

        return G_loss, G_loss_U, G_loss_U_e, G_loss_S, G_loss_V, KL_loss, G_loss_std, MS_loss_masked


    def _inference(self, Z, T):
        """Inference for generating synthetic data
        Args:
            - Z: the input noise
            - T: the temporal information
        Returns:
            - X_hat: the generated data
        """
        # Generator Forward Pass
        E_hat = self.generator(Z, T)
        H_hat = self.supervisor(E_hat, T)

        # Synthetic data generated
        X_hat = self.recovery(H_hat, T)
        return X_hat

    def forward(self, X, T, Z, M, obj, gamma=1):
        """
        Args:
            - X: the input features (B, H, F)
            - T: the temporal information (B)
            - Z: the sampled noise (B, H, Z)
            - obj: the network to be trained (`autoencoder`, `supervisor`, `generator`, `discriminator`)
            - gamma: loss hyperparameter
        Returns:
            - loss: The loss for the forward pass
            - X_hat: The generated data
        """
        if obj != "inference":
            if X is None:
                raise ValueError("`X` should be given")

            # X = torch.FloatTensor(X)
            # X = X.to(self.device)
            X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
            M = torch.as_tensor(M, dtype=torch.bool).to(self.device)

        if Z is not None:
            # Z = torch.FloatTensor(Z)
            # Z = Z.to(self.device)
            Z = torch.as_tensor(Z, dtype=torch.float32).to(self.device)

        if obj == "autoencoder":
            # Embedder & Recovery
            loss = self._recovery_forward(X, T, M)

        elif obj == "supervisor":
            # Supervisor
            loss = self._supervisor_forward(X, T, M)

        elif obj == "generator":
            if Z is None:
                raise ValueError("`Z` is not given")

            # Generator
            loss = self._generator_forward(X, T, Z,M)

        elif obj == "discriminator":
            if Z is None:
                raise ValueError("`Z` is not given")
            
            # Discriminator
            loss = self._discriminator_forward(X, T, Z,M)
            
            return loss

        elif obj == "inference":

            X_hat = self._inference(Z, T)
            X_hat = X_hat.cpu().detach()

            return X_hat

        else: raise ValueError("`obj` should be either `autoencoder`, `supervisor`, `generator`, or `discriminator`")

        return loss
