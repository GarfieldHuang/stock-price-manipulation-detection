"""
Model definitions for both training and analysis.
"""
import torch
import torch.nn.functional as F
import math
torch.autograd.set_detect_anomaly(True)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[0, :x.size(1), :]
        return self.dropout(x)


class TransformerCOCA(torch.nn.Module):
    # Deprecated
    def __init__(self, nb_features, seq_length, d_model, dim_ff, num_layers, latent_dim):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.nb_features = nb_features
        self.latent_dim = latent_dim

        # Transformer layers
        self.pe = PositionalEncoding(d_model=d_model, max_len=seq_length, dropout=0.1)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=8,
                                                              batch_first=True, dim_feedforward=dim_ff, dropout=0.1)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=8,
                                                              batch_first=True, dim_feedforward=dim_ff, dropout=0.1)

        self.encoder = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=num_layers)

        self.linear_embedding = torch.nn.Linear(in_features=nb_features, out_features=d_model)
        self.linear_output = torch.nn.Linear(in_features=d_model, out_features=nb_features)
        self.decoder_projection = torch.nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.linear_encoding = torch.nn.Linear(in_features=d_model * seq_length, out_features=d_model)

        self.projection_head = torch.nn.Linear(in_features=d_model * seq_length, out_features=latent_dim, bias=False)
        self.projection_dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x, mode='train', pretraining=False):
        q1 = self.embedding(x)
        context = self.encoding(q1)
        q2 = self.decoding(x, context, mode=mode, pretraining=pretraining)

        if pretraining:
            return torch.empty(self.latent_dim), torch.empty(self.latent_dim), q1, q2, context
        else:
            projection = self.projection(q1)
            rec_projection = self.projection(q2)
            return F.normalize(projection, dim=1), F.normalize(rec_projection, dim=1), q1, q2, context

    def embedding(self, x):
        x = self.linear_embedding(x)
        x = self.pe(x)
        return x

    def encoding(self, x):
        x = self.encoder(x)
        x = self.linear_encoding(x.view((-1, self.seq_length * self.d_model)))
        return x

    def decoding(self, x, context, mode='train', pretraining=False):
        batch = x.shape[0]
        seq_length = x.shape[1]

        if pretraining:
            if next(self.encoder.parameters()).is_cuda:
                decoder_output = torch.zeros((batch, seq_length + 1, self.nb_features), device='cuda:0').cuda()
                output = torch.zeros_like(x).cuda()
                mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_length).cuda()
                device = 'cuda:0'
            else:
                decoder_output = torch.zeros((batch, seq_length + 1, self.nb_features))
                output = torch.zeros_like(x)
                mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_length)
                device = 'cpu'

            if mode == "train":
                target = torch.cat((torch.zeros_like(x[:, 0:1, :]), x[:, :-1, :]), dim=1).cuda()
                target = self.embedding(target)
                output = self.decoder(tgt=target, memory=context.view((batch, 1, self.d_model)), tgt_mask=mask)
                output = self.linear_output(output)

            elif mode == "generate":
                for t in range(1, seq_length + 1):
                    target = self.embedding(decoder_output[:, :t, :])
                    mask = torch.nn.Transformer.generate_square_subsequent_mask(target.shape[1]).to(device=device)
                    decoder_output_t = self.decoder(tgt=target, memory=context.view((batch, 1, self.d_model)),
                                                    tgt_mask=mask)
                    decoder_output[:, t, :] = self.linear_output(decoder_output_t[:, -1, :])
                output = decoder_output[:, 1:, :]

        else:
            if next(self.encoder.parameters()).is_cuda:
                decoder_output = torch.zeros((batch, seq_length + 1, self.nb_features)).cuda()
                decoder_projection = torch.zeros((batch, seq_length + 1, self.d_model)).cuda()
                output = torch.zeros_like(x).cuda()
                mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_length).cuda()
            else:
                decoder_output = torch.zeros((batch, seq_length + 1, self.nb_features))
                decoder_projection = torch.zeros((batch, seq_length + 1, self.d_model))
                output = torch.zeros_like(x)
                mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_length)

            if mode == "train":
                target = torch.cat((torch.zeros_like(x[:, 0:1, :]), x[:, :-1, :]), dim=1).cuda()
                target = self.embedding(target)
                output = self.decoder(tgt=target, memory=context.view((batch, 1, self.d_model)), tgt_mask=mask)

            elif mode == "generate":
                for t in range(1, seq_length + 1):
                    target = self.embedding(decoder_output[:, :t, :])
                    mask = torch.nn.Transformer.generate_square_subsequent_mask(target.shape[1]).cuda()
                    decoder_output_t = self.decoder(tgt=target, memory=context.view((batch, 1, self.d_model)),
                                                    tgt_mask=mask)
                    decoder_projection[:, t, :] = decoder_output_t[:, -1, :]
                    decoder_output[:, t, :] = self.linear_output(decoder_output_t[:, -1, :])
                output = decoder_projection[:, 1:, :]

        return output

    def projection(self, x):
        x = self.projection_dropout(x)
        return self.projection_head(x.view((-1, self.seq_length * self.d_model)))


class TransformerAutoencoder(TransformerCOCA):
    def __init__(self, nb_features, seq_length, d_model, dim_ff, num_layers):
        super().__init__(nb_features, seq_length, d_model, dim_ff, num_layers, 32)

    def forward(self, x, mode):
        emb = self.embedding(x)
        context = self.encoding(emb)
        seq = self.decoding(x, context, mode=mode, pretraining=True)
        return seq


class StackedLSTMAutoencoder(torch.nn.Module):
    def __init__(self, nb_features, hidden_units, layers):
        super().__init__()
        self.nb_features = nb_features
        self.hidden_units = hidden_units
        self.d_model = hidden_units
        self.layers = layers

        self.encoder = torch.nn.LSTM(input_size=nb_features, hidden_size=hidden_units, num_layers=layers,
                                     batch_first=True, dropout=0.1)

        self.decoder = torch.nn.LSTM(input_size=hidden_units, hidden_size=hidden_units, num_layers=layers,
                                     batch_first=True, dropout=0.1)

        self.linear_output = torch.nn.Linear(in_features=hidden_units, out_features=nb_features)

    def forward(self, x):
        h, c = self.encoding(x)
        output = self.decoding(x, h, c)
        return output

    def encoding(self, x):
        _, (h, c) = self.encoder(x)
        return h[-1].reshape((-1, self.hidden_units)), c[-1].reshape((-1, self.hidden_units))

    def decoding(self, x, h, c):
        seq_length = x.shape[1]
        input = h.reshape((-1, 1, self.hidden_units)).repeat((1, seq_length, 1))
        h = h.repeat((self.layers, 1, 1))
        c = c.repeat((self.layers, 1, 1))

        pre_output, _ = self.decoder(input, (h, c))
        output = self.linear_output(pre_output)

        return output


class MLPAutoencoder(torch.nn.Module):
    def __init__(self, nb_features, seq_length, latent_dim, nb_layers, hidden_units, activation):
        super().__init__()
        self.nb_features = nb_features
        self.seq_length = seq_length
        if activation == 'sigmoid':
            activation_layer = torch.nn.Sigmoid()
        elif activation == 'relu':
            activation_layer = torch.nn.ReLU()
        elif activation == 'tanh':
            activation_layer = torch.nn.Tanh()

        dropout = torch.nn.Dropout(p=0.1)
        encoder_layers = []
        encoder_layers.append(torch.nn.Linear(nb_features * seq_length, hidden_units[0]))
        encoder_layers.append(activation_layer)
        encoder_layers.append(dropout)

        decoder_layers = []
        decoder_layers.append(torch.nn.Linear(latent_dim, hidden_units[-1]))
        decoder_layers.append(activation_layer)

        if nb_layers > 1:
            for layer in range(1, nb_layers):
                encoder_layers.append(torch.nn.Linear(hidden_units[layer - 1], hidden_units[layer]))
                encoder_layers.append(activation_layer)

                decoder_layers.append(torch.nn.Linear(hidden_units[-layer], hidden_units[-layer - 1]))
                decoder_layers.append(activation_layer)

        encoder_layers.append(torch.nn.Linear(hidden_units[-1], latent_dim))
        decoder_layers.append(torch.nn.Linear(hidden_units[0], nb_features * seq_length))

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoding(x)
        x = self.decoding(z)
        return x

    def encoding(self, x):
        x = x.reshape((-1, self.seq_length * self.nb_features))
        return self.encoder(x)

    def decoding(self, z):
        return self.decoder(z).reshape((-1, self.seq_length, self.nb_features))



