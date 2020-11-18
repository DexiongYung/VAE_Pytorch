import torch
import torch.nn as nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, device: str, args):
        super(VariationalAutoEncoder, self).__init__()
        self.sos_idx = args.sos_idx
        self.device = device
        self.encoder = Encoder(device, args)
        self.decoder = Decoder(device, args)
        self.to(device)

    def forward(self, X: torch.Tensor, X_lengths: torch.Tensor, max_len: int = None, is_teacher_force: bool = False):
        if max_len is None:
            max_len = X.shape[1]

        Z, mu, logit_sigma = self.encoder.forward(X, X_lengths)
        if is_teacher_force:
            logits, probs = self.decoder.forward(X, Z)
        else:
            decoder_input = torch.LongTensor([self.sos_idx]).to(self.device)
            logits, probs = self.decoder.forward(decoder_input, Z, max_len)
        return logits, probs, mu, logit_sigma

    def checkpoint(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class Encoder(nn.Module):
    def __init__(self, device: str, args):
        super(Encoder, self).__init__()
        self.vocab_size = len(args.vocab)
        self.hidden_size = args.RNN_hidden_size
        self.num_layers = args.num_layers
        self.word_embed_dim = args.word_embed_dim
        self.latent_size = args.latent_size
        # MLPs should take the hidden state size, which is num_layers * hidden_size
        self.mlp_input_size = self.hidden_size
        self.device = device
        self.char_embedder = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.word_embed_dim,
            padding_idx=args.pad_idx
        )
        self.gru = nn.GRU(self.word_embed_dim,
                          self.hidden_size, self.num_layers, batch_first=True)
        self.mu_mlp = NeuralNet(self.mlp_input_size, self.latent_size)
        self.sigma_mlp = NeuralNet(self.mlp_input_size, self.latent_size)

        # Molecular SMILES VAE initialized embedding to uniform [-0.1, 0.1]
        torch.nn.init.uniform_(self.char_embedder.weight,  -0.1, 0.1)

    def reparam_trick(self, mu: torch.Tensor, log_sigma: torch.Tensor):
        sd = torch.exp(0.5 * log_sigma)
        eps = 1e-2 * torch.randn_like(sd)
        sample = mu + (eps * sd)

        return sample

    def forward(self, X: torch.Tensor, X_lengths: torch.Tensor):
        batch_size = X.shape[0]
        X_embed = self.char_embedder(X)

        # Forward through X_pps to get hidden and cell states
        _, HC = self.gru(X_embed)


        # Get mu and sigma
        mu = self.mu_mlp(HC[0])
        logit_sigma = self.sigma_mlp(HC[0])

        # Use reparam trick to sample latents
        z = self.reparam_trick(mu, logit_sigma)

        return z, mu, logit_sigma


class Decoder(nn.Module):
    def __init__(self, device: str, args):
        super(Decoder, self).__init__()
        self.vocab_size = len(args.vocab)
        self.hidden_size = args.RNN_hidden_size
        self.num_layers = args.num_layers
        self.word_embed_dim = args.word_embed_dim
        self.latent_size = args.latent_size
        self.device = device
        self.pad_idx = args.pad_idx
        self.char_embedder = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.word_embed_dim,
            padding_idx=args.pad_idx
        )
        self.gru = nn.GRU(self.latent_size + self.word_embed_dim, self.hidden_size,
                          self.num_layers, batch_first=True)
        self.fc1 = NeuralNet(self.hidden_size, self.vocab_size)
        self.softmax = torch.nn.Softmax(dim=2)

        # Molecular SMILES VAE initialized embedding to uniform [-0.1, 0.1]
        torch.nn.init.uniform_(self.char_embedder.weight, -0.1, 0.1)

    def forward(self, X: torch.Tensor, Z: torch.Tensor, max_len: int = None):
        is_teacher_force = X is not None
        batch_size = Z.shape[0]

        # Embed input
        embeded_input = self.char_embedder(X)

        if is_teacher_force:
            max_len = X.shape[1]
            Z = Z.unsqueeze(1).repeat(1, max_len, 1)
            input = torch.cat((Z, embeded_input), dim=2)
            out, H = self.gru(input)
            # Reshape because LL is (batch, features)
            out_reshape = out.contiguous().view(-1, out.size(-1))
            fc1_outs = self.fc1(out_reshape)
            all_logits = fc1_outs.contiguous().view(out.size(0), -1, fc1_outs.size(-1))
        else:
            all_logits = None

            # All inputs should have Z appended to input
            input = torch.cat((Z, embeded_input.repeat(
                (batch_size, 1))), dim=1).unsqueeze(1)

            for i in range(max_len):
                lstm_out, HC = self.gru(input, HC)
                logits = self.fc1(lstm_out)
                max_chars = torch.argmax(logits, dim=2).squeeze(1)
                embeded_input = self.char_embedder(max_chars)
                input = torch.cat((Z, embeded_input), dim=1).unsqueeze(1)

                # Save logits for back prop
                if i == 0:
                    all_logits = logits
                else:
                    all_logits = torch.cat((all_logits, logits), dim=1)

        # Return logits and probs
        return all_logits, self.softmax(all_logits)


class NeuralNet(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(NeuralNet, self).__init__()
        self.ll = nn.Linear(input_size, output_size)
        self.selu = nn.SELU()

        # Molecular VAE initializes linear layer using Xavier
        torch.nn.init.xavier_uniform_(self.ll.weight)

    def forward(self, X: torch.Tensor):
        X = self.ll(X)
        return self.selu(X)
