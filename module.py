import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim * 3
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim*3, hid_dim*3)
        self.w_k = nn.Linear(hid_dim*3, hid_dim*3)
        self.w_v = nn.Linear(hid_dim*3, hid_dim*3)
        self.fc = nn.Linear(hid_dim*3, hid_dim*3)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim*3 // n_heads])).to(device) # hid_dim//n_heads = 32

    def forward(self, query, key, value, mask=None):

        bsz = query.shape[0]
        # query = key = value = [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, sent len_Q, sent len_K]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.hid_dim)
        # x = [batch size, sent len_Q, hid dim] [8, 145, 768]

        x = self.fc(x)
        # x = [batch size, sent len_Q, hid dim] [8, 145, 768]

        return x

class GatedCon(nn.Module):
    """protein feature extraction."""

    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device

        # self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs1 = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in
             range(self.n_layers)])  # convolutional layers1
        self.convs2 = nn.ModuleList([nn.Conv1d(hid_dim, 2 * hid_dim, 5, padding=(5 - 1) // 2) for _ in
                                     range(self.n_layers)])  # convolutional layers2
        self.convs3 = nn.ModuleList([nn.Conv1d(hid_dim, 2 * hid_dim, 7, padding=(7 - 1) // 2) for _ in
                                     range(self.n_layers)])  # convolutional layers3
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.ln = nn.LayerNorm(hid_dim * 3)

    def forward(self, protein):

        conv_input = self.fc(protein)
        # conv_input=[batch size,protein len,hid dim] 64
        # permute for convolutional layer
        conv_input1 = conv_input.permute(0, 2, 1)
        conv_input2 = conv_input.permute(0, 2, 1)
        conv_input3 = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs1):
            conved = (F.glu(conv(self.dropout(conv_input1)), dim=1) + conv_input1) * self.scale
            conv_input1 = conved

        for i, conv in enumerate(self.convs2):
            conved = (F.glu(conv(self.dropout(conv_input2)), dim=1) + conv_input2) * self.scale
            conv_input2 = conved

        for i, conv in enumerate(self.convs3):
            conved = (F.glu(conv(self.dropout(conv_input3)), dim=1) + conv_input3) * self.scale
            conv_input3 = conved

        conved = torch.cat((conv_input1, conv_input2, conv_input3), 1)
        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_key=64, dim_value=64, dropout=0.):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)
        self.to_out = nn.Linear(dim_value * heads, dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, mask=None):
        n, h = x.shape[-2], self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        q = q * self.scale
        logits = einsum('b h i d, b h j d -> b h i j', q, k)

        if mask is not None:
            logits.masked_fill(mask == 0, -1e9)

        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)#, attn


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim*3, pf_dim*3, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim*3, hid_dim*3, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]
        # x = [batch size, pf dim, sent len]
        x = self.fc_2(self.do(F.relu(self.fc_1(x))))
        # x = [batch size, hid dim, sent len]
        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]
        return x
class Self_Attention(nn.Module):
    def __init__(self, num_hidden, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(num_hidden / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        q = self.transpose_for_scores(q)  # [bsz, heads, protein_len, hid]
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        if mask is not None:
            attention_mask = (1.0 - mask) * -10000
            att = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(
                1)  # [bsz, heads, protein_len, protein_len] + [bsz, 1, 1, protein_len]

        attention_scores = nn.Softmax(dim=-1)(attention_scores)

        outputs = torch.matmul(attention_scores, v)

        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        outputs = outputs.view(*new_output_shape)
        return outputs


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.leaky_relu(self.W_in(h_V))
        h = self.W_out(h)
        return h


class TransformerLayer(nn.Module):
    def __init__(self, num_hidden=64, num_heads=4, dropout=0.2):
        super(TransformerLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden, eps=1e-6) for _ in range(2)])

        self.attention = Self_Attention(num_hidden, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, mask=None):
        # Self-attention
        dh = self.attention(h_V, h_V, h_V, mask)
        # print(f'dh: {dh}')
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask is not None:  # mask掉padding的节点
            mask = mask.unsqueeze(-1)
            h_V = mask * h_V
        return h_V