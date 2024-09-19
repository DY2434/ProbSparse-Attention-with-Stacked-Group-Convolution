import torch.nn as nn

class ConvolutionBlock1(nn.Module):
    def __init__(self, d_model, hidden_dim, kernel_size, dropout_rate, groups):
        super(ConvolutionBlock1, self).__init__()
        # 确保d_model和hidden_dim能够被组数整除
        assert d_model % groups == 0, "d_model must be divisible by groups"
        assert hidden_dim % groups == 0, "hidden_dim must be divisible by groups"

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=hidden_dim, kernel_size=kernel_size,
                               padding=kernel_size // 2, groups=groups)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=d_model, kernel_size=kernel_size,
                               padding=kernel_size // 2, groups=groups)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size,
                               padding=kernel_size // 2, groups=groups)

    def forward(self, x):
        x = x.transpose(1, 2)  # 转换为 (batch_size, channels, seq_length) 以适应一维卷积
        x = self.activation(self.conv1(x))
        x = self.channel_shuffle(x, groups=3)  # 确保这里的groups值与init中的一致
        x = self.activation(self.conv2(x))
        x = self.channel_shuffle(x, groups=3)
        x = self.dropout(self.conv3(x))
        x = x.transpose(1, 2)  # 转换回 (batch_size, seq_length, channels)
        return x

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, seq_length = x.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batch_size, groups, channels_per_group, seq_length)
        # transpose
        x = x.transpose(1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, seq_length)
        return x

class ConvolutionBlock2(nn.Module):
    def __init__(self, d_model, hidden_dim, kernel_size, dropout_rate, groups):
        super(ConvolutionBlock2, self).__init__()
        # 确保d_model和hidden_dim能够被组数整除
        assert d_model % groups == 0, "d_model must be divisible by groups"
        assert hidden_dim % groups == 0, "hidden_dim must be divisible by groups"

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=hidden_dim, kernel_size=kernel_size,
                               padding=kernel_size // 2, groups=groups)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=d_model, kernel_size=kernel_size,
                               padding=kernel_size // 2, groups=groups)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.activation(self.conv1(x)))
        x = self.dropout(self.activation(self.conv2(x)))
        return x