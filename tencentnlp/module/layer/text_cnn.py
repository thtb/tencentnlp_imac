import torch

from tencentnlp.config.config import ModuleConfigBase
from tencentnlp.config.config_property import Property


class TextCNN(torch.nn.Module):
    class Config(ModuleConfigBase):
        input_dim = Property(value_type=int)
        kernel_sizes = Property(value_type=list)
        num_kernel = Property(value_type=int)
        top_k = Property(value_type=int)

    def __init__(self, config: Config):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for kernel_size in config.kernel_sizes:
            self.convs.append(torch.nn.Conv1d(
                config.input_dim, config.num_kernel,
                kernel_size, padding=kernel_size - 1))

        self.top_k = self.config.TextCNN.top_k
        self.dropout = torch.nn.Dropout(p=config.dropout)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: Tensor with shape [batch, seq_len, input_dim],
                    always word/token embeddings
        Return:
            Tensor with shape [batch, kernel_sizes * num_kernel * top_k]
        """
        inputs = inputs.transpose(1, 2)
        pooled_outputs = []
        for i, conv in enumerate(self.convs):
            convolution = torch.nn.functional.relu(conv(inputs))
            pooled = torch.topk(convolution, self.top_k)[0].view(
                convolution.size(0), -1)
            pooled_outputs.append(pooled)

        doc_embedding = torch.cat(pooled_outputs, 1)
        return self.dropout(self.linear(doc_embedding))
