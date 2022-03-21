import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.ModuleList):
    def __init__(self, hidden_dim = 768, vocab_size = 50257):
        super(BiLSTM, self).__init__()

        """
        batch_size: là batch_size
        hidden_dim: là số chiều của vector embedding
        sequence_len: là chiều dài chủa chuỗi, tức số vector embedding trong một chuỗi
        vocab_size: len(vocab)
        """
        self.hidden_dim = hidden_dim
        self.num_classes = vocab_size

        # Bi-LSTM
		# Forward and backward
        self.lstm_cell_forward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.lstm_cell_backward = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        #LSTM_layer 
        self.lstm_cell = nn.LSTMCell(self.hidden_dim * 2, self.hidden_dim * 2)
        # Linear layer
        self.linear = nn.Linear(self.hidden_dim * 2, self.num_classes)
        # Vì concate embedding vector hai chiều xuôi và ngược nên lúc này số chiều nhân 2

    def forward(self, x):
        """
        x: tensor([batch, seq_length, hidden_dim])
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_size = x.size(0)
        sequence_len = x.size(1)
        # Khởi tạo các giá trị đầu
        hs_lstm = torch.zeros(x.size(0), self.hidden_dim * 2)
        cs_lstm = torch.zeros(x.size(0), self.hidden_dim * 2)

        # Bi-LSTM
		# hs = [batch_size x hidden_size]
		# cs = [batch_size x hidden_size]
        hs_forward = torch.zeros(x.size(0), self.hidden_dim)
        cs_forward = torch.zeros(x.size(0), self.hidden_dim)
        hs_backward = torch.zeros(x.size(0), self.hidden_dim)
        cs_backward = torch.zeros(x.size(0), self.hidden_dim)

        # Weights initialization
        torch.nn.init.kaiming_normal_(hs_forward)
        torch.nn.init.kaiming_normal_(cs_forward)
        torch.nn.init.kaiming_normal_(hs_backward)
        torch.nn.init.kaiming_normal_(cs_backward)
        torch.nn.init.kaiming_normal_(hs_lstm)
        torch.nn.init.kaiming_normal_(cs_lstm)

        hs_forward = hs_forward.to(device)
        cs_forward = cs_forward.to(device)
        hs_backward = hs_backward.to(device)
        cs_backward = cs_backward.to(device)
        hs_lstm = hs_lstm.to(device)
        cs_lstm = cs_lstm.to(device)

        out = x.view(sequence_len, x.size(0), -1)

        forward = []
        backward = []

        # Forward
        for i in range(sequence_len):
            hs_forward, cs_forward = self.lstm_cell_forward(out[i], (hs_forward, cs_forward))
            forward.append(hs_forward)
		 	
        # Backward
        for i in reversed(range(sequence_len)):
            hs_backward, cs_backward = self.lstm_cell_backward(out[i], (hs_backward, cs_backward))
            backward.append(hs_backward)
        

        for fwd, bwd in zip(forward, backward):
            input_tensor = torch.cat((fwd, bwd), 1)
            hs_lstm, cs_lstm = self.lstm_cell(input_tensor, (hs_lstm, cs_lstm))

        # Last hidden state is passed through a linear layer
        out = self.linear(hs_lstm)

        return out
        