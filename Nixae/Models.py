''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BN_RELU(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.bn = nn.BatchNorm1d(channel, track_running_stats=False)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        outputs = self.bn(inputs)
        outputs = self.relu(outputs)
        return outputs


class Nixae(nn.Module):

    def __init__(
            self, len_max_seq, brn, d_model, label=8, dropout=0.5, device='cpu'):
        super().__init__()

        self.brn = brn
        self.d_model = d_model
        self.device = device
        self.C_0 = 32

        n_position = len_max_seq + 1

        self.first_conv = nn.Conv2d(1, self.C_0, (1, 256), stride=1)
        self.inception_conv = nn.Conv1d(d_model, self.C_0, 1, stride=1, padding=(1 // 2))
        self.chose_conv = nn.Conv1d(self.C_0, brn, 1, stride=1, padding=(1 // 2))


        self.bn_norm_1 = BN_RELU(self.C_0)
        self.bn_norm_2 = BN_RELU(self.C_0)
        self.bn_norm_3 = BN_RELU(self.C_0)

        # add to __init__()
        self.dense_alpha1 = nn.Linear(self.brn, self.brn * self.brn, bias=False)
        self.dense_alpha2 = nn.Linear(self.brn * self.brn, self.brn)

        self.brn_convs = nn.ModuleList(
            [nn.Conv1d(self.C_0, self.d_model, i + 1, stride=1, padding=(i // 2)) for i in range(self.brn)])

        #AvgPool
        self.avgPooling = nn.AvgPool1d(2, 2)

        # dropout layer
        self.dropout = nn.Dropout(p=dropout)
        self.chose_softmax = nn.Softmax(dim=1)
        self.dense_layer1 = nn.Linear(int(len_max_seq  * self.C_0), 256)
        self.dense_layer2 = nn.Linear(256, 128)
        self.dense_layer3 = nn.Linear(128, label)


    def forward(self, src_seq, return_attns=False):
        # -- Preparing
        enc_output = F.one_hot(src_seq, num_classes=256)
        enc_output = enc_output.unsqueeze(1).float()

        inputs = self.first_conv(enc_output)
        inputs = inputs.squeeze(3)  # x

        # -- Weight Network
        branch_chose = self.chose_conv(inputs)
        branch_chose = branch_chose.unsqueeze(2)
        branch_chose = self.chose_softmax(branch_chose)


        # --  Inception Network
        branch_stack = []
        for i in range(1, self.brn+1):
            if i % 2 != 0:
                branch_1 = F.relu(self.brn_convs[i - 1](inputs))
                branch_1 = self.bn_norm_2(self.inception_conv(branch_1))
                branch_stack.append(branch_1)
            else:
                p1d = (0, 1)  # pad last dim on the right side
                pad_input = F.pad(inputs, p1d, "constant", 0)
                branch_2 = F.relu(self.brn_convs[i - 1](pad_input))
                branch_2 = self.bn_norm_2(self.inception_conv(branch_2))
                branch_stack.append(branch_2)
        branch_stack = torch.stack(branch_stack, 1)
        
        # --  Combine
        branch_now = torch.mul(branch_stack, branch_chose)
        branch_now = torch.sum(branch_now, 1)  # F(x)

        # -- ResNet
        resnet_output = self.bn_norm_3(branch_now + inputs)

        # --  Fully Connected Layers
        fully = resnet_output.reshape([resnet_output.shape[0], -1])
        out = self.dropout(fully)

        out = F.relu(self.dense_layer1(out))
        out = F.relu(self.dense_layer2(out))
        final_output = self.dense_layer3(out)
        return final_output



