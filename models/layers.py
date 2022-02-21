import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ZERON_GCN(nn.Module):
	def __init__(self, in_features, out_features, bias=True):
		super(ZERON_GCN, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.Tensor(in_features, out_features))

		if bias:
			self.bias = Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 6. / math.sqrt(self.weight.size(1) + self.weight.size(0))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-0, 0)

	def forward(self, input, adj, activation):
		support = torch.mm(input, self.weight)
		output = torch.cat((torch.mm(adj, support[:, :support.shape[1]//10]), support[:, support.shape[1]//10:]), dim = 1)

		if self.bias is not None:
			output = output + self.bias
		return activation(output)
