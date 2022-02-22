import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BatchZERON_GCN(nn.Module):
	def __init__(self, in_features, out_features, bias=True):
		super(BatchZERON_GCN, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.Tensor(in_features, out_features))

		if bias:
			self.bias = nn.Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 6. / math.sqrt(self.weight.size(1) + self.weight.size(0))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-0, 0)

	def forward(self, input, adj, activation):
		support = torch.matmul(input, self.weight.unsqueeze(0))
		output = torch.matmul(adj, support[:,:,:support.shape[-1]//10])
		output = torch.cat((output, support[:,:, support.shape[-1]//10:]), dim = -1)
		
		if self.bias is not None:
			output = output + self.bias
		return activation(output)

class BatchGCNMax(nn.Module):
	def __init__(self, in_features, print_length):
		super(BatchGCNMax, self).__init__()
		self.in_features = in_features
		self.print_length = print_length
		self.weight_Ws = nn.ParameterList(nn.Parameter(torch.Tensor(in_features, print_length)) for i in range(1))
		self.weight_Bs = nn.ParameterList(nn.Parameter(torch.Tensor(print_length)) for i in range(1))
		self.reset_parameters()

	def reset_parameters(self):
		for i in range(1):
			stdv = 6. / math.sqrt(self.weight_Bs[i].size(0))

			self.weight_Bs[i].data.uniform_(-stdv, stdv)
			stdv = 6. / math.sqrt(self.weight_Ws[i].size(0) + self.weight_Ws[i].size(1))
			self.weight_Ws[i].data.uniform_(-stdv, stdv)
			
	def forward(self, r_s, adj, activation):
		bias = self.weight_Bs[0]
		weight_W = self.weight_Ws[0]
		
		support = torch.matmul(r_s, weight_W.unsqueeze(0))
		output = torch.matmul(adj, support[:,:,:support.shape[-1]//10])
		output = torch.cat((output, support[:,:, support.shape[-1]//10:]), dim = -1)

		v_s = output + bias
		i_s = activation(v_s)      
		f   = torch.max(v_s, dim = 1)[0]       
	
		return f                      