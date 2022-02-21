import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ZERON_GCN

class MeshEncoder(nn.Module):
	def __init__(self, latent_length):
		super(MeshEncoder, self).__init__()
		self.h1 = ZERON_GCN(3, 60)
		self.h21 = ZERON_GCN(60, 60)
		self.h22 = ZERON_GCN(60, 60)
		self.h23 = ZERON_GCN(60, 60)
		self.h24 = ZERON_GCN(60,120)
		self.h3 = ZERON_GCN(120, 120)
		self.h4 = ZERON_GCN(120, 120)
		self.h41 = ZERON_GCN(120, 150)
		self.h5 = ZERON_GCN(150, 200)
		self.h6 = ZERON_GCN(200, 210)
		self.h7 = ZERON_GCN(210, 250)
		self.h8 = ZERON_GCN(250, 300)
		self.h81 = ZERON_GCN(300, 300)
		self.h9 = ZERON_GCN(300, 300)
		self.h10 = ZERON_GCN(300, 300)
		self.h11 = ZERON_GCN(300, 300)
		self.reduce = GCNMax(300,latent_length)

	def resnet( self, features, res):
		temp = features[:,:res.shape[1]]
		temp = temp + res
		features = torch.cat((temp,features[:,res.shape[1]:]), dim = 1)
		return features, features

	def forward(self, positions,  adj, play = False):
		res = positions
		features = self.h1(positions, adj, F.elu)
		features = self.h21(features, adj, F.elu)
		features = self.h22(features, adj, F.elu)
		features = self.h23(features, adj, F.elu)
		features = self.h24(features, adj, F.elu)
		features = self.h3(features, adj, F.elu)
		features = self.h4(features, adj, F.elu)
		features = self.h41(features, adj, F.elu)
		features = self.h5(features, adj, F.elu)
		features = self.h6(features, adj, F.elu)
		features = self.h7(features, adj, F.elu)
		features = self.h8(features, adj, F.elu)
		features = self.h81(features, adj, F.elu)
		features = self.h9(features, adj, F.elu)
		features = self.h10(features, adj, F.elu)
		features = self.h11(features, adj, F.elu)

		latent = self.reduce(features , adj, F.elu)

		return latent
