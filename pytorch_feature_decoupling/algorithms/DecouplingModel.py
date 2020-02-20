import pdb

import os
import time
import numpy as np

import torch

from . import Algorithm

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

class DecouplingModel(Algorithm):
	def __init__(self, opt):
		self.lambda_loss = opt['lambda_loss']
		self.gama = opt['gama']

		##########################
		self.margin = 0.0
		self.beta = 1
		# self.small_vec_size = 128
		# self.mu_lin = torch.nn.Linear(self.small_vec_size, self.small_vec_size).to('cuda:0')
		# self.std_lin = torch.nn.Linear(self.small_vec_size, self.small_vec_size).to('cuda:0')
		# self.mu_opt = torch.optim.SGD(self.mu_lin.parameters(), lr=0.1, momentum=0.9, nesterov=True)
		# self.std_opt = torch.optim.SGD(self.std_lin.parameters(), lr=0.1, momentum=0.9, nesterov=True)
		##########################

		Algorithm.__init__(self, opt)

	def loadPUImageProb(self):
		with open(os.path.join(self.exp_dir, 'prob', 'prob.dat'), 'r') as file_input:
			train_prob_str = file_input.readlines()
			train_prob = [float(i_prob_str.rstrip('\n')) for i_prob_str in train_prob_str]
		self.train_weight = [1.0 if 0==i%4 else 1-train_prob[i]**self.gama for i in range(len(train_prob))]

	def allocate_tensors(self):
		self.tensors = {}
		self.tensors['dataX'] = torch.FloatTensor()
		self.tensors['index'] = torch.LongTensor()
		self.tensors['index_index'] = torch.LongTensor()
		self.tensors['labels'] = torch.LongTensor()
		self.tensors['class_labels'] = torch.LongTensor()
		self.loadPUImageProb()

	def create_triplets(self, labels):
		triplet_indices = torch.IntTensor(size=(labels.size(0), 3)).to('cuda:0')
		triplet_labels = torch.IntTensor(size=(labels.size(0), 3)).to('cuda:0')
		labels_set = set(labels.cpu().numpy())
		label_to_indices = {label: np.where(labels.cpu().numpy() == label)[0] for label in labels_set}

		for i, label in enumerate(labels):
			label = label.item()
			pos_label = label
			anchor_idx = i
			pos_idx = i
			while pos_idx == i:
				pos_idx = np.random.choice(label_to_indices[label])
			neg_label = np.random.choice(list(labels_set - {label}))
			neg_idx = np.random.choice(label_to_indices[neg_label])
			triplet_indices[i] = torch.tensor([anchor_idx, pos_idx, neg_idx]).to('cuda:0')
			triplet_labels[i] = torch.tensor([labels[anchor_idx], labels[pos_idx], labels[neg_idx]]).to('cuda:0')

		return triplet_indices, triplet_labels

	def compute_batch_triplet(self, norm_features, labels, margin):
		triplet_loss = 0
		triplet_indices, triplet_labels = self.create_triplets(labels)
		# self.logger.info('TRIPLETS {} {} {} {}'.format(str(triplet_indices.size()), str(triplet_indices.tolist()),
		#                                                str(triplet_labels.size()), str(triplet_labels.tolist())))

		anchor = []
		positive = []
		negative = []
		for (anchor_idx, pos_idx, neg_idx) in triplet_indices:
			anchor.append(norm_features[anchor_idx])
			positive.append(norm_features[pos_idx])
			negative.append(norm_features[neg_idx])
		anchor = torch.stack(anchor).to('cuda:0')
		positive = torch.stack(positive).to('cuda:0')
		negative = torch.stack(negative).to('cuda:0')

		distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
		distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
		losses = torch.nn.functional.relu(distance_positive - distance_negative + margin)
		triplet_loss = losses.mean()

		# self.logger.info('Triplet stuff: {} {} {} {} {} {}', str(anchor.size()), str(distance_positive.size()),
		#                  str(distance_positive.tolist()), str(distance_negative.tolist()), str(losses.size()),
		#                  str(triplet_loss.item()))
		return triplet_loss


	def train_step(self, batch):
		start = time.time()
		#*************** LOAD BATCH (AND MOVE IT TO GPU) ********
		self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
		dataX_90 = torch.flip(torch.transpose(self.tensors['dataX'],2,3),[2])
		dataX_180 = torch.flip(torch.flip(self.tensors['dataX'],[2]),[3])
		dataX_270 = torch.transpose(torch.flip(self.tensors['dataX'],[2]),2,3)

		dataX = torch.stack([self.tensors['dataX'], dataX_90, dataX_180, dataX_270], dim=1)
		batch_size, rotations, channels, height, width = dataX.size()
		dataX = dataX.view([batch_size*rotations, channels, height, width])

		self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
		self.tensors['index'].resize_(batch[2].size()).copy_(batch[2])

		self.tensors['class_labels'].resize_(batch[3].size()).copy_(batch[3])

		labels = self.tensors['labels']
		index = self.tensors['index']
		class_labels = self.tensors['class_labels']
		# self.logger.info('Class labels {} {}'.format(str(class_labels.size()), str(class_labels.tolist())))

		idx_train = 4*batch[2].numpy()
		idx_train[1::4] += 1
		idx_train[2::4] += 2
		idx_train[3::4] += 3
		#********************************************************
		batch_load_time = time.time() - start

		start = time.time()
		#************ FORWARD THROUGH NET ***********************
		for _, network in self.networks.items():
			for param in network.parameters():
				param.requires_grad = True

		with torch.set_grad_enabled(True):
			feature = self.networks['feature'](dataX)
			feature_rot, feature_invariance = torch.split(feature, 2048, dim=1)

			pred = self.networks['classifier'](feature_rot)

			feature_invariance_instance = feature_invariance[0::4,:] + feature_invariance[1::4,:] + feature_invariance[2::4,:] + feature_invariance[3::4,:]
			feature_invariance_instance = torch.mul(feature_invariance_instance, 0.25)
			feature_nce_norm = self.networks['norm'](feature_invariance_instance)

			########################
			# mu = self.mu_lin(feature_nce_norm.clone())
			# std = self.std_lin(feature_nce_norm.clone())
			# # mu, std = vec[:, :self.small_vec_size], vec[:, self.small_vec_size:]
			# std = std.mul(0.5).exp()
			# epsilon = torch.randn_like(std).to('cuda:0')
			# mytensor = mu + epsilon * std
			# # feature_nce_norm = mu + epsilon * std
			triplet_loss = self.compute_batch_triplet(feature_nce_norm, class_labels, margin=self.margin)

			########################

		with torch.set_grad_enabled(False):
			self.tensors['index_index'].resize_(torch.Size([int(index.size(0)/4)])).copy_(index[0::4])
			index_instance = self.tensors['index_index']
			feature_invariance_instance_mean = torch.unsqueeze(feature_invariance_instance,1).expand(-1,4,-1).clone()
			feature_invariance_instance_mean = feature_invariance_instance_mean.view(4*len(feature_invariance_instance),2048)
		#********************************************************

		#*************** COMPUTE LOSSES *************************
		weight = torch.tensor(np.array(self.train_weight)[idx_train], dtype=torch.float, device=labels.device, requires_grad=False)
		with torch.set_grad_enabled(True):
			loss_cls_each = self.criterions['loss_cls'](pred, labels)
			if self.curr_epoch<210:
				loss_cls = torch.sum(loss_cls_each)/loss_cls_each.shape[0]
			else:
				loss_cls = torch.dot(loss_cls_each,weight)/loss_cls_each.shape[0]

			loss_mse = self.criterions['loss_mse'](feature_invariance, feature_invariance_instance_mean)

			output_nce = self.criterions['nce_average'](feature_nce_norm, index_instance)
			loss_nce, kl_loss = self.criterions['nce_criterion'](output_nce, index_instance)

			# print('kl', kl_loss)
			loss_total = self.lambda_loss['cls']*loss_cls + self.lambda_loss['mse']*loss_mse + \
						 self.lambda_loss['nce']*loss_nce + self.beta * kl_loss + triplet_loss

		record = {}
		record['prec_cls'] = accuracy(pred, labels, topk=(1,))[0].item()

		record['loss'] = loss_total.item()
		record['loss_cls'] = loss_cls.item()
		record['loss_mse'] = loss_mse.item()
		record['loss_nce'] = loss_nce.item()
		record['loss_triplet'] = triplet_loss.item()
		#********************************************************

		#****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
		self.optimizers['feature'].zero_grad()
		self.optimizers['classifier'].zero_grad()
		self.optimizers['norm'].zero_grad()
		# self.mu_opt.zero_grad()
		# self.std_opt.zero_grad()
		loss_total.backward()
		self.optimizers['feature'].step()
		self.optimizers['classifier'].step()
		self.optimizers['norm'].step()
		# self.mu_opt.step()
		# self.std_opt.step()
		#********************************************************
		batch_process_time = time.time() - start
		total_time = batch_process_time + batch_load_time
		record['load_time'] = 100*(batch_load_time/total_time)
		record['process_time'] = 100*(batch_process_time/total_time)

		return record

	def evaluation_step(self, batch):
		start = time.time()
		#*************** LOAD BATCH (AND MOVE IT TO GPU) ********
		self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])
		dataX_90 = torch.flip(torch.transpose(self.tensors['dataX'],2,3),[2])
		dataX_180 = torch.flip(torch.flip(self.tensors['dataX'],[2]),[3])
		dataX_270 = torch.transpose(torch.flip(self.tensors['dataX'],[2]),2,3)

		dataX = torch.stack([self.tensors['dataX'], dataX_90, dataX_180, dataX_270], dim=1)
		batch_size, rotations, channels, height, width = dataX.size()
		dataX = dataX.view([batch_size*rotations, channels, height, width])

		self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
		#********************************************************
		batch_load_time = time.time() - start

		start = time.time()
		#************ FORWARD THROUGH NET ***********************
		for _, network in self.networks.items():
			for param in network.parameters():
				param.requires_grad = False

		with torch.set_grad_enabled(False):
			feature = self.networks['feature'](dataX)
			feature_rot, feature_invariance = torch.split(feature, 2048, dim=1)
			pred_rot = self.networks['classifier'](feature_rot)
			pred_inv = self.networks['classifier'](feature_invariance)
		#********************************************************

		#*************** COMPUTE LOSSES *************************
		with torch.set_grad_enabled(False):
			loss_rot_each = self.criterions['loss_cls'](pred_rot, self.tensors['labels'])
			loss_inv_each = self.criterions['loss_cls'](pred_inv, self.tensors['labels'])
			loss_rot = torch.sum(loss_rot_each)/loss_rot_each.shape[0]
			loss_inv = torch.sum(loss_inv_each)/loss_inv_each.shape[0]
		record = {}
		record['prec_rot'] = accuracy(pred_rot, self.tensors['labels'], topk=(1,))[0].item()
		record['prec_inv'] = accuracy(pred_inv, self.tensors['labels'], topk=(1,))[0].item()
		record['loss_rot'] = loss_rot.item()
		record['loss_inv'] = loss_inv.item()
		#********************************************************
		batch_process_time = time.time() - start
		total_time = batch_process_time + batch_load_time
		record['load_time'] = 100*(batch_load_time/total_time)
		record['process_time'] = 100*(batch_process_time/total_time)

		return record
