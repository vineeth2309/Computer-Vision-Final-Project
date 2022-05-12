import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim
from Models import Unet
from DataSetClass import DatasetClass
from LossFunctions import *
from EvaluationMetrics import *



class Main:
	def __init__(self, train=True, test=True, eval=True, train__dir='./data/train', test_dir='./data/test', val_dir="./data/val", epochs=3,
				 learning_rate=1e-3, reg=1e-3, batch_size=16, num_workers=2, load_model_params=True, save_model_params=True,
				 saved_params_path="models/focalloss_earlystop.pt", loss_fn='Focal', tensorboard_log='runs/test_1',
				 model_arch='Unet', aug=True):

		# initialize class properties from arguments
		self.train_dir, self.test_dir = train__dir, test_dir
		self.epochs, self.lr, self.reg = epochs, learning_rate, reg
		self.batch_size, self.num_workers = batch_size, num_workers
		self.val_dir = val_dir
		self.save_model_params = save_model_params
		self.saved_params_path = saved_params_path
		self.model_arch = model_arch
		self.aug = aug

		# determine if GPU can be used and assign to 'device' class property
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# set properties relating to dataset
		self.train_dataset_size = len(glob.glob(os.path.join(self.train_dir, 'image', "*.png")))
		self.train_dataset = DatasetClass(self.train_dir, flag='train', aug=self.aug)
		self.val_dataset = DatasetClass(self.val_dir,flag='val')
		self.test_dataset = DatasetClass(self.test_dir,flag='test')

		# load data ready for training
		self.train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True,
										   num_workers=self.num_workers, pin_memory=True, drop_last=False)
		self.val_dataloader = DataLoader(self.val_dataset, self.batch_size, shuffle=True,
										   num_workers=1, pin_memory=True, drop_last=False)
		self.test_dataloader = DataLoader(self.test_dataset, self.batch_size, shuffle=False,
										   num_workers=1, pin_memory=True, drop_last=False)

		# create the model
		self.model = Unet(image_channels=1, hidden_size=32).to(self.device)

		# load model parameters from file
		if load_model_params and os.path.isfile(saved_params_path):
			self.model.load_state_dict(torch.load(saved_params_path, map_location=self.device))

		# set loss function
		self.loss = DiceLoss()

		# set optimiser
		self.optim = optim.Adam(self.model.parameters(), self.lr, weight_decay=self.reg)

		# evaluation metrics
		self.eval_metrics = Evaluation_Metrics()

		if train:
			self.writer = SummaryWriter(tensorboard_log)
			self.train()		# carry out training
			self.writer.close()

		if test:
			self.model_test()  # Test forward pass of model

	def calculatescores(self, gt, pred):
		gt_classes = len(np.unique(gt))
		iou_scores = []
		precision_scores = []
		recall_scores = []
		f1_scores = []
		ssim_scores = []
		acc_scores = []
		gt_mask = np.zeros((gt.shape[0], gt.shape[1], len(np.unique(gt))))
		for i in range(len(np.unique(gt))):
			gt_mask[:, :, i][np.where(gt == i)] = 255
		for i in range(gt_classes):
			target = gt_mask[:,:,i]
			prediction = pred[:,:,i]
			ssim_score = ssim(target, prediction)
			TP = np.sum(np.logical_and(prediction == 255, target == 255))
			TN = np.sum(np.logical_and(prediction == 0, target == 0))
			FP = np.sum(np.logical_and(prediction == 255, target == 0))
			FN = np.sum(np.logical_and(prediction == 0, target == 255))
			intersection = np.logical_and(target, prediction)
			union = np.logical_or(target, prediction)
			iou_score = np.sum(intersection) / np.sum(union)
			precision = TP / (TP+FP)
			recall = TP / (TP+FN)
			f1 = (2 * precision * recall) / (precision + recall)
			acc = (TP+TN)/(TP+FP+FN+TN)
			iou_scores.append(iou_score)
			precision_scores.append(precision)
			recall_scores.append(recall)
			f1_scores.append(f1)
			ssim_scores.append(ssim_score)
			acc_scores.append(acc)
		return iou_scores, precision_scores, recall_scores, f1_scores, ssim_scores, acc_scores

	def train(self):
		"""Carry out training on the model"""
		train_losses = []
		val_losses = []
		early_stopping_check = None
		patience, best_score= 5, 0
		self.model.train()
		for epoch in range(self.epochs):
			train_epoch_loss = 0
			val_epoch_loss = 0
			val_scores = []
			for batch_idx1, (data, label) in enumerate(self.train_dataloader):
				data, label = data.to(self.device), label.to(self.device)
				out = self.model(data) # forward pass
				loss = self.loss(out, label) # loss calculation
				self.optim.zero_grad() # back-propogation
				loss.backward()
				self.optim.step()
				train_epoch_loss += loss

			for batch_idx2, (data, label) in enumerate(self.val_dataloader):
				data, label = data.to(self.device), label.to(self.device)
				with torch.no_grad():
					out = self.model(data) # forward pas
					loss = self.loss(out, label) # val loss calculation
					val_epoch_loss += loss
					val_score = self.eval_metrics.evaluate(out, label)
					val_scores.append(val_score)

			train_epoch_loss = train_epoch_loss.cpu().item() / max(1, batch_idx1)
			val_epoch_loss = val_epoch_loss.cpu().item() / max(1, batch_idx2)
			train_losses.append(train_epoch_loss)
			val_losses.append(val_epoch_loss)
			val_scores = np.mean(np.array(val_scores), axis=0)

			self.writer.add_scalar("Train Loss", train_epoch_loss, epoch)
			self.writer.add_scalar("Val Loss", val_epoch_loss, epoch)
			self.writer.add_scalars('Mean Scores', {'Dice Score':np.mean(val_scores[0]),
								'Accuracy':np.mean(val_scores[1]),
								'IOU': np.mean(val_scores[2])}, epoch)

			for i in range(val_scores.shape[1]):
				self.writer.add_scalars('Class{}'.format(str(i)), {'Dice Score':val_scores[0][i],
									'Accuracy':val_scores[1][i],
									'IOU': val_scores[2][i]}, epoch)


			print("EPOCH {}: ".format(epoch), train_epoch_loss, val_epoch_loss)
			torch.save(self.model.state_dict(), self.saved_params_path)

	def model_test(self):
		"""display performance on each of validation data"""
		self.img_files = glob.glob(os.path.join('data/train', 'image', '*.png'))
		self.mask_files = []
		for img_path in self.img_files:
			basename = os.path.basename(img_path)
			self.mask_files.append(os.path.join('data/train', 'mask', basename[:-4]+'.png'))
		IOU, Precision, Recall, F1, Ssim, acc = [],[],[],[],[],[]
		for img_file,label_file in zip(self.img_files,self.mask_files):
			data = cv2.resize(cv2.imread(img_file, cv2.IMREAD_UNCHANGED),(224,224))
			label = cv2.resize(cv2.imread(label_file, cv2.IMREAD_UNCHANGED),(224,224))
			data = np.expand_dims(np.expand_dims(data, 0),0)
			self.img_tensor = torch.from_numpy(data).float()/255
			out = self.model(self.img_tensor)
			out = F.softmax(out, 1).permute(0, 2, 3, 1)
			out = self.one_hot(out).cpu().numpy()[0]*255
			iou_scores, precision_scores, recall_scores, f1_scores, ssim_scores, acc_scores = self.calculatescores(label, out)
			IOU.append(iou_scores)
			Precision.append(precision_scores)
			Recall.append(recall_scores)
			F1.append(f1_scores)
			Ssim.append(ssim_scores)
			acc.append(acc_scores)
			out_img = np.hstack((
								 out[:, :, 0].astype(np.uint8),
								 out[:, :, 1].astype(np.uint8),
								 out[:, :, 2].astype(np.uint8),
								 out[:, :, 3].astype(np.uint8),
								 out[:, :, 4].astype(np.uint8),
								 out[:, :, 5].astype(np.uint8)))

			# process mask to produce mirror stacked image as above, with features in the same places
			out_actual = np.zeros((label.shape[0], label.shape[1], len(np.unique(label))))
			for i in range(len(np.unique(label))):
				out_actual[:, :, i][np.where(label == i)] = 255
			out_actual_img = np.hstack((
										out_actual[:, :, 0].astype(np.uint8),
										out_actual[:, :, 1].astype(np.uint8),
										out_actual[:, :, 2].astype(np.uint8),
										out_actual[:, :, 3].astype(np.uint8),
										out_actual[:, :, 4].astype(np.uint8),
										out_actual[:, :, 5].astype(np.uint8)))

			# display both images
			cv2.imshow("MODEL OUTPUT", out_img)
			cv2.imshow("ACTUAL MASK", out_actual_img)
			k = cv2.waitKey()
			if k == ord('q'):
				exit()
		mean_score = (np.mean(IOU,axis=0) + np.mean(F1,axis=0) + np.mean(Ssim,axis=0) + np.mean(acc,axis=0))/4
		print("IOU: ", np.mean(IOU,axis=0),
		"PRECISION: ", np.mean(Precision,axis=0),
		"RECALL: ", np.mean(Recall,axis=0),
		"F1: ", np.mean(F1,axis=0),
		"SSIM: ", np.mean(Ssim,axis=0),
		"ACC: ", np.mean(acc,axis=0),
		"MEAN: ", mean_score,
		"MEAN VAL: ", np.mean(mean_score))

	def one_hot(self, masks):
		return F.one_hot(torch.argmax(masks, axis=-1))

	def un_one_hot(self, masks):
		return torch.argmax(masks, axis=-1)

if __name__ == '__main__':
	Main(load_model_params=True, save_model_params=True, saved_params_path="models/unettrainval.pt", train=False, test=True, epochs = 30,
		tensorboard_log='runs/new', aug=True)
