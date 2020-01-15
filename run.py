from __future__ import print_function

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from dataset import *
from sampling_metrics import compute_train_test_scores, compute_iw_scores, compute_bias_variance
from uncertainty_utils import extended_calibration_curve, plot_calibration
from sklearn.metrics import mean_squared_error


class FeatureCritic(nn.Module):
    def __init__(self, use_mlp):

        super(FeatureCritic, self).__init__()

        self.use_mlp = use_mlp
        self.inception = models.inception_v3(
                pretrained=True, transform_input=False)
        self.fc = self.inception.fc
        self.fc2 = None

        if use_mlp:
            self.fc2 = nn.Linear(1000, 1)


    def forward(self, input):

        if self.training and self.inception.aux_logits:
            logits, _ = self.inception(input)
        else:
            logits = self.inception(input)

        if self.use_mlp:
            output = self.fc2(F.relu(logits))
        else:
            output = logits[:, 0]

        return output



def train(args, model, device, train_loader, optimizer, epoch):

	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data).squeeze()
		loss = F.binary_cross_entropy_with_logits(output, target)
		loss.backward()
		optimizer.step()

		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()), 
			flush=True)

def test(args, model, device, test_loader, validation=True):

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item() # sum up batch loss
            pred = torch.round(torch.sigmoid(output))
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    if validation:
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc),
        flush=True)
    return test_loss, acc

def test_calibration(args, model, device, data_loader, n_bins=10, savefile=None):

    model.eval()
    pred_probs = []
    targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            pred_probs.append(torch.sigmoid(output))
            targets.append(target)
    pred_probs = torch.cat(pred_probs).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    fraction_of_positives, mean_predicted_value, bin_weights = extended_calibration_curve(targets, pred_probs, n_bins=n_bins)
    calibration_error = mean_squared_error(fraction_of_positives, mean_predicted_value, sample_weight=bin_weights)
    print('Calibration error', calibration_error)

    if savefile is not None:
        cal_savefile = os.path.join(args.sampledir, 'calib_'+savefile[:-3]) 
        plot_calibration(fraction_of_positives, mean_predicted_value, cal_savefile)
    return calibration_error

def main():

	# Training settings
	parser = argparse.ArgumentParser(description='Debiasing generative models via importance weighting')
	parser.add_argument('--sampledir', type=str, default='./samples/pixelcnnpp')
	parser.add_argument('--modeldir', type=str, default='./models/pixelcnnpp')
	parser.add_argument('--datadir', type=str, default='./datasets/')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
						help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--log-interval', type=int, default=100, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--train', dest='train', action='store_true', default=False,
						help='trains a model from scratch if true')
	parser.add_argument('--use-feature', dest='use_feature', action='store_true', default=False,
						help='trains a model on inception v3 features if true')
	parser.add_argument('--use-mlp', dest='use_mlp', action='store_true', default=False,
                        help='adds a hidden layer on top of inception v3 features')

	parser.add_argument('--self-norm', dest='self_norm', action='store_false', default=True,
						help='self normalize importance weights')
	parser.add_argument('--flatten', dest='flatten', action='store_true', default=False,
						help='flatten the weights to power alpha if True')
	parser.add_argument('--clip', dest='clip', action='store_true', default=False,
						help='clip the weights to lower bound beta if True')

	parser.add_argument('--reference', dest='reference', action='store_true', default=False,
						help='evaluates reference sample quality metrics using real data if True')

	parser.add_argument('--bias-variance', dest='bias_variance', action='store_true', default=False,
						help='perform bias variance analysis')
	parser.add_argument('--calibration', dest='calibration', action='store_true', default=False,
						help='check calibration of classifier')
	parser.add_argument('--use-half', dest='use_half', action='store_true', default=False,
						help='uses 50 percent examples to perform bias variance analysis')


	args = parser.parse_args()

	torch.manual_seed(args.seed)
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	train_loader, valid_loader, test_loader = get_loaders(
		real_datadir=args.datadir,
		gen_datadir=args.sampledir,
		batch_size=args.batch_size,
		test_batch_size=args.test_batch_size,
		use_feature=args.use_feature,
		kwargs=kwargs
		)
	model = FeatureCritic(use_mlp=args.use_mlp).to(device)
	savefile = 'classifier_feature_mlp.pt'
	savepath = os.path.join(args.modeldir, savefile)

	if args.train:
		train_params = list(model.fc.parameters()) 
		if args.use_mlp:
			train_params += list(model.fc2.parameters())
		optimizer = optim.SGD(train_params, lr=args.lr, momentum=args.momentum)
		best_valid_acc = 0.
		for epoch in range(1, args.epochs + 1):
			train(args, model, device, train_loader, optimizer, epoch)
			_, valid_acc = test(args, model, device, valid_loader)
			if valid_acc > best_valid_acc:
				best_valid_acc = valid_acc
				torch.save(model.state_dict(), savepath)
	state_dict = torch.load(savepath)
	model.load_state_dict(state_dict)

	test_loss, test_acc = test(args, model, device, test_loader, validation=False)
	print('Test loss: {:.4f}, accuracy: {:.2f}'.format(test_loss, test_acc))

	if args.calibration:
		test_calibration(args, model, device, test_loader, n_bins=10, savefile=None)

	if args.bias_variance:
		compute_bias_variance(args, model, device, kwargs)

	if args.reference:
		compute_train_test_scores(args, model, device, kwargs)
	compute_iw_scores(args, model, device, kwargs)
		
if __name__ == '__main__':
	main()