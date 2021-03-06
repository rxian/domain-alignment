import torch
import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None

class GradientReversalLayer(torch.nn.Module):
    def __init__(self, lambd=1):
        super(GradientReversalLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradReverse.apply(x, self.lambd)

    def extra_repr(self):
        return 'lambda={}'.format(self.lambd)


def calc_gradient_penalty(netD, real_data, fake_data, center=0):
    """Computes ell_2 gradient penalty at interpolates between real and fake data.
    
    Adapted from https://github.com/caogang/wgan-gp

    Args:
        netD: The discriminator network whose gradient will be regularized.
        real_data: Real input data.
        fake_data: Fake input data.
        center: Desired norm of discriminator gradients.

    Returns:
        Gradient penalty loss.
    """

    dev = real_data.device

    length = min(len(real_data),len(fake_data))
    real_data = real_data[:length]
    fake_data = fake_data[:length]

    alpha = torch.rand(length, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(dev)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(dev)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size()).to(dev),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean()
    return gradient_penalty


def im_weights_update(source_cov, source_y, target_y):
    """
    Solve a Quadratic Program to compute the optimal importance weight 
    under the generalized label shift assumption. 
    
    Adapted from https://github.com/microsoft/Domain-Adaptation-with-Conditional-Distribution-Matching-and-Generalized-Label-Shift

    Args:
        source_cov: The covariance matrix of predicted-label and true 
                    label of the source domain.
        source_y: The marginal label distribution of the source domain.
        target_y: The marginal pseudo-label distribution of the target 
                  domain from the current classifier.

    Returns:
        Estimated importance weights.
    """

    dim = source_cov.shape[0]
    source_y = source_y.reshape(-1, 1).astype(np.double)
    target_y = target_y.reshape(-1, 1).astype(np.double)
    source_cov = source_cov.astype(np.double)

    P = matrix(np.dot(source_cov.T, source_cov), tc="d")
    q = -matrix(np.dot(source_cov, target_y), tc="d")
    G = matrix(-np.eye(dim), tc="d")
    h = matrix(np.zeros(dim), tc="d")
    A = matrix(source_y.reshape(1, -1), tc="d")
    b = matrix([1.0], tc="d")
    sol = solvers.qp(P, q, G, h, A, b)

    im_weights = np.array(sol["x"])
    return im_weights


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Computes Gaussian (for MK-MMD) kernel between source and target.

    Adapted from https://github.com/microsoft/Domain-Adaptation-with-Conditional-Distribution-Matching-and-Generalized-Label-Shift/blob/main/loss.py

    Args:
        source: Source data.
        target: Target data.
        kernel_mul: Multiplicative factor of kernel bandwidth.
        kernel_num: Number of kernels.
        fix_sigma: Fix kernel bandwidth (otherwise use distance between data).
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


class AdversarialNetwork(torch.nn.Module):
    """A single-hidden-layer ReLU discriminator."""
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_feature,hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size,2),
        )

    def forward(self, x):
        output = self.net(x)
        score = (output[:,0] - output[:,1]).view(-1,1)
        return score


class W1CriticWithImWeights(torch.nn.Module):
    """An adversary with class-importance-weighted Wasserstein-1 loss."""

    def __init__(self, in_feature, hidden_size, im_weights=None):
        """Inits the W1 critic.

        Args:
            in_feature: The input data/feature dimension (scalar).
            hidden_size: The width of the hidden layer.
            im_weights: Importance weights for weighting the critic loss.        
        """

        super(W1CriticWithImWeights, self).__init__()
        self.net = AdversarialNetwork(in_feature, hidden_size)
        if im_weights is not None:
            self.register_buffer('im_weights', im_weights.view(-1), persistent=True)
        else:
            self.im_weights = None

    def get_im_weights(self):
        """Gets importance weights."""
        return self.im_weights

    def forward(self, x, domain_labels, y_true=None, alpha=1):
        """Computes unregularized W1 critic loss; need to add gradient penalty.
        
        Args:
            x: The input data/features.
            y_true: The true class labels.
            is_target_dom: Whether the data is from the target domain.
            alpha: Interpolates between using im_weights (alpha=1) and uniform weights (alpha=0).

        Returns:
            The unregularized W1 critic loss.
        """

        score = self.net(x)
        score = score.view(-1)
        im_weights = self.get_im_weights()
        if im_weights is not None:
            # Apply importance weights to the critic's scores on source examples.
            im_weights = (alpha*im_weights + (1-alpha)).detach()
            score[domain_labels==0] *= im_weights[y_true]
        loss = score[domain_labels==0].mean() - score[domain_labels==1].mean()
        return loss


class JSDAdversaryWithImWeights(torch.nn.Module):
    """An adversary with class-importance-weighted cross-entropy loss."""

    def __init__(self, in_feature, hidden_size, im_weights):
        """Inits the JSD adversary.

        Args:
            in_feature: The input data/feature dimension (scalar).
            hidden_size: The width of the hidden layer.
            im_weights: Importance weights for weighting the critic loss.        
        """

        super(JSDAdversaryWithImWeights, self).__init__()
        self.net = AdversarialNetwork(in_feature, hidden_size)
        self.register_buffer('im_weights', im_weights, persistent=True)

    def get_im_weights(self):
        """Gets importance weights."""
        return self.im_weights

    def forward(self, x, domain_labels, y_true=None, alpha=1):
        """Computes cross-entropy adversarial loss.
        
        Args:
            x: The input data/features.
            y_true: The true class labels.
            is_target_dom: Whether the data is from the target domain.
            alpha: Interpolates between using im_weights (alpha=1) and uniform weights (alpha=0).

        Returns:
            The cross-entropy adversarial loss.
        """

        outputs = self.net(x)
        score = torch.sigmoid(outputs)
        score = torch.nn.BCELoss(reduction='none')(score, domain_labels)
        score = score.view(-1)
        im_weights = self.get_im_weights()
        if im_weights is not None:
            # Apply importance weights to the critic's scores on source examples.
            im_weights = (alpha*im_weights + (1-alpha)).detach()
            score[domain_labels==0] *= im_weights[y_true]
        loss = score.mean()
        return loss


class MMDWithImWeights(torch.nn.Module):
    """MMD estimator with class-importance-weighting."""

    def __init__(self, im_weights=None, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """Inits MMD estimator.

        Args:
            im_weights: Importance weights for weighting the critic loss.
            See `gaussian_kernel` for the rest.
        """

        super(MMDWithImWeights, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma
        self.register_buffer('im_weights', im_weights, persistent=True)

    def get_im_weights(self):
        """Gets importance weights."""
        return self.im_weights

    def forward(self, x, domain_labels, y_true=None, alpha=1):
        """Computes unregularized W1 critic loss; need to add gradient penalty.
        
        Args:
            x: The input data/features.
            y_true: The true class labels.
            domain_labels: The domain label.
            alpha: Interpolates between using im_weights (alpha=1) and uniform weights (alpha=0).

        Returns:
            The unregularized W1 critic loss.
        """

        x_s, x_t = x[domain_labels==0], x[domain_labels==1]
        n_s, n_t = len(x_s), len(x_t)

        kernels = gaussian_kernel(x_s, x_t, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)

        im_weights = self.get_im_weights()
        if im_weights is not None:
            im_weights = (alpha*im_weights + (1-alpha)).detach()
        else:
            im_weights = torch.ones(y_true.max(),dype=kernels.dtype,device=kernels.device)

        # Non-cross terms
        loss_1 = 0

        idx = torch.triu_indices(n_s, n_s, 1)
        w = im_weights[y_true[idx[0]]]*im_weights[y_true[idx[1]]]
        loss_1 += (kernels[idx[0],idx[1]]*w).sum() / float((n_s*(n_s-1))/2)
        
        idx = torch.triu_indices(n_t, n_t, 1)
        loss_1 += (kernels[idx[0]+n_s,idx[1]+n_s]).sum() / float((n_t*(n_t-1))/2)

        # Cross terms
        loss_2 = 0

        idx_s, idx_t = torch.meshgrid(torch.arange(n_s), torch.arange(n_t), indexing='ij')
        idx_s, idx_t = idx_s.reshape(-1), idx_t.reshape(-1)
        w = im_weights[y_true[idx_s]]
        loss_2 += (kernels[idx_s,idx_t+n_s]*w).sum() / float(n_s*n_t)

        return loss_1 - loss_2*2


class ImWeightsEstimator(torch.nn.Module):
    """A class importance weight estimator.

    Includes helpers to collect statistics needed for estimation, and 
    to compute the loss for optimizing the weights.
    """

    def __init__(self, num_classes, source_class_dist, im_weights_init=None, hard_confusion_mtx=True, confusion_mtx_agg_mode='exp'):
        """Inits the estimator.

        Args:
            num_classes: The number of classes.
            source_class_dist: The class distribution of the source domain.
            im_weights_init: Initial importance weights (if None, initialized to 1).
            hard_confusion_mtx: Whether use hard label statistics for estimating 
                                importance weights.
            confusion_mtx_agg_mode: Mode for aggregating statistics. Can be one of
                                    ['exp','mean'].
        
        - 'exp' mode: Initialize `source_confusion_mtx` with `source_class_dist` and
                      `target_pred_dist` with `source_class_dist * im_weights_init`,
                      so that the IWs are optimal at initialization. The statistics
                      are thereafter updated with exponential decay.
        - 'mean' mode: Statistics are averaged (initialized to 0).
        """

        super(ImWeightsEstimator, self).__init__()

        self.num_classes = num_classes
        self.register_buffer('source_class_dist', source_class_dist.view(-1), persistent=True)
        self.hard_confusion_mtx = hard_confusion_mtx
        self.confusion_mtx_agg_mode = confusion_mtx_agg_mode

        # im_weights = im_weights_init + im_weights_d. 
        # This decomposition allows us to easily regularize the deviations `im_weights_d`.
        if im_weights_init is None:
            im_weights_init = torch.ones(num_classes)
        self.register_buffer('im_weights_init', im_weights_init.view(-1), persistent=True)
        self.im_weights_d = torch.nn.Parameter(torch.zeros(num_classes))

        # Initialize statistics.
        if confusion_mtx_agg_mode == 'exp':
            self.register_buffer('target_pred_dist', self.source_class_dist.data, persistent=True)
            self.register_buffer('source_confusion_mtx', torch.diag(self.source_class_dist.data*self.im_weights_init.data), persistent=True)
        else: 
            self.register_buffer('target_pred_dist', torch.zeros_like(self.source_class_dist.data), persistent=True)
            self.register_buffer('source_confusion_mtx', torch.diag(torch.zeros_like(self.source_class_dist.data)), persistent=True)

    def get_target_pred_dist(self):
        """Gets the predicted target class distribution."""
        return self.target_pred_dist/self.target_pred_dist.sum()

    def get_source_confusion_mtx(self):
        """Gets the source confusion matrix."""
        return self.source_confusion_mtx/self.source_confusion_mtx.sum()

    def get_im_weights(self):
        """Gets the importance weights.

        Since gradient updates may send the weights outside the feasible region, 
        we project them back before returning them. This is done via normalization
        s.t. (source_class_dist * im_weights).sum() = 1.
        """
        iw_weights_old = torch.clamp(self.im_weights_d + self.im_weights_init, min=0).detach()
        source_dot_im_weights = (self.source_class_dist * iw_weights_old).sum()
        iw_weights_norm = iw_weights_old/source_dot_im_weights
        self.im_weights_d.data = iw_weights_norm - self.im_weights_init
        return self.im_weights_init + self.im_weights_d

    def get_im_weights_loss(self):
        """Computes ell_2 loss of the importance weights (based on current statistics)."""
        im_weights = self.get_im_weights()
        source_confusion_mtx = self.get_source_confusion_mtx()
        target_pred_dist = self.get_target_pred_dist()       
        loss = (((source_confusion_mtx @ im_weights.view(-1,1)) - target_pred_dist.view(-1,1))**2).sum()
        return loss

    def update_im_weights_qp(self):
        """Updates the importance weights to their optimal values (based on 
        current statistics) by solving a QP.

        This could be slower than gradient-based methods by back-propagating
        the ell_2 loss from `get_im_weights_loss` on GPU training.
        """
        source_confusion_mtx = self.get_source_confusion_mtx()
        target_pred_dist = self.get_target_pred_dist()       
        iw_weights_new = im_weights_update(source_confusion_mtx.cpu().numpy(), self.source_class_dist.cpu().numpy(), target_pred_dist.cpu().numpy())
        iw_weights_new = torch.tensor(iw_weights_new.reshape(-1),device=self.im_weights_d.device,dtype=self.im_weights_d.dtype)
        self.im_weights_d.data = iw_weights_new - self.im_weights_init
        return iw_weights_new

    def forward(self, y_true=None, y_proba=None, is_target_dom=None, s=5e-3):
        """Updates the source domain confusion matrix and target domain 
        predicted class distribution with the provided samples.

        Args:
            y_true: The true class labels.
            y_proba: The (predicted) class probabilities.
            is_target_dom: Whether the data is from the target domain.
            s: The learning rate for updating confusion matrix and prediction distribution. 
               Ignored if `confusion_mtx_agg_mode` is 'exp'.
        
        Returns:
            The ell_2 loss of the importance weights.
        """

        if is_target_dom:
            if self.hard_confusion_mtx:
                y_pred = torch.argmax(y_proba,dim=1).view(-1)
                l, c = torch.unique(y_pred,return_counts=True)
                this_target_pred_dist = torch.zeros_like(self.target_pred_dist)
                this_target_pred_dist[l] += c
            else:
                this_target_pred_dist = torch.sum(y_proba, dim=0)
            if self.confusion_mtx_agg_mode == 'exp':
                self.target_pred_dist = (1-s)*self.target_pred_dist + s*this_target_pred_dist
            else:
                self.target_pred_dist += this_target_pred_dist
        else:
            y_true = y_true.long().view(-1)
            if self.hard_confusion_mtx:
                y_pred = torch.argmax(y_proba,dim=1).view(-1)
                this_source_confusion_mtx = torch.zeros_like(self.source_confusion_mtx)
                l, c = torch.unique(torch.stack([y_pred,y_true]).t(),dim=0,return_counts=True)
                this_source_confusion_mtx[l[:,0], l[:,1]] += c
            else:
                y_true_onehot = torch.zeros((len(y_true),self.num_classes),device=self.source_confusion_mtx.device).scatter(1, y_true.view(-1,1), 1)
                this_source_confusion_mtx = (y_proba.transpose(1, 0) @ y_true_onehot)
            if self.confusion_mtx_agg_mode == 'exp':
                self.source_confusion_mtx = (1-s)*self.source_confusion_mtx + s*this_source_confusion_mtx
            else:
                self.source_confusion_mtx += this_source_confusion_mtx

        return self.get_im_weights_loss()
