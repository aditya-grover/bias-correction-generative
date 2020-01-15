import gzip
import math
import os
import pathlib
import pickle
import torch
import tensorflow as tf
import urllib

from scipy import linalg
from scipy.special import logsumexp
from scipy.stats import sem
from dataset import *
from utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

eps = 1e-20
INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score.pb'
# INCEPTION_INPUT = 'Mul:0'
INCEPTION_INPUT = 'ExpandDims:0'

INCEPTION_OUTPUT = 'logits:0'
INCEPTION_FINAL_POOL = 'pool_3:0'


def distance(X, Y, sqrt):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX,-1)
    X2 = (X*X).sum(1).resize_(nX,1)
    Y = Y.view(nY,-1)
    Y2 = (Y*Y).sum(1).resize_(nY,1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.io.gfile.GFile(pth, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='Inception_Net')


def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / INCEPTION_FROZEN_GRAPH
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract(INCEPTION_FROZEN_GRAPH, str(model_file.parent))
    return str(model_file)


def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'Inception_Net/' + INCEPTION_FINAL_POOL
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)

    layername = 'Inception_Net/' + INCEPTION_OUTPUT
    logits = sess.graph.get_tensor_by_name(layername)
    ops = logits.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    
    return pool3, logits


def calculate_frechet_distance(real_features, 
                        gen_features, 
                        eps=1e-6,
                        mu_weights=None, 
                        sigma_weights=None):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)

    sum_mu_wts = 1.
    if mu_weights is not None:
        sum_mu_wts = np.sum(mu_weights)
    sum_sigma_wts = 1.
    if sigma_weights is not None:
        sum_sigma_wts = np.sum(sigma_weights)

    mu2 = sum_mu_wts*np.average(gen_features, axis=0, weights=mu_weights)
    sigma2 = sum_sigma_wts*np.cov(gen_features, rowvar=False, aweights=sigma_weights)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        #warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    mean_diff = diff.dot(diff)
    cov_diff = np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return mean_diff+cov_diff, mean_diff, cov_diff


def get_kernel_matrices(real_features, gen_features):
    '''
    quick hack: convert to torch tensors and call torch function 'distance'
    '''
    feature_r = torch.from_numpy(real_features)
    feature_f = torch.from_numpy(gen_features)

    Mxx = distance(feature_r, feature_r, False).detach().cpu().numpy()
    Mxy = distance(feature_r, feature_f, False).detach().cpu().numpy()
    Myy = distance(feature_f, feature_f, False).detach().cpu().numpy()

    scale = np.mean(Mxx)
    sigma = 1
    Mxx = np.exp(-Mxx / (scale * 2 * sigma * sigma))
    Mxy = np.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = np.exp(-Myy / (scale * 2 * sigma * sigma))

    return Mxx, Mxy, Myy


def calculate_mmd(Mxx, Mxy, Myy, weights=None):

    imp_weights_matrix_yy = None
    imp_weights_matrix_xy = None
    if weights is not None:
        imp_weights_matrix_yy = np.outer(weights, weights)
        imp_weights_matrix_xy = np.outer(np.ones(Mxy.shape[0]), weights)

    sum_wts = 1.
    sum_sq_wts = 1.
    if weights is not None:
        sum_sq_wts = np.sum(imp_weights_matrix_yy)
        sum_wts = np.sum(weights)

    mmd = math.sqrt(np.mean(Mxx) \
        + sum_sq_wts * np.average(Myy, weights=imp_weights_matrix_yy) \
        - 2 * sum_wts * np.average(Mxy, weights=imp_weights_matrix_xy))

    return mmd


def calculate_inception_score(X, weights=None):

    sum_wts = 1.
    if weights is not None:
        sum_wts = np.sum(weights)

    kl = X * (np.log(X+eps)-np.log(sum_wts*np.average(X, axis=0, weights=weights)[None, :]+eps))
    score = np.exp(sum_wts*np.average(np.sum(kl, axis=1), weights=weights))

    return score


def get_activations(images, 
                    sess, 
                    batch_size=50, 
                    verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer, logits_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0//batch_size
    n_used_imgs = n_batches*batch_size
    pred_arr = np.empty((n_used_imgs,2048))
    logits_arr = np.empty((n_used_imgs,1008))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        end = start + batch_size
        batch = images[start:end]
        pred, logits = sess.run([inception_layer, logits_layer], {'Inception_Net/'+INCEPTION_INPUT: batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
        logits_arr[start:end] = logits.reshape(batch_size,-1)

    if verbose:
        print(" done")
    return pred_arr, logits_arr


def get_activation_stats(images, 
                        sess,
                        batch_size):

    features, logits = get_activations(images, 
                            sess, 
                            batch_size,
                            verbose=True)

    print(logits.shape, features.shape)
    
    return logits, features


def compute_scores_tf(real_images, 
                    gen_images, 
                    batch_size,
                    inception_path=None, 
                    imp_weights_logits=None,
                    baseline=False,
                    use_inception=True,
                    self_norm=False,
                    flatten=False,
                    clip=False):
    
    if use_inception:
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        inception_path = check_or_download_inception(inception_path)
        create_inception_graph(str(inception_path))
        print('inception graph created', flush=True)
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            real_logits, real_features = get_activation_stats(real_images,
                                                                sess,
                                                                batch_size=batch_size)
            gen_logits, gen_features = get_activation_stats(gen_images,
                                                                sess, 
                                                                batch_size=batch_size)
    else:
        print('here')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LeNet().to(device)
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), "logs/mnist/classifier.pt"))) 

        num_batches = len(real_images)//batch_size
        real_logits, real_features, gen_logits, gen_features = [], [], [], []
        
        for batch_idx, data in enumerate(real_images):
            data = data.to(device)
            batch_real_logits, batch_real_features = model(data)
            real_logits.append(batch_real_logits)
            real_features.append(batch_real_features)

        for batch_idx, data in enumerate(gen_images):
            data = data.to(device)
            batch_gen_logits, batch_gen_features = model(data)
            gen_logits.append(batch_gen_logits)
            gen_features.append(batch_gen_features)

        real_logits = np.concatenate(real_logits, axis=0)
        real_features = np.concatenate(real_features, axis=0)
        gen_logits = np.concatenate(gen_logits, axis=0)
        gen_features = np.concatenate(gen_features, axis=0)

    probs = softmax(gen_logits, axis=1)

    scores = {}

    imp_weights = None
    if imp_weights_logits is not None:
        imp_weights = np.exp(imp_weights_logits)/len(imp_weights_logits)

    scores['inception_score'] = calculate_inception_score(probs, weights=imp_weights)
    scores['fid'], scores['fid_mean'], scores['fid_cov'] = calculate_frechet_distance(real_features, gen_features, 
                                                mu_weights=imp_weights,
                                                sigma_weights=imp_weights)
    Mxx, Mxy, Myy = get_kernel_matrices(real_features, gen_features)
    scores['mmd'] = calculate_mmd(Mxx, Mxy, Myy, weights=imp_weights)

    if imp_weights_logits is not None:
        if self_norm:
            imp_weights_self_norm = np.exp(imp_weights_logits - logsumexp(imp_weights_logits))
            print('self norm sum weights', np.sum(imp_weights_self_norm))
            scores['inception_score_self_norm'] = calculate_inception_score(probs, weights=imp_weights_self_norm)
            scores['fid_self_norm'], scores['fid_mean_self_norm'], scores['fid_cov_self_norm'] = calculate_frechet_distance(real_features, gen_features, 
                                                        mu_weights=imp_weights_self_norm,
                                                        sigma_weights=imp_weights_self_norm)
            scores['mmd_self_norm'] = calculate_mmd(Mxx, Mxy, Myy, weights=imp_weights_self_norm)
        if flatten:
            alphas = [0.25, 0.5, 0.75]
            for alpha in alphas:
                imp_weights_alpha = np.power(np.exp(imp_weights_logits), alpha)/len(imp_weights_logits)
                scores['inception_score_alpha_'+str(alpha)] = calculate_inception_score(probs, weights=imp_weights_alpha)
                scores['fid_alpha_'+str(alpha)], scores['fid_mean_alpha_'+str(alpha)], scores['fid_cov_alpha_'+str(alpha)] = calculate_frechet_distance(real_features, gen_features, 
                                                            mu_weights=imp_weights_alpha,
                                                            sigma_weights=imp_weights_alpha)
                scores['mmd_alpha_'+str(alpha)] = calculate_mmd(Mxx, Mxy, Myy, weights=imp_weights_alpha)
        if clip:
            betas = [0.001, 0.01, 0.1, 1.]
            for beta in betas:
                imp_weights_beta = np.maximum(np.exp(imp_weights_logits), beta)/len(imp_weights_logits)
                scores['inception_score_beta_'+str(beta)] = calculate_inception_score(probs, weights=imp_weights_beta)
                scores['fid_beta_'+str(beta)], scores['fid_mean_beta_'+str(beta)], scores['fid_cov_beta_'+str(beta)] = calculate_frechet_distance(real_features, gen_features, 
                                                            mu_weights=imp_weights_beta,
                                                            sigma_weights=imp_weights_beta)
                scores['mmd_beta_'+str(beta)] = calculate_mmd(Mxx, Mxy, Myy, weights=imp_weights_beta)

    baseline_scores = None
    if baseline:
        baseline_scores = {}
        baseline_scores['inception_score'] = calculate_inception_score(probs)
        baseline_scores['fid'], baseline_scores['fid_mean'], baseline_scores['fid_cov'] = calculate_frechet_distance(real_features, gen_features)
        baseline_scores['mmd'] = calculate_mmd(Mxx, Mxy, Myy, weights=None)

    return scores, baseline_scores


def compute_train_test_scores(args, model, device, kwargs):

    model.eval()
    real_datadir = args.datadir
    xtr_real, xva_real, xte_real = get_real_data(real_datadir)

    x1 = np.transpose(xte_real, (0, 2, 3, 1))
    x2 = np.transpose(xtr_real[:10000], (0, 2, 3, 1))
    use_inception = True

    scores, _ = compute_scores_tf(x1, x2,
                batch_size=args.test_batch_size,
                imp_weights_logits=None,
                baseline=False,
                use_inception=use_inception)
    
    print('real2real inception score', scores['inception_score'])
    print('real2real fid', scores['fid'])
    print('real2real fid mean', scores['fid_mean'])
    print('real2real fid cov', scores['fid_cov'])
    print('real2real mmd', scores['mmd'])

    all_is, all_fid, all_mmd, all_fid_mean, all_fid_cov = [], [], [], [], []

    for _ in range(10):
        print('Run:', _, flush=True)
        x2_idx = np.random.choice(len(x2), size=len(x2), replace=True)
        x2_bootstrap = x2[x2_idx] 
        scores, _ = compute_scores_tf(x1, x2_bootstrap,
            batch_size=args.test_batch_size,
            imp_weights_logits=None,
            baseline=False,
            use_inception=use_inception)

        all_is.append(scores['inception_score'])
        all_fid.append(scores['fid'])
        all_mmd.append(scores['mmd'])

        all_fid_mean.append(scores['fid_mean'])
        all_fid_cov.append(scores['fid_cov'])

    all_is = np.array(all_is)
    all_fid = np.array(all_fid)
    all_mmd = np.array(all_mmd)

    print('IS mean/std:', np.mean(all_is), sem(all_is))
    print('FID mean/std:', np.mean(all_fid), sem(all_fid))
    print('mmd mean/std:', np.mean(all_mmd), sem(all_mmd))

    print('FID_mean mean/std:', np.mean(all_fid_mean), sem(all_fid_mean))
    print('FID_cov mean/std:', np.mean(all_fid_cov), sem(all_fid_cov))

    return


def compute_iw_scores(args, model, device, kwargs):
    model.eval()
    real_datadir = args.datadir
    gen_datadir = args.sampledir

    _, _, real_test_loader, _, _, _ = \
        get_eval_loaders(
            real_datadir=real_datadir,
            gen_datadir=gen_datadir,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            use_feature=args.use_feature,
            kwargs=kwargs)

    xte_real = []
    for _, real_data in enumerate(real_test_loader):
        print(_, end=' ', flush=True)
        xte_real.append(real_data.cpu().numpy())

    # inception net requires input in [0, 255]
    xte_real = np.transpose(255*(np.concatenate(xte_real, axis=0)+1)*0.5, (0, 2, 3, 1))
    use_inception = True

    
    all_is, all_fid, all_mmd, all_fid_mean, all_fid_cov = [], [], [], [], []
    all_bs_is, all_bs_fid, all_bs_mmd, all_bs_fid_mean, all_bs_fid_cov = [], [], [], [], []
    
    all_scores = []
    for idx in range(10):
        print('\nRun:', idx)
        _, _, _, _, _, gen_test_loader = \
            get_eval_loaders(
                real_datadir=real_datadir,
                gen_datadir=gen_datadir,
                batch_size=args.batch_size,
                test_batch_size=args.test_batch_size,
                use_feature=args.use_feature,
                kwargs=kwargs,
                test_idx=idx)
        all_logits = []
        xte_gen = []
        with torch.no_grad():
            for _, gen_data in enumerate(gen_test_loader):
                print(_, end=' ', flush=True)
                xte_gen.append(gen_data.numpy())
                gen_data = gen_data.to(device)
                logits = model(gen_data)
                all_logits.append(logits.detach().cpu().numpy())
        all_logits = np.concatenate(all_logits, axis=0)
        imp_weights_logits = all_logits
        # normalized_density_ratios = np.exp(all_logits - logsumexp(all_logits)).squeeze()
        # print(normalized_density_ratios.shape, normalized_density_ratios.sum())
        
        # inception net requires input in [0, 255]
        xte_gen = np.transpose(255*(np.concatenate(xte_gen, axis=0)+1)*0.5, (0, 2, 3, 1))

        scores, baseline_scores = compute_scores_tf(xte_real,
            xte_gen,
            batch_size=args.test_batch_size,
            imp_weights_logits=imp_weights_logits.squeeze(),
            baseline=True, 
            use_inception=use_inception,
            self_norm=args.self_norm,
            flatten=args.flatten,
            clip=args.clip)

        all_scores.append(scores) 
        print(scores)

        all_bs_is.append(baseline_scores['inception_score'])
        all_bs_fid.append(baseline_scores['fid'])
        all_bs_mmd.append(baseline_scores['mmd'])

        all_bs_fid_mean.append(baseline_scores['fid_mean'])
        all_bs_fid_cov.append(baseline_scores['fid_cov'])
        print()

    all_metrics = ['inception_score', 'fid', 'mmd', 'fid_mean', 'fid_cov']
    all_methods = [''] 
    if args.self_norm:
        all_methods.append('_self_norm')
    if args.flatten:
        all_methods.extend(['_alpha_'+str(alpha) for alpha in [0.25,0.5, 0.75]])
    if args.clip:
        all_methods.extend(['_beta_'+str(beta) for beta in [0.001, 0.01, 0.1, 1.]])
    for metric in all_metrics:
        for method in all_methods:
            key = metric+method
            all_runs = [scores[key] for scores in all_scores]
            print(key + ' mean/std:', np.mean(all_runs), sem(all_runs))
        print()

    all_bs_is = np.array(all_bs_is)
    all_bs_fid = np.array(all_bs_fid)
    all_bs_mmd = np.array(all_bs_mmd)

    print('IS BS mean/std:', np.mean(all_bs_is), sem(all_bs_is))
    print('FID BS mean/std:', np.mean(all_bs_fid), sem(all_bs_fid))
    print('mmd BS mean/std:', np.mean(all_bs_mmd), sem(all_bs_mmd))

    print('FID_mean BS mean/std:', np.mean(all_bs_fid_mean), sem(all_bs_fid_mean))
    print('FID_cov BS mean/std:', np.mean(all_bs_fid_cov), sem(all_bs_fid_cov))

    return scores


def calculate_bv_components(
    real_features, 
    gen_features, 
    eps=1e-6,
    mu_weights=None):

    mu1 = np.mean(real_features, axis=0)

    sum_mu_wts = 1.
    if mu_weights is not None:
        sum_mu_wts = np.sum(mu_weights)

    mu2 = sum_mu_wts*np.average(gen_features, axis=0, weights=mu_weights)

    return mu1, mu2


def compute_bv(
    real_images, 
    gen_images, 
    batch_size,
    inception_path=None, 
    imp_weights_logits=None,
    baseline=False,
    self_norm=False,
    flatten=False,
    clip=False):
    
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    inception_path = check_or_download_inception(inception_path)
    create_inception_graph(str(inception_path))
    print('inception graph created', flush=True)
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        real_logits, real_features = get_activation_stats(real_images,
                                                            sess,
                                                            batch_size=batch_size)
        gen_logits, gen_features = get_activation_stats(gen_images,
                                                            sess, 
                                                            batch_size=batch_size)

    scores = {}

    if imp_weights_logits is not None:
        if self_norm:
            imp_weights_self_norm = np.exp(imp_weights_logits - logsumexp(imp_weights_logits))
            print('self norm sum weights', np.sum(imp_weights_self_norm))
            scores['reference'], scores['self_norm'] = calculate_bv_components(real_features, gen_features, 
                                                        mu_weights=imp_weights_self_norm)
        if flatten:
            alphas = [0, 0.25, 0.5, 0.75, 1.]
            for alpha in alphas:
                if alpha == 0:
                    imp_weights_alpha = 1./len(imp_weights_logits) * np.ones_like(imp_weights_logits)
                else:
                    imp_weights_alpha = np.power(np.exp(imp_weights_logits), alpha)/len(imp_weights_logits)
                _, scores['alpha_'+str(alpha)] = calculate_bv_components(real_features, gen_features, 
                                                            mu_weights=imp_weights_alpha)
        if clip:
            betas = [0.001, 0.01, 0.1, 1.]
            for beta in betas:
                imp_weights_beta = np.maximum(np.exp(imp_weights_logits), beta)/len(imp_weights_logits)
                _, scores['beta_'+str(beta)] = calculate_bv_components(real_features, gen_features, 
                                                            mu_weights=imp_weights_beta)

    return scores


def compute_bias_variance(args, model, device, kwargs):

    model.eval()
    real_datadir = args.datadir
    gen_datadir = args.sampledir

    _, _, real_test_loader, _, _, _ = \
        get_eval_loaders(
            real_datadir=real_datadir,
            gen_datadir=gen_datadir,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            use_feature=args.use_feature,
            kwargs=kwargs)

    xte_real = []
    for _, real_data in enumerate(real_test_loader):
        print(_, end=' ', flush=True)
        xte_real.append(real_data.cpu().numpy())

    # inception net requires input in [0, 255]
    xte_real = np.transpose(255*(np.concatenate(xte_real, axis=0)+1)*0.5, (0, 2, 3, 1))

    all_is, all_fid, all_mmd, all_fid_mean, all_fid_cov = [], [], [], [], []
    all_bs_is, all_bs_fid, all_bs_mmd, all_bs_fid_mean, all_bs_fid_cov = [], [], [], [], []
    
    all_scores = []
    for idx in range(2):
        print('\nRun:', idx)
        _, _, _, _, _, gen_test_loader = \
            get_eval_loaders(
                real_datadir=real_datadir,
                gen_datadir=gen_datadir,
                batch_size=args.batch_size,
                test_batch_size=args.test_batch_size,
                use_feature=args.use_feature,
                kwargs=kwargs,
                test_idx=idx)
        all_logits = []
        xte_gen = []
        with torch.no_grad():
            for _, gen_data in enumerate(gen_test_loader):
                print(_, end=' ', flush=True)
                xte_gen.append(gen_data.numpy())
                gen_data = gen_data.to(device)
                logits = model(gen_data)
                all_logits.append(logits.detach().cpu().numpy())
        all_logits = np.concatenate(all_logits, axis=0)
        imp_weights_logits = all_logits

        # inception net requires input in [0, 255]
        xte_gen = np.transpose(255*(np.concatenate(xte_gen, axis=0)+1)*0.5, (0, 2, 3, 1))

        if args.use_half:
            full_len = len(xte_gen)
            xte_gen = xte_gen[:int(full_len/2)]
            imp_weights_logits = imp_weights_logits[:int(full_len/2)]
            print(xte_gen.shape, imp_weights_logits.shape)

        scores = compute_bv(xte_real,
            xte_gen,
            batch_size=args.test_batch_size,
            imp_weights_logits=imp_weights_logits.squeeze(),
            baseline=True, 
            self_norm=args.self_norm,
            flatten=args.flatten,
            clip=args.clip)

        all_scores.append(scores) 
        print()

    all_methods = [] 
    if args.self_norm:
        all_methods.append('self_norm')
    if args.flatten:
        all_methods.extend(['alpha_'+str(alpha) for alpha in [0, 0.25,0.5, 0.75, 1.]])
    if args.clip:
        all_methods.extend(['beta_'+str(beta) for beta in [0.001, 0.01, 0.1, 1.]])
    ref_score = all_scores[0]['reference']
    for method in all_methods:
        all_runs = [scores[method] for scores in all_scores]
        bias_estimate = np.mean(np.array([ref_score-run for run in all_runs]), axis=0)
        variance_estimate = np.var(np.array([run for run in all_runs]), axis=0)
        mse_estimate = np.mean(np.square(np.array([ref_score-run for run in all_runs])), axis=0)
        print(bias_estimate.shape, variance_estimate.shape, mse_estimate.shape)
        print(method+ ' absolute bias mean/std:', np.abs(np.mean(bias_estimate)), sem(bias_estimate), \
            'square bias mean/std:', np.mean(np.square(bias_estimate)), sem(np.square(bias_estimate)), \
            'variance mean/std:', np.mean(variance_estimate), sem(variance_estimate), \
            'mse mean/std:', np.mean(mse_estimate), sem(mse_estimate))
    print()

    return scores