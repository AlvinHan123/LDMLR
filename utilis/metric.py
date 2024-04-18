# Calculate FID
import torch
import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import cosine_similarity
# Function to extract feature vectors
def unload_feature_vectors(data_loader):
    features = []
    feature_labels = []
    for batch in data_loader:
        feature, feature_label = batch
        features.append(feature)
        feature_labels.append(feature_label)
    return torch.cat(features, dim=0), torch.cat(feature_labels, dim=0)

def calculate_fid(real_features, generated_features):
    # Move both tensors to the same device (e.g., CPU)
    real_features = real_features.to('cpu')
    generated_features = generated_features.to('cpu')

    mu_real = torch.mean(real_features, dim=0)
    mu_generated = torch.mean(generated_features, dim=0)

    # Calculate the covariance matrices manually
    cov_real = torch.mm((real_features - mu_real).t(), (real_features - mu_real)) / (real_features.size(0) - 1)
    cov_generated = torch.mm((generated_features - mu_generated).t(), (generated_features - mu_generated)) / (generated_features.size(0) - 1)

    mu_diff = torch.norm(mu_real - mu_generated)

    # Compute the square root of the covariance product using SVD
    U_real, S_real, V_real = torch.svd(cov_real)
    U_generated, S_generated, V_generated = torch.svd(cov_generated)
    square_root_real = torch.mm(U_real, torch.diag((torch.sqrt(S_real))))
    covar_product = torch.mm(U_real, torch.mm(torch.diag(torch.sqrt(S_real)), V_real.t()))

    # Ensure that covar_product is a real-valued matrix
    if torch.is_complex(covar_product):
        covar_product = covar_product.real

    trace = torch.trace(cov_real + cov_generated - 2 * covar_product)

    fid = mu_diff**2 + trace
    return fid.item()

# input the feature vector, feature label, genrated feature vector, generated feature label, and the number of the class.
def cal_FID_per_class(feature_vec, feature_vec_label, gen_feature, gen_feature_label, class_img_counts):
    fid_values = {}
    device = 'cpu'
    for class_label in range(class_img_counts):
        # Extract feature vectors for both real and generated data
        real_features = feature_vec[feature_vec_label == class_label]
        generated_features = gen_feature[gen_feature_label == class_label]

        real_features = real_features.to(device).detach().numpy()
        generated_features = generated_features.to(device).detach().numpy()
        # Calculate FID for this class
        # fid_value = calculate_fid(real_features, generated_features)
        real_mu, real_cov = calculate_mean_and_cov(real_features)
        generated_mu, generated_cov = calculate_mean_and_cov(generated_features)
        fid_value = calculate_frechet_distance(real_mu, real_cov, generated_mu, generated_cov)
        fid_values[class_label] = fid_value
    return fid_values


def calculate_mean_and_cov(vector):
    mu = np.mean(vector, axis=0)
    cov = (vector - mu).T @ (vector - mu) / (vector.shape[0] - 1)
    return mu, cov

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def cal_cosine_similarity(feature_vec, feature_vec_label, gen_feature, gen_feature_label, class_img_counts):
    cosine = {}
    device = 'cpu'
    for class_label in range(class_img_counts):
        # Extract feature vectors for both real and generated data
        real_features = feature_vec[feature_vec_label == class_label]
        generated_features = gen_feature[gen_feature_label == class_label]

        real_features = real_features.to(device).detach().numpy()
        generated_features = generated_features.to(device).detach().numpy()
        # Calculate the mean feature vectors
        real_mean = np.mean(real_features, axis=0)
        gen_mean = np.mean(generated_features, axis=0)
        real_mean = real_mean.reshape(1, -1)
        gen_mean = gen_mean.reshape(1, -1)
        # Calculate the cosine similarity between the means
        similarity = cosine_similarity(real_mean, gen_mean)

        cosine[class_label] = similarity

    return cosine