import numpy
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from model.ddpm_conditional import UNet_conditional, ConditionalDiffusion1D
from model.CGAN import Generator
import torch
from code.sample import generate_conditional_feature
from copy import deepcopy
import random
from utilis.metric import unload_feature_vectors
from utilis.diffusion_model import Diffusion, sample
from torch.autograd import Variable
import os

# sample function of conditional_diffusion
@torch.no_grad()
def conditional_sample(model, classes):
    # The sampling loop
    pred_img = model.sample(classes)
    # If we are on the last timestep, output the denoised image
    return pred_img

# sample function of conditional_diffusion
def generate_feature(diffusion_model_ema, device):

    class_num = []
    for _ in range(10):
        class_num.append(50)

    # Returns a new list with each nonzero index (zero-based) repeated as many times as the value at that position in the original list.
    fake_classes = [i for i, count in enumerate(class_num) for _ in range(count)]
    fake_classes = torch.tensor(fake_classes).to(device)

    # sample 64 images
    samples = conditional_sample(diffusion_model_ema, fake_classes)

    return samples, fake_classes

# draw sample from training set
def sample_from_trainingset(features, labels):
    import numpy as np

    # Assuming you have your features and class labels in 'features' and 'labels' arrays

    # Define the number of samples to pick from each class
    num_samples_per_class = 50

    # Get the unique class labels
    unique_classes = np.unique(labels)

    # Initialize arrays to store the selected features and labels
    selected_features = []
    selected_labels = []

    # Randomly select samples from each class
    for class_label in unique_classes:
        # Get the indices of features with the current class label
        class_indices = np.where(labels == class_label)[0]

        # Randomly select 'num_samples_per_class' indices from the current class
        selected_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)

        # Append the selected features and labels
        selected_features.extend(features[selected_indices])
        selected_labels.extend(labels[selected_indices])

    # Convert the selected features and labels back to NumPy arrays
    selected_features = np.array(selected_features)
    selected_labels = np.array(selected_labels)

    return selected_features, selected_labels

# sample from GAN
def sample_from_GAN():
    n_classes = 10
    latentdim = 100
    image_size = 8
    FT_a = torch.FloatTensor
    generator = Generator(n_classes, latentdim, (image_size, image_size))
    generator = generator.load_state_dict(torch.load('./saves/models/best_GAN.pth'))

    gen_labels = torch.arange(10, device="cuda").repeat_interleave(1000, 0)
    noise = Variable(FT_a(np.random.normal(0, 1, (10000, latentdim))))
    fake = generator(noise, gen_labels)

    gen_labels = gen_labels.to('cpu').numpy()
    fake = fake.to('cpu').numpy()
    return gen_labels, fake

# sample function of Diffusion without Resblock
def generate_feature_ratio(diffusion_model_ema, eta, seed, device):
    torch.manual_seed(seed)
    class_num = []
    for _ in range(10):
        class_num.append(50)

    fake_classes = [i for i, count in enumerate(class_num) for _ in range(count)]
    fake_classes = torch.tensor(fake_classes).to(device) # add to device

    # check noise and class_num(fake_label1)
    noise = torch.randn([sum(class_num), 64], device=device)

    # sample 64 images
    samples = sample(diffusion_model_ema, noise, 1000, eta, fake_classes)

    return samples, fake_classes

# sample function of Diffusion without Resblock
def sample_from_diffusion_wo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eta = 0.
    seed = 123
    diffusion_model = Diffusion("CIFAR10")
    diffusion_model.load_state_dict(torch.load('./saves/models/diffusion_wo_Res.pth'))
    diffusion_model.eval()
    diffusion_model.to(device)
    diffusion_model_ema = deepcopy(diffusion_model)
    features, labels = generate_feature_ratio(diffusion_model_ema, eta, seed, device )
    features, labels = features.to('cpu').numpy(), labels.to('cpu').numpy()
    return features,labels

# preprocesing the training set and save the features and the labels.
def preprocessing(feature_loader):
    feature_vec, feature_vec_label = unload_feature_vectors(feature_loader)
    feature_vec, feature_vec_label = feature_vec.to('cpu').numpy(), feature_vec_label.to('cpu').numpy()
    tr_feature, tr_label = sample_from_trainingset(feature_vec, feature_vec_label)
    numpy.save('./saves/data/tr_feature.npy', tr_feature)
    numpy.save('./saves/data/tr_label.npy', tr_label)
    return tr_feature, tr_label

# sample from conditional diffusion
def sample_from_diffusion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Unet Initialization
    model = UNet_conditional(
        dim=64,
        dim_mults=(1, 2, 4),
        channels=1,
        num_classes=10
    )

    # diffusion Initialization
    diffusion_model = ConditionalDiffusion1D(
        model,
        seq_length=64,
        timesteps=1000,
        objective='pred_x0'
    )
    diffusion_model.load_state_dict(torch.load('./saves/models/con_diffusion.pth'))
    diffusion_model.eval()
    diffusion_model.to(device)
    diffusion_model_ema = deepcopy(diffusion_model)

    features, classes = generate_feature(diffusion_model_ema, device)

    features = features.squeeze(1)

    features = features.to('cpu').numpy()
    classes = classes.to('cpu').numpy()
    return features, classes

def visualise():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # got 50 feature samples from each class
    output_directory = './imgs/features'
    tr_feature, tr_label = np.load('./saves/data/tr_feature.npy'), np.load('./saves/data/tr_label.npy')
    df_without_resblock_feature, df_without_resblock_label = sample_from_diffusion_wo()
    df_feature, df_label = sample_from_diffusion()

    # Apply PCA to reduce the feature vectors to 3 dimensions
    pca = PCA(n_components=3)
    tr_feature_pca = pca.fit_transform(tr_feature)
    df_feature_pca = pca.transform(df_feature)
    df_wo_feature_pca = pca.transform(df_without_resblock_feature)

    # Combine the feature vectors and labels
    all_features = np.vstack((tr_feature_pca, df_feature_pca, df_wo_feature_pca))
    all_labels = np.hstack((tr_label, df_label, df_without_resblock_label))

    # Create a dictionary to indicate whether a point is from tr, df, or gan
    point_source = {}
    point_source.update({idx: 'tr' for idx in range(len(tr_feature_pca))})
    point_source.update({idx: 'df' for idx in range(len(tr_feature_pca), len(tr_feature_pca) + len(df_feature_pca))})
    point_source.update({idx: 'wo' for idx in range(len(tr_feature_pca) + len(df_feature_pca), len(all_features))})

    # Get unique class labels
    unique_classes = np.unique(all_labels)

    # Create a 3D scatter plot for each class
    for class_label in unique_classes:
        class_indices = (all_labels == class_label)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for idx in np.where(class_indices)[0]:
            if point_source[idx] == 'tr':
                color = 'b'
            elif point_source[idx] == 'df':
                color = 'r'
            else:
                color = 'g'  # Use green for GAN points
            ax.scatter(all_features[idx, 0], all_features[idx, 1], all_features[idx, 2], c=color)

        ax.set_title(f'PCA-Reduced Feature Visualization for Class {class_label} in 3D')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')

        # Add a caption to indicate the colors
        ax.text2D(0.05, 0.95, 'Blue: Training \nRed: Diffusion with Resblock \nGreen: Diffusion without Resblock',
                  transform=ax.transAxes)

        # Save the image to the output directory
        image_filename = os.path.join(output_directory, f'class_{class_label}_3D_plot.png')
        plt.savefig(image_filename, dpi=300, bbox_inches='tight')

        plt.close()