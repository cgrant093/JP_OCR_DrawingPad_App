
import os
import torch 

# set device to 'cuda' if available, otherwise cpu' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# define model hyperparameters
HYPERPARAMETERS = {
    'LR' : 1e-3,
    'WT_DECAY' : 1e-2,
    'NUM_EPOCHS' : 50,
    'IMG_PX_LEN' : 28,
    'LATENT_DIM' : 2,
    'HIDDEN_DIM' : 512,
    'BATCH_SIZE' : 128
}

# PATIENCE = 2
# IMAGE_SIZE = 32
# CHANNELS = 1
# BATCH_SIZE = 64
# EMBEDDING_DIM = 2
# SHAPE_BEFORE_FLATTENING = (128, IMAGE_SIZE // 8, IMAGE_SIZE // 8)

# # define model hyperparameters
# LR = 0.001
# PATIENCE = 2
# IMAGE_SIZE = 32
# CHANNELS = 1
# BATCH_SIZE = 64
# EMBEDDING_DIM = 2
# EPOCHS = 100
# SHAPE_BEFORE_FLATTENING = (128, IMAGE_SIZE // 8, IMAGE_SIZE // 8)

# # create output directory
# output_dir = "output"
# os.makedirs("output", exist_ok=True)

# # create the training_progress directory inside the output directory
# training_progress_dir = os.path.join(output_dir, "training_progress")
# os.makedirs(training_progress_dir, exist_ok=True)

# # create the model_weights directory inside the output directory
# # for storing variational autoencoder weights
# model_weights_dir = os.path.join(output_dir, "model_weights")
# os.makedirs(model_weights_dir, exist_ok=True)

# # define model_weights
# MODEL_WEIGHTS_PATH = os.path.join(model_weights_dir, "best_vae.pt")

# # define reconstruction & real before training images paths
# FILE_RECON_BEFORE_TRAINING = os.path.join(output_dir, "reconstruct_before_train.png")
# FILE_REAL_BEFORE_TRAINING = os.path.join(output_dir, "real_test_images_before_train.png")

# # define reconstruction & real after training images paths
# FILE_RECON_AFTER_TRAINING = os.path.join(output_dir, "reconstruct_after_train.png")
# FILE_REAL_AFTER_TRAINING = os.path.join(output_dir, "real_test_images_after_train.png")

# # define latent space and image grid embeddings plot paths
# LATENT_SPACE_PLOT = os.path.join(output_dir, "embedding_visualize.png")
# IMAGE_GRID_EMBEDDINGS_PLOT = os.path.join(output_dir, "image_grid_on_embeddings.png")

# # define linearly and normally sampled latent space reconstructions plot paths
# LINEARLY_SAMPLED_RECONSTRUCTIONS_PLOT = os.path.join(output_dir, "linearly_sampled_reconstructions.png")
# NORMALLY_SAMPLED_RECONSTRUCTIONS_PLOT = os.path.join(output_dir, "normally_sampled_reconstructions.png")

# # define class labels dictionary
# CLASS_LABELS = {
#     0: "T-shirt/top",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle boot",
# }