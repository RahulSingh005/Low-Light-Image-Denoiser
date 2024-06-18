# Low-Light-Image-Denoiser 
#### Associated with VLG IIT Roorkee
## Table of Contents
* Introduction
* Dataset
* Models Tried
* Final Model Code
* Results
* Potential Improvements
* Application
* References

## Introduction
This Project aims to develop a image denoising model to remove noise from the images while preserving important details and structures. Different types of Noises such as Gaussian Noise, Salt-and-pepper noise etc. can arise in images during acquisition or processing, and can reduce image quality and make it difficult to interpret.

![image](https://github.com/RahulSingh005/Low-Light-Image-Denoiser/assets/134304672/b66bbc17-ddb5-4964-81e0-3a65375fbb26)



              Clean Image                                 Noised Image

## Dataset
* The dataset consists of both High light and Low light images. 
* Training Dataset includes 437 High light and their corresponding Low light images.
 * Test Dataset includes 48 High Light and their corresponding low light images.  

 ## Models Tried   
 The main models tried are:-

 * DnCNN (Denoising Convolutional Neural Networks)
 * U-NET
 * CG-Net

### DnCNN (Denoising Convolutional Neural Networks)
DnCNN (Denoising Convolutional Neural Network) is implemented using deep convolutional layers with residual connections. It learns to map noisy images to clean ones by minimizing a loss function like MSE during training. This architecture is effective for various denoising tasks in image processing.

#### Results:-
* Average PSNR: 18.289548114975243
* MSE: 0.01817338318325739
* MAE: 0.10775301897974217

### U-NET
U-Net is a convolutional neural network architecture designed for biomedical image segmentation tasks.
It consists of a contracting path to capture context and an expansive path for precise localization.
Skip connections between mirrored layers help preserve spatial information crucial for accurate segmentation.

#### Results:-
* Average PSNR: 21.476344380257814
* MSE: 0.011056872089780232
* MAE: 0.08457357411466772

### CG-NET
CGNet (Context Gating Network) is a lightweight convolutional neural network designed for efficient semantic segmentation.
It employs a context gating mechanism to enhance feature representation by selectively emphasizing informative features.

#### Results:-
* Average PSNR: 17.39830354230752
* MSE: 0.02107877942757838
* MAE: 0.11978779146134835

## Final Model
### U-Net Model
* #### Model Architecture Overview:
  The model follows a U-Net structure with residual blocks (similar to ResNet).
  Each block consists of two convolutional layers with batch normalization and ReLU activation.
  Residual connections are added between convolutional layers to help propagate gradients and preserve information.
* #### Encoder (Contracting Path):

  Starts with a series of convolutional layers followed by residual blocks.
  Downsamples the spatial dimensions using max pooling to capture context.

* #### Decoder (Expanding Path):

  Utilizes upsampling layers to gradually increase the spatial resolution.
  Concatenates feature maps from the encoder to preserve detailed information through skip connections.
* #### Output Layer:

  The final layer uses a convolution with sigmoid activation to output denoised images with three channels (RGB).
* #### Training Setup:

  Optimizer: Adam optimizer with a specified learning rate.

  Loss function: Mean squared error (MSE) used to measure the difference between predicted and ground truth clean images.

  Callbacks: Early stopping to prevent overfitting and reduce learning rate on plateau to fine-tune training.
* #### Calculations:
  * Mean Absolute Error (MAE):
    
     ![download](https://github.com/RahulSingh005/Low-Light-Image-Denoiser/assets/134304672/47f51e69-0fb2-4b3f-bd59-ed2e9a072361)



  * Mean Squared Error (MSE):
    
       ![1_BtVajQNj29LkVySEWR_4ww](https://github.com/RahulSingh005/Low-Light-Image-Denoiser/assets/134304672/61c45ad8-3bf6-4a60-a229-dbc3c0b40116)

  * Peak Signal-to-Noise Ratio (PSNR):
    
     ![images](https://github.com/RahulSingh005/Low-Light-Image-Denoiser/assets/134304672/9803dd17-0c05-4479-a092-56e66745dce8)


###   Code
```ruby
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, UpSampling2D, Conv2DTranspose
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def load_images_from_directory(directory, size=(32, 32)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            img = img.resize(size)
            img = np.array(img) / 255.0
            images.append(img)
    return np.array(images)

train_noisy_dir = 'Path of Train Noisy Image '
train_clean_dir = 'Path of Train Clean Image'
test_noisy_dir = 'Path of Test Clean Image'
test_clean_dir = 'Path of Test Clean Image'

train_noisy_images = load_images_from_directory(train_noisy_dir)
train_clean_images = load_images_from_directory(train_clean_dir)
test_noisy_images = load_images_from_directory(test_noisy_dir)
test_clean_images = load_images_from_directory(test_clean_dir)

train_noisy_images = train_noisy_images.astype('float64')
train_clean_images = train_clean_images.astype('float64')
test_noisy_images = test_noisy_images.astype('float64')
test_clean_images = test_clean_images.astype('float64')

def residual_block_lab(input_tensor, filters):
    layer = Conv2D(filters, 3, padding='same')(input_tensor)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer = Conv2D(filters, 3, padding='same')(layer)
    layer = BatchNormalization()(layer)

    shortcut = Conv2D(filters, 1, padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    layer = Add()([x, shortcut])
    layer = Activation('relu')(layer)
    return layer

def unet_model_with_residuals(input_size=(32, 32, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = residual_block_lab(conv1, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = residual_block_lab(conv2, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = residual_block_lab(conv3, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = residual_block_lab(conv4, 256)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(up5)
    conv5 = residual_block_lab(conv5, 128)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(up6)
    conv6 = residual_block_lab(conv6, 64)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(up7)
    conv7 = residual_block_lab(conv7, 32)

    conv8 = Conv2D(3, 1, activation='sigmoid')(conv7) 

    model = Model(inputs=[inputs], outputs=[conv8])

    return model

class PSNRCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(PSNRCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        noisy_images = self.validation_data[0]
        clean_images = self.validation_data[1]
        
        denoised_images = self.model.predict(noisy_images)
        
        psnrs = []
        for i in range(len(clean_images)):
            psnr = calculate_psnr(clean_images[i], denoised_images[i])
            psnrs.append(psnr)
        
        avg_psnr = np.mean(psnrs)
        
        print(f' - val_psnr: {avg_psnr:.2f}')

def calculate_psnr(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

model = unet_model_with_residuals((32, 32, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_squared_error')

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=20, min_lr=0.000001)
    PSNRCallback(validation_data=(test_noisy_images, test_clean_images))
]

history = model.fit(train_noisy_images, train_clean_images, epochs=50, batch_size=32, validation_data=(test_noisy_images, test_clean_images), callbacks=callbacks)

denoised_images = model.predict(test_noisy_images)

psnr_scores = [calculate_psnr(test_clean_images[i], denoised_images[i]) for i in range(len(test_clean_images))]
avg_psnr = np.mean(psnr_scores)
mse_score = np.mean((test_clean_images - denoised_images) ** 2)
mae_score = np.mean(np.abs(test_clean_images - denoised_images))

print(f'Average PSNR: {avg_psnr}')
print(f'MSE: {mse_score}')
print(f'MAE: {mae_score}')

def load_images_from_directory(directory, size=(600, 400)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            img = img.resize(size)
            img = np.array(img) / 255.0
            images.append(img)
    return np.array(images)

test_noisy_images_original = load_images_from_directory(test_noisy_dir)
test_clean_images_original = load_images_from_directory(test_clean_dir)

from PIL import Image
import numpy as np
denoised_images_resized = []

for img_array in denoised_images:
    img = Image.fromarray((img_array * 255).astype(np.uint8)) 
    img_resized = img.resize((600, 400), Image.Resampling.LANCZOS)
    denoised_images_resized.append(np.array(img_resized))

denoised_images_resized = np.array(denoised_images_resized)

n = 2
plt.figure(figsize=(20, 10))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(test_noisy_images[i])
    plt.title("Noisy")
    plt.axis("off")

    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(test_clean_images[i])
    plt.title("Clean")
    plt.axis("off")

    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(denoised_images_resized[i])
    plt.title("Denoised")
    plt.axis("off")
plt.show()
```
### Results
* #### Images:-
  ![Screenshot 2024-06-18 201400](https://github.com/RahulSingh005/Low-Light-Image-Denoiser/assets/134304672/2fb011aa-103d-4d73-a721-077c1ace4fe4)

* #### PSNR Over Time:-
  ![Screenshot 2024-06-18 204708](https://github.com/RahulSingh005/Low-Light-Image-Denoiser/assets/134304672/63cfd4ec-7acb-48eb-b348-42f7284df47c)

* #### Scatter Plot of Prediction:-
  ![Screenshot 2024-06-18 210550](https://github.com/RahulSingh005/Low-Light-Image-Denoiser/assets/134304672/18a9848e-6d3c-4a19-be6d-4556fccd79a5)


### Potential Improvements:
  * I have compressed the images to 32*32 size from 600*400 size due to Storage and RAM issue due to which I a was unable to extract large features and this cause low PSNR Value.
  * Other Models with more complex architecture can be used to improve PSNR Value.  

### Application
* Photography: Enhancing the quality of photos taken in low-resolution settings.
* Satellite Imagery: Improving the resolution of satellite images for better analysis and interpretation.
* Medical Imaging: Enhancing the quality of medical images like MRIs and CT scans.

### References
  * https://paperswithcode.com/task/image-denoising
  * https://alain.xyz/blog/machine-learning-denoising
  * https://arxiv.org/abs/1505.04597
  * https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00444-8
  * Extreme Low-Light Single-Image Denoising Model- Stanford University (PDF)
