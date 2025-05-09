#pip install tensorflow matplotlib numpy pillow
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras import models

# Load the content and style images
def load_and_process_image(image_path):
    img = image.load_img(image_path, target_size=(512, 512))  # Resize for faster computation
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

# De-process image to display
def deprocess_image(img):
    img = img.reshape((512, 512, 3))
    img = img[:, :, ::-1]  # Convert back to BGR
    img += np.array([103.939, 116.779, 123.68])  # Add back the means
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# Load the VGG19 model with pre-trained ImageNet weights
def get_model():
    vgg_model = VGG19(weights='imagenet', include_top=False)
    vgg_model.trainable = False
    return models.Model(inputs=vgg_model.input, outputs=vgg_model.output)

# Load and process the content and style images
content_image_path = 'path_to_your_content_image.jpg'  # Replace with your image path
style_image_path = 'path_to_your_style_image.jpg'      # Replace with your style image path

content_image = load_and_process_image(content_image_path)
style_image = load_and_process_image(style_image_path)

# The model
model = get_model()

# Define the layers for feature extraction (content and style)
content_layers = ['block5_conv2']  # You can experiment with different layers
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Extract features from the model
def get_features(model, content_image, style_image):
    content_outputs = model(content_image)
    style_outputs = model(style_image)
    return content_outputs, style_outputs

# Define loss functions (style and content loss)
def compute_content_loss(content, generated):
    return tf.reduce_mean(tf.square(content - generated))

def compute_style_loss(style, generated):
    style_loss = tf.reduce_mean(tf.square(style - generated))
    return style_loss

# Create the generated image by applying style transfer
def style_transfer(content_image, style_image, model, num_iterations=1000, learning_rate=0.02):
    # Initialize the generated image
    generated_image = tf.Variable(content_image)
    
    optimizer = tf.optimizers.Adam(learning_rate)
    
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            
            content_features, style_features = get_features(model, content_image, style_image)
            generated_features = get_features(model, generated_image, style_image)
            
            content_loss = compute_content_loss(content_features, generated_features)
            style_loss = compute_style_loss(style_features, generated_features)
            
            total_loss = content_loss + style_loss
            
        grads = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(grads, generated_image)])
        
        if i % 100 == 0:
            print(f"Iteration {i}: Content Loss = {content_loss}, Style Loss = {style_loss}")
            plt.imshow(deprocess_image(generated_image.numpy()))
            plt.show()
    
    return generated_image

# Run style transfer
generated_image = style_transfer(content_image, style_image, model)

# Display the final result
plt.imshow(deprocess_image(generated_image.numpy()))
plt.show()
