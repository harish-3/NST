import tensorflow as tf
import numpy as np
from PIL import Image
import time

class NeuralStyleTransfer:
    def __init__(self):
        # Load VGG19 model with a try-except block to handle initialization
        try:
            # Initialize TensorFlow's name scope
            with tf.name_scope('VGG19'):
                self.model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
            self.model.trainable = False
        except Exception as e:
            print(f"Error loading VGG19: {str(e)}")
            raise
        
        # Content and style layers
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1',
                           'block2_conv1',
                           'block3_conv1',
                           'block4_conv1',
                           'block5_conv1']
        
        # Get the style and content feature extractor
        self.extractor = self.get_feature_extractor()

    def load_img(self, path_to_img):
        max_dim = 256
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        # Ensure both images are resized to the same dimensions
        img = tf.image.resize_with_pad(img, max_dim, max_dim)
        img = img[tf.newaxis, :]
        return img

    def get_feature_extractor(self):
        # Build a model with our selected layers
        style_outputs = [self.model.get_layer(name).output for name in self.style_layers]
        content_outputs = [self.model.get_layer(name).output for name in self.content_layers]
        model_outputs = style_outputs + content_outputs

        return tf.keras.Model([self.model.input], model_outputs)

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def style_content_loss(self, outputs, style_targets, content_targets, style_weight, content_weight):
        """Calculate style and content loss with explicit weights"""
        style_outputs = outputs[:len(self.style_layers)]
        content_outputs = outputs[len(self.style_layers):]

        # Calculate style loss with user-defined weight
        style_loss = tf.add_n([tf.reduce_mean((tf.image.resize(style_outputs[i], tf.shape(style_targets[i])[1:3]) - style_targets[i])**2)
                              for i in range(len(style_outputs))])
        style_loss *= style_weight / len(self.style_layers)

        # Calculate content loss with user-defined weight
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[i] - content_targets[i])**2)
                                for i in range(len(content_outputs))])
        content_loss *= content_weight / len(self.content_layers)

        total_loss = style_loss + content_loss
        return total_loss

    def train_step(self, image, style_targets, content_targets, style_weight, content_weight, opt):
        """Training step with explicit style and content weights"""
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs, style_targets, content_targets, 
                                         style_weight=style_weight,  # Explicitly use style_weight
                                         content_weight=content_weight)  # Explicitly use content_weight

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, 0.0, 1.0))
        return loss

    def finalize_image(self, image):
        final_image = tf.squeeze(image, axis=0)
        final_image = tf.keras.preprocessing.image.array_to_img(final_image)
        return final_image

    def transfer_style(self, content_path, style_path, iterations=100, content_weight=1e4, style_weight=1e-2):
        content_image = self.load_img(content_path)
        style_image = self.load_img(style_path)

        # Extract style and content features
        style_targets = [self.gram_matrix(style) for style in 
                        self.extractor(style_image)[: len(self.style_layers)]]
        content_targets = self.extractor(content_image)[len(self.style_layers):]

        # Initialize image to be optimized
        image = tf.Variable(content_image)

        # Optimization settings
        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        
        # Training loop
        for n in range(iterations):
            loss = self.train_step(image, style_targets, content_targets, style_weight, content_weight, opt)

        # Convert to PIL image
        final_image = self.finalize_image(image)
        
        return final_image
