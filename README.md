# Pytorch_MNIST
Pytorch CNN &amp; VAE using MNIST dataset
MNIST is a prefect start point to dive into deep learning and train models.

I used MNIST dataset to conduct two mini-projects.

1. Compare GD vs GD with Momentum to learn the effect of momentum in training updates

2. Variational Auto Encoder to learn latent representation and generate same figure: https://colab.research.google.com/drive/1fN4sxUk90rN5eoYck9aNCIcXmYgjGqMV

The encoder is a neural network. Its input is a datapoint x, its output is a hidden representation z, and it has weights and biases.To be concrete, in my model, xx is a 28 by 28-pixel photo of a handwritten number. The encoder ‘encodes’ the data which is 784x784 dimensional into a latent (hidden) representation space z, which is much less than 784x784 dimensions. 

The decoder is another neural net. Its input is the representation z, it outputs the parameters to the probability distribution of the data, and has weights and biases. Running with the handwritten digit example, let’s say the photos are black and white and represent each pixel as 0 or 1. The probability distribution of a single pixel can be then represented using a Bernoulli distribution. The decoder gets as input the latent representation of a digit z and outputs 784x784 Bernoulli parameters, one for each of the 784x784 pixels in the image. The decoder ‘decodes’ the real-valued numbers in z into 784x784 real-valued numbers between 0 and 1. 
