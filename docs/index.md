# Bruna Ceruti - Portfólio em Engenharia e Análise de Dados

This works addresses the problem of automatic image colorization using deep learning techniques. We explore a way of generating colored images from grayscale images with some degree of spatial control, so we could get partial colored images with an artistic effect if desired.

This project was developed as a conclusion work for the course *<a href="http://lvelho.impa.br/ip18/" target="_blank">Fundamentals and Trends in Vision and Image Processing</a>* (August-November, 2018) at IMPA, which had the theme *from Data to Generative Models*.


# Sobre mim

Possuo graduação em Engenharia e tenho experiência de 8 anos no mercado de Oleo e Gás. Eu sempre fui muito interessada na área de tecnologia, por estruturar e tornar processos mais eficientes e também aprender como eles funcionam. Entendo também que o acesso a informação de todos dentro de uma organização é fundamental para tomada de decisões estratégicas acertadas e vejo que a área dados é uma ferramenta chave para levar a informação correta, precisa e de forma segura para os usuários e gestores da organização.
Meu principal objetivo é contribuir em projetos em diferentes áreas do conhecimento, aprendendendo sempre e conectando pessoas através da análise e engenharia de dados.


## Habilidades

Background: Python, SQL, PowerBI


One of the most impressive and deeply used results to disseminate the advances reached in the area is artistic style transfer. Not only it generates beautiful and astonishing images but also It is something of ease access for someone who is a layperson on the subject. When we show people images like those in Figure 1, they can easily understand the relationship between them and usually ask the question: *"did a computer do that?"*.

<figure>
<img src='https://github.com/hallpaz/colorization-masks-gans/blob/master/docs/imgs/style_transfer.png?raw=true' alt='Artistic style transfer'/>
<figcaption>Figure1: Gatys, 2016</figcaption> 
</figure>

The work of (Gatys, 2016) addresses artistic style transfer using neural networks as a patterns recognizer and an optimization tool, presenting the features that should be computed and used from the layers of the neural network. The paper also describes how to compute an objetive function, or loss function, using these features. We didn't want to try to extend the results on artistic style transfer, but we did want to achieve something that could be used in an artistic way. This application could be either by generating new images directly or by acting as a tool to facilitate artist's work.

## Projetos Publicados em Engenharia e Análise de Dados

Following the ideia of style transfer, one could think about a "color transfer" or "recoloring tool", which could be thought as a a way of transposing the colors of a source image or a selected palette to a target image. Many years ago, Reinhard (Reinhard, 2001) presented a color transfer algorithm based on statistical metrics as the **mean** and the **standard deviation** of the values in each channel of the image — looking at the LAB space. 

In a first moment this approach seemed particularly interesting for us, because machine learning deals heavily with statistics. Trying to figure out a way of training a neural network to manipulate this metrics and generate images with a desired property could be an interesting work. However, once we have metrics like mean, standard deviation or even the histogram of colors of an image, we lose the correlation with spatial information and it's not clear how we could deal with it in the context of neural networks.

<figure>
<img src='https://github.com/hallpaz/colorization-masks-gans/blob/master/docs/imgs/color_transfer.png?raw=true' alt='Artistic style transfer'/>
<figcaption>Figure 2: Reinhard, 2001</figcaption> 
</figure>

On the other hand, acting with spatial information, Levin (Levin, 2004) presented a way of coloring a grayscale a image using some inputs of the user as hints. They present a closed form solution to an optimization problem based on the hypothesis that nearby pixels with similar luminance values should have similar chrominance values. The ideia behind this paper could help building an interesting tool for assisted colorization.


<figure>
<img src='https://github.com/hallpaz/colorization-masks-gans/blob/master/docs/imgs/levin_colorization_optimization.png?raw=true' alt='Colorization using optimization'/>
<figcaption>Figure 3: Levin, 2004</figcaption> 
</figure>

Note that none of these works deals with neural networks, but since they are based on optimization techniques and statistical metrics of the images, it is reasonable to suppose that it's possible to construct and train neural networks capable of reproducing similar results. We didn't follow this path though.


## Experiência Profissional

When we search for image colorization in the context of deep learning we can find a variety of works interested in make a computer guess the colors of a grayscale image making it a photorealistic colored scene. Based only on the value of luminance and some spatial information, we have largely degrees of freedom in the task of computing color information, which makes the problem ill-posed.

Researching about colorization with neural networks, we identified basically 3 strategies to solve the problem. The first one consists of trying to modify the input image extrapolating (or extending) the concept of backpropagation; one can think of this as if we used the inputs to the network as a hidden layer and updated the values with back-propagation. The second ideia is to try to estimate a color for each pixel of the image, something similar to a regression – that's the approach at (Zhang, 2016) for example. The last strategy is the use of Generative Adversarial Networks.


# Contato 

With the idea of making something with a potential artistic style and some knowledge about the solutions which addressed some problems in the colorization area, we decided to make colorization with some degree of spatial control. We would like to build a way to get partial colorization of images and achieve artistic results like those presented in figure 4.

<figure>
<img src='https://github.com/hallpaz/colorization-masks-gans/blob/master/docs/imgs/partial_colorization.png?raw=true' alt='Images with partial colorization'/>
<figcaption>Figure 4: Images with partial colorization (Andrew, 2016)</figcaption> 
</figure>

# Our approach

After studying many projects about colorization with deep learning, we decided to approach the problem using Generative Adversarial Networks (GANs). We found some very good results in other works which didn't use GANs, particularly (Zhang, 2016) and (Luan, 2018). The former doesn't address the colorization problem specifically, but the ideas developed to harmonize the style and colors of images pasted into a painting could be useful. The problem is we didn't figure out a simple way to experiment spatial control in the architectures of networks presented in these works. Besides that, these works had some delicate steps which could need more time to do correctly than we could provide for the project. 

### The network

We based our solution on the code available in 3 different repositories on the Github. We started studying the code at (Zaneri, 2018), which has a detailed paper explaining the results, but we decided to start most of our solution from the network model available at (Han, 2018) and the software architecture available at (Zhao, 2018) because they were written in *Pytorch* – the framework we were most comfortable to work. The generator  architecture is composed of 11 layers with an encoding structure followed by a decoding one, with skipping connections between layers symmetrically disposed. This direct connections between non sequential layers (skipping) are necessary to keep spatial information in the final layers.

<figure>
<img src='https://github.com/hallpaz/Image-Colorization/blob/master/asset/unet.png?raw=true' alt='Images with partial colorization'/>
<figcaption>Figure 5: U-Net architecture of the generator network [Illustration from (Han, 2018)] </figcaption> 
</figure>

We read some works where the discriminator network was trained using some pre-trained network as baseline, for example VGG-16. In (Zaneri, 2018) they built their own model for the discriminator and we chose to use this model which has an architecture that resembles that of the generator, but only the encoding side, leading to an output of the probability that the image taken from the generator is classified as a fake.

<figure>
<img src='http://via.placeholder.com/640x360' alt='Images with partial colorization'/>
<figcaption>Figure 6: [DISCRIMINATOR ARCHITECTURE IMAGE WILL BE FABRICATED] </figcaption> 
</figure>

### The masks

To incorporate spatial control in the colorization made using GANs, we tried to follow the ideia of Conditional GANs (Mirza, 2014), changing the generator network to receive as input not only a grayscale image but also a binary image which would act as a mask. The discriminator network was also modified to receive the same mask in addition to the colored image. In both cases the mask ended up concatenated as another channel of the other image. The last change we had to do was to add a term to the total loss function, a mask loss, penalizing the distance between the generated colors and the desired colors with the squared L2 norm. With this approach we expected to teach the network to color only the regions where the pixels had a non-zero value. 

As we weren't sure if this approach would prove itself promising, we tested our hypothesis over the 17 Categories Flower Dataset, training the networks using 1360 images with a single synthetic mask, an image divided vertically. We expected the network to learn how to generate images half in color and half in grayscale. Unfortunately, this didn't work, as the network began to generate images completely colored, but with a very visible line where it was the hard division of the mask (figure 7). Before training the modified network with masks, we trained the original networks using only the images from our selected dataset to check if it was enough to make the network learn how to color the flowers. The colorization obtained was "fine", with a bias to the yellow color. The network identified the flowers and usually painted them with a yellow color, but the leaves, the sky and other elements were painted reasonably.

<figure>
<img src='https://github.com/hallpaz/colorization-masks-gans/blob/master/docs/imgs/colored_samples_with_line.png?raw=true' alt='Images with partial colorization'/>
<figcaption>Figure 7: Results obtained using a single mask with half of the pixels active and the other half inactive </figcaption> 
</figure>

Although the results of this experiment weren't encouraging, we decided to try again using a set of masks (figure 8), instead of a single one. We made 10 synthetic masks using stripes and grids in a graphic editor software and trained the network again (starting from random weights). This time we got results near to that we were expecting, as we show in figures 9, 10 and 11.

<figure>
<img src='https://github.com/hallpaz/colorization-masks-gans/blob/master/docs/imgs/masks.png?raw=true' alt='Images with partial colorization'/>
<figcaption>Figure 8: Set of synthetic masks</figcaption> 
</figure>


<figure>
<img src='https://github.com/hallpaz/colorization-masks-gans/blob/master/docs/imgs/half_vertical_flower.png?raw=true' alt='Images with partial colorization'/>
<figcaption>Figure 9: Generated with half vertical mask</figcaption> 
</figure>

<figure>
<img src='https://github.com/hallpaz/colorization-masks-gans/blob/master/docs/imgs/stripes_horizontal_flower.png?raw=true' alt='Images with partial colorization'/>
<figcaption>Figure 10: Generated with horizontal stripes mask</figcaption> 
</figure>

<figure>
<img src='https://github.com/hallpaz/colorization-masks-gans/blob/master/docs/imgs/grid_white_flower.png?raw=true' alt='Images with partial colorization'/>
<figcaption>Figure 11: Generated with a grid mask</figcaption> 
</figure>
