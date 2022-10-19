# Ford project:

## Test Siamese networks and different loss functions on MNIST dataset
### instalation: 
- create Venv
- install requirements.txt in thes env
- run code



### BCE_LOSS:

Input is pair of images, output is 1 if the images are the same number, 0 is different Numbers
BCE is used. the embedings are concatenated and sigmoit is used to get output between 0 and 1. 


### TRIPLET LOSS:

Input is a triplet of images. One achor image, One Positve image and one Negative Image. 
The loss is so that the distace between the anchor and the Positve embeddign is minimised, and the distace between the anchor and the negative embedding is maximised. The network creates clusters of embeddings. (same img close, differt img far away in embedded space)
Classification is done to calculate the embeding distance to a set of querryimages. (Here one image of each class)






