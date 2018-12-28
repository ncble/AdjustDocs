# AdjustDocs


### Objective
This simple/toy project amis to develop an application that adjusts documents after taking pictures. Most of mobile phone are capable of adjusting natural images but not documents (especially the handwritten documents). 


### Dataset
The (private) dataset is constituted by

- More than 5,000 pictures of my personal handwritten scribe note since 6 years. (mostly scientific note, including mathematics, physics and chemistry)
- More than 1,000 Resumes
- More than 10,000 pages of mathematic/physic documents/papers (typed)
- More than 500 administrative documents (bill, tax form, contract, etc)
- More than 10,000 natural pictures
- More than 5,000 graffiti, illustration, manga, animation...etc

### Results

Developing a network that fits for the dataset with a such diversity isn't simple. By focusing on both handwritten and typed documents, I have achieved **99,93% accuracy** for four classes (0, 90, 180, 270 degrees rotation) classification. Note that in the handwritten collection, *some pictures are alomost blank with scribbled note*. The main idea is to pass a filter (unsupervised) that crops the interesting area then to train the network on patch of images. As a comparison, without these preprocessing, the accuracy is only 70%.


Further improvement could be done by introducing some transfer learning techniques. (TODO)



