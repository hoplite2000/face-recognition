# Face Recognition

This face recognition module is built using OpenCV. I have tested this module and it worked well with laptop webcam over a range of 3 meters.

### Execution:
1. Create your custom directory with the person's name and add some sample pics in it inside dataset directory of those individuals whom you want your module to recognize (it would be better if you name it in a pattern) and add some unknown images also in unknown directory. I have provided some images and even tried with my own images, it works well and good.
2. Execute embedder.py file by typing "python embedder.py" in terminal or command prompt to make embeddings for the dataset.
3. Then trainer.py by executing "python trainer.py" to train your face recognition model.
4. and recognizer.py by executing "python recognizer.py" to run the module.

Done, the live video capture has started and recognizes faces.

### Directories : 
1. face_detection : Contains a pre-trained Caffe deep learning model provided by OpenCV to detect faces. This model detects and localizes faces in an image.

2. embedder : A Torch deep learning model which produces the 128-D facial embeddings. We’ll be using this deep learning model to prepare encoding for dataset.

3. outputs : The outputs of the embedder and trainer module are saved in output directory. This includes output pickle files. If you’re working with your own dataset, you can store your output files here as well.
* embeddings.pickle : A serialized facial embeddings file. Embeddings have been computed for every face in the dataset and are stored in this file.
* le.pickle : Our label encoder. Contains the name labels for the people that our model can recognize.
* recognizer.pickle : Our model. This is responsible for actually recognizing faces.

### Note:
* For better results increase the images in dataset. Dataset must contain 10-20 images of an individual.
* Works well in brightness.
* If working on a linux system use python3 command in the terminal.

### Requirements:
1. OpenCV 3.3 or above
2. sklearn

