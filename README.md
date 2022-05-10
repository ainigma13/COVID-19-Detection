# COVID-19-Detection
A Python GUI application based on Tkinter  to detect COVID-19 from normal people using chest X-ray images


<img align="center" src="https://github.com/ainigma13/COVID-19-Detection/blob/main/pic/3.png" width="900" height="400">

**1."train_covid19.py"**

Purpose:To use tranfer learning method to solve binary classification problem of classifying chest X-ray images as COVID-19 or normal.

CODE description and usage:

i.Load dataset:
Run the python program in terminal and pass the dataset as command line argument (python train_covid19.py --dataset dataset).

ii.Pre-processing:
Read the images in dataset (cv2.imread()).
Swap color channels of images (cv2.COLOR_BGR2RGB)
Resize the images to 224x224 pixels (cv2.resize)
Perform one-hot encoding on the labels.(LabelBinarizer())
Store the images and labels in numpy arrays.

iii.Model architecture:
Define hyperparameters- learning rate, batch size, and number of epochs.
Partition the dataset into training and testing sets.(train_test_split())
Load the VGG16 as base model with weights pre-trained on ImageNet and excluding the ImageNet classifier on top
Construct the new model to be placed on top of base model.
Freeze the base model inorder to stop updating the layers in base model during the process of training.

iv.Model evaluation:
Compile the model with ADAM optimizer and binsry crossentropy as loss function.
Train model with training dataset and evaluate with testing dataset.
Plot training loss and accuracy on dataset vs number of epochs.
Plot the confusion matrix.
Save the model on disk.

<img align="center" src="https://github.com/ainigma13/COVID-19-Detection/blob/main/pic/COVID-19_Detection.gif" width="640" height="480">


**2."gui_covid19.py"**

Purpose:To develope a GUI that will use the trained model to predict a new chest X-ray image as COVID-19 or normal.

CODE description and usage:

i.Image selection:
Run the python program in terminal and pass the previously saved model as a command line argument (python gui_covid19.py --model covid19.model)
Press the “Load ” button in the GUI to browse for the location of new chest X-ray image we want to predict.

ii.Pre-processing:
Read the image from its path location (cv2.imread()).
Swap color channels of images (cv2.COLOR_BGR2RGB)
Resize the images to 224x224 pixels (cv2.resize)
Change the image to numpy array.

iii.Label prediction:
Predict the label for the new image by passing it to previously evaluated model.
Display the new image and its predicted label. (cv2.imshow())

