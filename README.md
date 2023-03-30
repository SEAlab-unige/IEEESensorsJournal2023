# IEEESensorsJournal2023
The repository contains an example of the models proposed in the paper "Affordance segmentation using tiny networks for sensing systems in wearable robotic devices". The manuscript has been submitted to IEEE sensor Journal and is currently under review.    

This repository contains the proposed affordance networks obtained using HW-NAS. The pretrained networks are provided to easily inspect the generated architectures. The networks were trained using the foreground of the object from the UMD dataset. In addition, a script shows an example of prediction using the proposed model is provided.


## Python project
### Requirements
The requirements to run the python code are the following:
* Python 3.7 (64-bit)
* Tensorflow 2.X
* OpenCV

### Description
There are 2 folders and 1 python script:
* `Models`: holds the main model for affordance detection (`MobileNetV1_UNET`) in *TFLite* format.
* `ExampleInputs`: holds some images used during the inference phase.
* `Demo.py`: execute the selected neural network on the images contained into the ExampleInputs folder.

