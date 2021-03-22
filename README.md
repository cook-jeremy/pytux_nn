# pytux_nn
A deep neural network trained to playing pysupertux kart. 

The main model used to play is the PuckDetector model, which detects the puck location in the image. If the puck is not in the image, then the model predicts a location behind the player so the AI knows to reverse and find the puck. The model is based on an FCN model, and uses up blocks, down blocks, and skip connections. In order to get this model to generalize properly, the data collection process had to be modified. First, instead of collecting data from every frame of the game, an image of the game is saved only every 10 frames to get a more diverse set of image locations on the field. Second, the data collection and model training was put in a loop using a shell script, so that every 30 epochs the model would receive a new batch of 10,000 images to train on, which drastically improved the model’s ability to detect whether the puck was in frame or not. Below are some sample images of the puck detector in action, where the green circle is the predicted puck location by the neural network, and the red circle is the adjusted target position for the player to score.

<img width="860" alt="Screen Shot 2021-03-21 at 10 23 30 PM" src="https://user-images.githubusercontent.com/12803067/111936707-47d70980-8a94-11eb-9e52-6b9ba59ee2ef.png">
