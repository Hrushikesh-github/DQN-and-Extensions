# DQN-and-Extensions-on-Pong

Using Deep Q-Networks and it's Extensions N-step DQN and Double DQN to solve [pong-an atari game](https://gym.openai.com/envs/Pong-v0/)

DQN applies Q-learning using a Neural Network. 

Various observation wrappers need to be applied for better training:
1.Wrapper that converts input observation from emulator to grayscale 84*84 and scales it to (0, 1)
2.A Wrapper that creates a stack of subsequent frames along the 1st dimension and returns them as an observation. 
3.Wrapper that takes the maximum of pixel values from the stack of frames.
These have been implemented in the wrappers.py module although gym provides a alternative .

Parameters such as batch size, learning rate etc are as recommended from the paper. Replay buffer size was however reduced.

An episode in Pong is considered completed when a player gets 21 points. Reward system is +/- 1 for each win/loss respectively. The game is assumed to be solved when we obtain a mean reward of +19.

Here is our model performing when we just start training, where the mean reward obtained is -20. 


![hope2](https://user-images.githubusercontent.com/56476887/102717165-91ce1900-4306-11eb-87cb-3461df5f1eec.gif)


The training is completed after going through around 510K frames. It took approx 3 hours on GeForce 940M using 400 MB. Increasing # of batches could speed up training but unlike image classification, the training here is dependent on batch number. This is because by changing the number of transitions sampled from the replay buffer, the network would see more or less better/worse transitions depending on the epsilon value (which is decaying) picked by the agent. 


![Screenshot from 2020-12-20 23-16-03](https://user-images.githubusercontent.com/56476887/102720374-eaf37800-4319-11eb-9b3e-ae2408f33998.png)


The frames per second (fps) processed in training was around 27 and the mean reward (averaged over last 100 episodes) obtained
![normal_fps](https://user-images.githubusercontent.com/56476887/102718670-5552eb00-430f-11eb-8e08-d872bee46299.png)


![normal_reward](https://user-images.githubusercontent.com/56476887/102718675-57b54500-430f-11eb-94f9-08905a1be623.png)

Here is the model in two stages when mean reward is +5 and when the game is solved:

![pong](https://user-images.githubusercontent.com/56476887/102718563-db226680-430e-11eb-8b7e-b3206806978c.gif) ---->
![pong2](https://user-images.githubusercontent.com/56476887/102718576-effefa00-430e-11eb-8415-128441967b77.gif)


# N-Step DQN and Double DQN 

A simple DQN is off policy, whereas N-Step DQN is on-policy. In N-Step, to calculate Q(s,a), rewards are unrolled usually 3 to 4 transitions (I have done it 4 times). Higher values of unrolling may not lead to convergence. N-step DQN speeds up training.  

A Double DQN alternative tries to explain that the basic DQN over estimates Q-values. This is more prominent in the starting of the training where training is noisy, and we use argmax operation to pick the best action over the noisy numbers. To make the training more robust we choose actions for the next state using the trained network, but take values of Q from the target network. This overestimation has been showed to improve over estimation.

In the following graph, blue denotes the params from training of N-step DQN and red that of Double DQN. Both are trained with the same parameters used for training the basic DQN version.

Looking at the fps value, we see that N-step DQN has an average FPS of 32, a good improvement. 

Double DQN has 27 fps, similar to that of basic DQN.


![fps](https://user-images.githubusercontent.com/56476887/102718606-0a38d800-430f-11eb-8dc6-11e33fb0090d.png)



Steps vs episode also shows us that N-step DQN solves the environment in lesser number of episodes 

compared to Double DQN


![steps_per_episode](https://user-images.githubusercontent.com/56476887/102718625-1f156b80-430f-11eb-94da-68d8582ecc86.png)


Double DQN took 8 hours of training on my GPU, which is more than twice the time taken for N-step or basic DQN. 

Also, Double DQN's convergence has lesser percentage compared to others, one of my training of Double DQN didn't appear to converge. 


![Screenshot from 2020-12-20 23-19-26](https://user-images.githubusercontent.com/56476887/102720377-ee86ff00-4319-11eb-8b18-9f040857e09e.png)
![Screenshot from 2020-12-16 18-28-46](https://user-images.githubusercontent.com/56476887/102718620-191f8a80-430f-11eb-886f-c39d266f6b35.png)


THe purpose of using Double DQN can be seen here, the blue denotes the average loss in the training process for N-step and red that of Double DQN. A lesser loss is an indication that Q-value predicted by the network is not higher than the ideal. Since N-step DQN doesn't consider fixing the over-estimation, it's loss value is higher. 


Average reward over 100 episodes. Note that the basic DQN took over 500K iterations to complete whereas N-step DQN does it within 250K iterations. An incredible performance.

![avg_loss](https://user-images.githubusercontent.com/56476887/102718595-ff7e4300-430e-11eb-88e5-b63f51e0c6e2.png)
![avg_reward](https://user-images.githubusercontent.com/56476887/102718603-060cba80-430f-11eb-9e8e-a7c699e27ee0.png)



Other methods such as Dueling DQN, Prioritzed DQN networks are more popular and RAINBOW DQN uses the best of all these modifications to implement the DQN.


Tensorboard loogers, Recordings, Models at various stages of training are uploaded
