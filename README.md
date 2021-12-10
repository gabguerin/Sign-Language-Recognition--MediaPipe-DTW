# Sign Language Recognition - using MediaPipe and DTW

This repository proposes an implementation of a Sign Recognition Model using the **MediaPipe** library 
for keypoint extraction and **Dynamic Time Warping** (DTW) as a similarity metric between signs.

![Results with a small french language sign dictionary](https://media.giphy.com/media/4xQRRkUOgxox6ltTWs/giphy-downsized-large.gif)

#### Source : https://www.sicara.ai/blog/
___

## Set up

### 1. Open terminal and go to the Project directory

### 2. Install the necessary libraries

- ` pip install -r requirements.txt `

### 3. Download or Import Videos of signs which will be considered as reference
To create a small dataset of French signs run:

- ` python yt_download.py `
> N.B. The more videos for each sign you import, the better the prediction will be.

### 4. Run the main file

- ` python main.py `

### 5. Press the "r" key (until the circle turns red) to record the sign. 

___
## Code Description

### *Landmark extraction (MediaPipe)*

- The **Hollistic Model** of MediaPipe allows us to extract the keypoints of the Hands, Pose and Face models.
For now, the implementation only uses the Hand model to predict the sign.


- **MediaPipe's Hand model**:

![This is an image](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)


### *Hand Model*

- In this project a **HandModel** has been created to define the Hand gesture at each frame. 
If a hand is not present we set all the positions to zero.

- In order to be **invariant to orientation and scale**, the **feature vector** of the
HandModel is a **list of the angles** between all the connexions of the hand.

### *Sign Model*

- The **SignModel** is created from a list of landmarks (extracted from a video)

- For each frame, we **store** the **feature vectors** of each hand.

### *Sign Recorder*

- The **SignRecorder** class **stores** the HandModels of left hand and right hand for each frame **when recording**.
- Once the recording is finished, it **computes the DTW** of the recorded sign and 
all the reference signs present in the dataset.
- Finally, a **voting logic** is added to output a result only if the prediction **confidence** is **higher than a threshold**.

### *Dynamic Time Warping*

-  DTW is widely used for computing **Time Series similarity**.

![This is an image](https://www.researchgate.net/profile/Andrea-Cannata/publication/233751236/figure/fig4/AS:300128591204353@1448567640170/Difference-between-DTW-distance-and-Euclidean-distance-green-lines-represent-mapping.png)

- In this project, we compute the DTW of the variation of hand connexion angles over time.

___

## References

 - [Pham Chinh Huu, Le Quoc Khanh, Le Thanh Ha : Human Action Recognition Using Dynamic Time Warping and Voting Algorithm](https://www.researchgate.net/publication/290440452)
 - [Mediapipe : Pose classification](https://google.github.io/mediapipe/solutions/pose_classification.html)
