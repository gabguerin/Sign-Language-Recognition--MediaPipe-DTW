import pandas as pd
import numpy as np
from collections import Counter

from pyparsing import Word

from utils.dtw import dtw_distances
from models.sign_model import SignModel
from utils.landmark_utils import extract_landmarks

import os, boto3
from contextlib import closing
import sys
from tempfile import gettempdir
import playsound

class SignRecorder(object):
    def __init__(self, reference_signs: pd.DataFrame, seq_len=50):
        # Variables for recording
        self.is_recording = False
        self.seq_len = seq_len
        
        #Connecting to Amazon Polly for speech synthesis
        self.defaultRegion = 'us-east-1'
        self.defaultUrl = 'https://polly.us-east-1.amazonaws.com'
        self.polly = self.connectToPolly()

        # List of results stored each frame
        self.recorded_results = []
        self.predicted_sign = ''

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_signs = reference_signs

    def record(self):
        """
        Initialize sign_distances & start recording
        """
        self.reference_signs["distance"].values[:] = 0
        self.is_recording = True

    def process_results(self, results) -> (str, bool):
        """
        If the SignRecorder is in the recording state:
            it stores the landmarks during seq_len frames and then computes the sign distances
        :param results: mediapipe output
        :return: Return the word predicted (blank text if there is no distances)
                & the recording state
        """
        if self.is_recording:
            if len(self.recorded_results) < self.seq_len:
                self.recorded_results.append(results)
            else:
                self.compute_distances()
                print(self.reference_signs)

        if np.sum(self.reference_signs["distance"].values) == 0:
            return "", self.is_recording
        return self._get_sign_predicted(), self.is_recording

    def compute_distances(self):
        """
        Updates the distance column of the reference_signs
        and resets recording variables
        """
        left_hand_list, right_hand_list = [], []
        for results in self.recorded_results:
            _, left_hand, right_hand = extract_landmarks(results)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        # Create a SignModel object with the landmarks gathered during recording
        recorded_sign = SignModel(left_hand_list, right_hand_list)

        # Compute sign similarity with DTW (ascending order)
        self.reference_signs = dtw_distances(recorded_sign, self.reference_signs)

        # Reset variables
        self.recorded_results = []
        self.is_recording = False

    def _get_sign_predicted(self, batch_size=5, threshold=0.5):
        """
        Method that outputs the sign that appears the most in the list of closest
        reference signs, only if its proportion within the batch is greater than the threshold

        :param batch_size: Size of the batch of reference signs that will be compared to the recorded sign
        :param threshold: If the proportion of the most represented sign in the batch is greater than threshold,
                        we output the sign_name
                          If not,
                        we output "Sign not found"
        :return: The name of the predicted sign
        """
        # Get the list (of size batch_size) of the most similar reference signs
        sign_names = self.reference_signs.iloc[:batch_size]["name"].values

        # Count the occurrences of each sign and sort them by descending order
        sign_counter = Counter(sign_names).most_common()

        predicted_sign, count = sign_counter[0]
        if count / batch_size < threshold:
           
            return "Unknown Sign"

        #Check if the predicted sign is equal to the recorded sign
        if predicted_sign != self.predicted_sign:
            self.predicted_sign = predicted_sign
            phrase = self.get_phrases(predicted_sign)      
            print(phrase)
            self.speak(self.polly, phrase)

        return predicted_sign

    def connectToPolly(self):
        return boto3.client('polly', region_name=self.defaultRegion, endpoint_url=self.defaultUrl)

    def speak(self,polly, text, format='mp3', voice='Lupe'):
        response =self.polly.synthesize_speech(Text=text, OutputFormat=format, VoiceId=voice,Engine = 'neural')
        if "AudioStream" in response:
            # Note: Closing the stream is important because the service throttles on the
            # number of parallel connections. Here we are using contextlib.closing to
            # ensure the close method of the stream object will be called automatically
            # at the end of the with statement's scope.
                with closing(response["AudioStream"]) as stream:
                    output = os.path.join(gettempdir(), "speech.mp3")                    
                    try:
                        # Open a file for writing the output as a binary stream
                            with open(output, "wb") as file:
                                file.write(stream.read())
                    except IOError as error:
                        # Could not write to file, exit gracefully
                        print(error)
                        sys.exit(-1)
        else:
            # The response didn't contain audio data, exit gracefully
            print("Could not stream audio")
            sys.exit(-1)

        # Play the audio using the platform's default player
        if sys.platform == "win32":
            playsound.playsound(output)
            # Removing the temporary audio file
            os.remove(output)
            #os.startfile(output)
        else:
            # The following works on macOS and Linux. (Darwin = mac, xdg-open = linux).
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, output])    

    def get_phrases(self,word):
        # Create a list which contains a key value pair for each sign
        # for example
        # [ 'Sign' : 'Name', 'Phrase' : 'My name is' ]
        phrases = [
            {
                'Sign': 'Nombre',
                'Phrase': 'Mi nombre es'
            },
        ]

        # Iterate through the list and return the phrase associated with the sign
        for phrase in phrases:
            if phrase['Sign'] == word:
                return phrase['Phrase']

        return word
