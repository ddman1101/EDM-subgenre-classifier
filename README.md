# EDM-subgenre-classifier
This program can classify the EDM (Electronic Dance Music) subgenre.

## Prerequisites

* Python  == 3.6.13
* librosa == 0.8.1
* torch   == 1.3.0
* numpy   == 1.16.0
* pandas  == 1.1.2

## Usage

You could run
  <pre><code>pip3 install -r requirements.txt
</code></pre>
first to establish the enviroments.

Then run
  <pre><code>python3 main.py
  </code></pre>
could predict the song's genre directly.

* Step 1 : Preparing the audio (mp3, wav) and put it under "./data/audio/"
* Step 2 : Extracting the feature, and the feature will under the ./data/{feature_folder}"
* Step 3 : Classifying the audio by feature
* Step 4 : The result will be in the "./result.csv"
* All the "step" could complete by main.py
