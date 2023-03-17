import pyaudio
import wave
import pandas as pd
import os
import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score,f1_score
import pickle
import xgboost as xgb


sample_rate = 22050
RECORD_SECONDS = 5

def record():
    
    FORMAT = pyaudio.paInt16
    MEOW=0
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "RECORDS/recordedFile{}.wav"
    device_index = 2
    audio = pyaudio.PyAudio()

    print("----------------------record device list---------------------")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

    print("-------------------------------------------------------------")

    index = int(input())
    print("recording via index "+str(index))
    for _ in range(6):
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,input_device_index = index,
                        frames_per_buffer=CHUNK)
        print ("recording started")
        Recordframes = []
        
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        print ("recording stopped")
        stream.stop_stream()
        stream.close()
        MEOW=MEOW+1
        waveFile = wave.open(WAVE_OUTPUT_FILENAME.format(str(MEOW)), 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()
    audio.terminate()

# Function to load dataset and extracting information from the filename

def extract_features(directory):
    inc=0
    col=range(0,180)
    features = pd.DataFrame(columns=['feature'])
    for file in os.listdir(directory):
        
        feature_set = np.array([])
        audio_file, _ = librosa.load("RECORDS/"+file, sr=sample_rate, duration=RECORD_SECONDS)

        feature_set = np.array([])
        
        # MFCC feature extraction
        # No. of MFCC Features = 40 (Default = 20)
        mfccs=np.mean(librosa.feature.mfcc(y=audio_file, sr=sample_rate, n_mfcc=40).T, axis=0)
        feature_set=np.hstack((feature_set, mfccs))
        
        ## Chroma feature extraction
        # No. of Chroma Features = 12 (Always)
        stft=np.abs(librosa.stft(audio_file))
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        feature_set=np.hstack((feature_set, chroma))
        
        ## Mel feature extraction
        # No. of Mel Features = 128 (Default = 128)
        mel=np.mean(librosa.feature.melspectrogram(audio_file, sr=sample_rate).T,axis=0)
        feature_set=np.hstack((feature_set, mel))
        
        features.loc[inc] = [feature_set]
        inc=inc+1

    final_data = pd.DataFrame(features['feature'].values.tolist())
    return(final_data)

def scalar_and_pca(df):
    #print(df.shape)

    scalar = StandardScaler()
    scalar.fit(df)
    scaled_data_train = scalar.transform(df)
    #print(scaled_data_train)
    
    train_data_load = pd.read_csv('train_features.csv')
    scalar_trained = StandardScaler()
    scalar_trained.fit(train_data_load.iloc[:, :-2])
    scaled_data_trained_before = scalar_trained.transform(train_data_load.iloc[:, :-2])

    pca = PCA(n_components=78)
    pca.fit(scaled_data_trained_before)
    x_pca = pca.transform(scaled_data_train)
    return x_pca 


record()
scalar_and_pca(extract_features("RECORDS"))
model = pickle.load(open('mlp.model','rb'))
output = model.predict(scalar_and_pca(extract_features("RECORDS")))

'''----------------------output--------------------------'''

labelmap = {
 'male_angry':0,
 'female_surprised':1,
 'male_neutral':2,
 'male_surprised':3,
 'male_calm':4,
 'female_happy':5,
 'male_disgust':6,
 'female_fearful':7,
 'male_sad':8,
 'female_neutral':9,
 'female_angry':10,
 'female_calm':11,
 'female_sad':12,
 'male_fearful':13,
 'male_happy':14,
 'female_disgust':15,
}

song_names = {
	'angry' : ['Dream on', 'Believer', 'Demons'],
	'surprised' : ['Y dis kolaberi', 'heart stereo', 'montero'],
	'neutral' : ['Pasoori', 'Adaat', 'Those eyes'],
	'calm' : ['Boba Tunnel', 'Tu Jane Naa', 'Dusk till Dawn'],
	'sad' : ['Bekhayali', 'Channa Mereya', 'Sun raha hai na tu'],
	'happy' : ['Aaj din Shaynara', 'Mera mann', 'Die for you'],
	'disguist' : ['Tumpa', 'Tunir Maa', 'Subway Sexist'],
	'fearful' : ["Lullaby Of Woe", 'Goodbye', "Something in the way"]
}

book_names = {
	'angry': ["Mindful Anger: A Pathway to Emotional Freedom", "Ikigai"],
	'calm' : ['Bhagawad Geeta', 'Siddhartha'],
	'surprised' : ['The Alchemist', 'The Girl with a Dragon Book'],
	'happy' : ['Happiness project', 'Book of Joy'],
	'sad' : ['The Bell Jar', 'Reasons to stay alive'],
	'disguist' : ["American Psycho", "The Girl Next Door"],
	'neutral' : ["The Nightangle", "Born a Crime"],
	'fearful' : ["Gift of Fear", "The Haunting of Hill House"]
}

obtained = [list(labelmap)[o] for o in output]

# print(obtained)

# obtained = ['female_angry', 'female_sad', 'male_angry', 'male_angry', 'male_disguist', 'female_calm']

sex = []
emotion = []

for x in obtained:
	s, e = x.split('_')
	sex.append(s)
	emotion.append(e)

print("Predictions of Highest Probability:")

print("\tsex: ", max(sex,key=sex.count))
print("\tEmotion: ", max(emotion,key=emotion.count))

print("\tReccomended Songs: ", *song_names[max(emotion,key=emotion.count)], sep="\n\t\t")
print("\tReccomended Books: ", *book_names[max(emotion,key=emotion.count)], sep="\n\t\t")
