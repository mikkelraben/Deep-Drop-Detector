import librosa

y, sr = librosa.load("wav/Freaks Have More Fun - Loudpvck Remix - Dada Life.wav", sr=22050, mono=True)
mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128))

print(mel.shape)  # (128, ~43 × song length in seconds)