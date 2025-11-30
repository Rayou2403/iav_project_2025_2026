Projet : Speech Emotion Recognition

1. Préprocessing audio
- Chargement des fichiers en mono
- Resampling en 16 kHz
- Normalisation
- Suppression des silences
- Génération d’augmentations (pitch, bruit, time-stretch, shift)

2. Extraction des features
MelSpectrogram :
- Représentation temps/fréquence
- Conserve le timbre, l’intensité et les harmoniques

MFCC :
- Version compressée du spectrogramme
- 40 coefficients
- Moins d’informations utiles pour l’émotion

3. Dataset
- Chargement des fichiers .npy (MFCC ou MelSpec)
- Attribution des labels d’émotion
- Pour MFCC : padding nécessaire car les durées varient

4. Modèle CNN (MelSpectrogram)
- Deux convolutions 2D + max-pooling
- Modèle simple utilisé comme baseline
- La perte descend rapidement

5. Modèle CNN + LSTM (MFCC)
- Convolution 1D suivie d’une LSTM
- Nécessite du padding
- Apprentissage plus lent que le CNN 2D

6. Observations
- Le CNN 2D sur MelSpectrogram fonctionne mieux
- Le CNN+LSTM sur MFCC progresse mais reste moins performant
- Les MFCC contiennent moins d’informations que les MelSpectrograms

Remarque: 
    MFCC :
    - Représentation compacte du signal
    - On calcule un spectrogramme, puis on applique une transformée en cosinus
    - Produit environ 40 coefficients par fenêtre
    - Conserve la forme globale du spectre
    - Perte des détails comme les harmoniques et le timbre
    - Très utilisé en reconnaissance de parole, moins précis pour l’émotion

    MelSpectrogram :
    - Représentation temps/fréquence du signal
    - Basé sur l’échelle Mel (proche de la perception humaine)
    - Conserve le timbre, l’intensité et les harmoniques
    - Apparence similaire à une image (temps × fréquences)
    - Plus riche en détails importants pour l’émotion
