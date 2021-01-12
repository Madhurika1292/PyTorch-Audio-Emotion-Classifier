# PyTorch-Audio-Emotion-Classifier
Audio classification may be used to interpret audio scenario, which is critical in turn for an artificial entity to understand and communicate more efficiently with its environment.  We'll create a classifier of audio emotions

Multiple Data Sources:

1.	Surrey Audio-Visual Expressed Emotion (SAVEE) database [http://kahlan.eps.surrey.ac.uk/savee/], size – 110MB

2.	The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) [ https://zenodo.org/record/1188976] – 440MB

3.	Toronto emotional speech set (TESS) [https://tspace.library.utoronto.ca/handle/1807/24487] – 440MB

4.	Crowd-Sourced Emotional Multimodal Actors Dataset [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4313618/] – 460MB

These are the four different AV/Audio/Video-[speech/song] datasets labelled into seven or eight different emotions and categorized by gender. Our goal here is to build a model using neural networks in pytorch framework, which can generalize a random input into specified categories with best accuracy. There is a need to combine all four datasets to avoid overfitting and gender bias or age group bias.

The process includes:

1.	Exploratory analysis on single dataset created by generalizing emotions from all four datasets.

2.	Feature extraction using techniques Spectral Roll-off, Spectral centroid, MF-CC, etc.. and Tensor conversion.

3.	Data Augmentation to generate syntactic data using Noise Injection, Change in Pitch and Speed of original data.

4.	Building robust neural net model by applying normalization and autoencoding techniques.

