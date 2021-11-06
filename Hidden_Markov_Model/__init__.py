from hmmlearn.hmm import GMMHMM,GaussianHMM
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from hmmlearn.hmm import MultinomialHMM
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt  
from sklearn.metrics import silhouette_score
import math