import Data_Generation as dg
import numpy as np


dg.generateNumpyArrayForTraining()
dg.saveDataOfLKRparty()

data = np.load("TweetAndParty.npy", allow_pickle=True)

print(dg.dictPartyToNumber)
dg.showTweetcountPerParty(list(dg.dictPartyToNumber.keys()), data)