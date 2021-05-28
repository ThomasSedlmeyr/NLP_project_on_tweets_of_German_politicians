import Data_Generation

dictPoliticianToParty = createDictionaryPoliticiansToParty()
dictPartyToNumber = createPartyToNumberDict()
generateNumpyArrayForTraining()

data = np.load("TweetAndParty.npy", allow_pickle=True)

print(dictPartyToNumber)
showTweetcountPerParty(list(dictPartyToNumber.keys()), data)