#import Data_Generation as dg
import numpy as np


#dg.generateNumpyArrayForTraining()
#print("Numpy Arrays were created")
#dg.saveDataOfLKRparty()

#data = np.load("TweetAndParty.npy", allow_pickle=True)

#print(dg.dictPartyToNumber)
#dg.showTweetcountPerParty(list(dg.dictPartyToNumber.keys()), data)


#Reads the dataset of German and English Tedtalk sentences and saves the data as numpy array
def createEnglishGermanSentenceDataSet():
    dataSet = []
    germanSentence = ""
    englishSentence = ""
    isGerman = False
    counter = 1
    with open("Dataset_TED_English_German.txt.") as file:
        for line in file:
            #An empty line indicates a new pair of English and German sentences 
            if(line == "\n"):
                dataSet.append([englishSentence, germanSentence])
            elif(isGerman):
                germanSentence = line
                isGerman = False
            else: 
                englishSentence = line
                isGerman = True

    numpyArray = np.array(dataSet)
    #np.save("Dataset_TED_English_German", numpyArray)
    print("DataSet was created!")

createEnglishGermanSentenceDataSet()
#lkrTweets = np.load('Dataset_TED_English_German.npy', allow_pickle=True)
#print("Fertig")