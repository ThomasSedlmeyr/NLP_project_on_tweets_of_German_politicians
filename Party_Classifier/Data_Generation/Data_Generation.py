from os import listdir
import numpy as np
import csv
from operator import itemgetter

#This module contains all the methods which were use to create the traindataset 

#Creates a dictionary which returns for every politician its party
def createDictionaryPoliticiansToParty():
    accountsNamesParties = readMdBcsvFile()    
    keys = []
    values = []
    for data in accountsNamesParties:
        keys.append(data[0])
        values.append(data[2])
    dictionary = dict(zip(keys, values))
    return dictionary

#Generates a list with triples containing the name of the twitter account, 
#the persons name and the political party
def readMdBcsvFile():
    with open('Party_Classifier/Data_Generation/MdB.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    accountsNamesParties = []
    for line in data:        
        indexOfSpace = line[1].rindex(" ")
        name = line[1][0:indexOfSpace]
        party = line[1][indexOfSpace+1:]        
        accountsNamesParties.append((line[0], name, party))
        
    return accountsNamesParties

#Generates a numpy array which is later used for training the network
#The returned numpy array contains a list of Tupels where each comment has the specific label of the party
def generateNumpyArrayForTraining():
    csvData = readAllCSVfilesOfAllPoliticans()
    numpyList = []
    #tweetTimesAndNames is a set which is used to store unique keys for eachTweet. This is used because
    #the data contains also tweets which were retweeted. These tweets will removed in the data.  
    #Also we made two runs to get our data from twitter. This is the reason, why we got much duplicates, which
    #will be removed
    tweetTimesAndNames = set()
    counter = 0
    for tweetsOfOnePolitician in csvData:
        counter += 1
        for tweetData in tweetsOfOnePolitician:
            #trainData list of length 2 containing tweet text and the coresponding number of the party           
            name = tweetData[0] 
            time = tweetData[2] 
            timeAndName = name + time
            if(not (timeAndName in tweetTimesAndNames)):    #removes duplicates             
                trainData = [] 
                trainData.append(tweetData[3])
                party = dictPoliticianToParty.get(tweetData[len(tweetData)-1])
                trainData.append(partyToArray(party))
                numpyList.append(trainData)
                tweetTimesAndNames.add(timeAndName)
         
    numpyArray = np.array(numpyList)
    np.random.shuffle(numpyArray)
    #We use 80% for training and 15% for testing
    trainIndex = int(0.8 * len(numpyArray))
    #We use five percent for validation
    testIndex = int(0.95 * len(numpyArray))    

    train = numpyArray[:trainIndex]
    test = numpyArray[trainIndex:testIndex]
    val = numpyArray[testIndex:]

    train = np.transpose(train)
    test = np.transpose(test)
    val = np.transpose(val)

    np.save("Train", train)
    np.save("Test", test)
    np.save("Val", val)
    np.save("TweetAndParty", np.transpose(numpyArray))


#returns a list with all twitter accounts
def getTwitterAccountNames():
    accountPartyNames = readMdBcsvFile()
    twitterAccounts = []
    for (account,_, _) in accountPartyNames:
        twitterAccounts.append(account)
    return twitterAccounts

#returns a list containing the content of all Small csv files of every politician 
def readAllCSVfilesOfAllPoliticans():
    resultData = []
    #Reading the data of the first run
    folder = "Party_Classifier/Data_Generation/AllTweets_1/"
    fileNames = listdir(folder)
    for fileName in fileNames:        
        data = readCSVfileOfOnePolitician(fileName, folder)        
        #The name of the politician is added to the data        
        resultData.append(data)

    #Reading the data of the second run
    folder = "Party_Classifier/Data_Generation/AllTweets_2/"
    fileNames = listdir(folder)
    for fileName in fileNames:        
         data = readCSVfileOfOnePolitician(fileName, folder)        
         #The name of the politician is added to the data        
         resultData.append(data)    
    return resultData

#reads the csv-file of a certain politician 
def readCSVfileOfOnePolitician(fileName, folder):
    nameOfPolitician = fileName[0:len(fileName)-4] 
    filePath = folder+fileName
    resultData = []
    
    with open(filePath, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        
        for d in data:            
            d.append(nameOfPolitician)
            resultData.append(d)
            
    return resultData[1:]

#creates a dictionary which maps every party to a different unique number
def createPartyToNumberDict():
    parties = list(dict.fromkeys(dictPoliticianToParty.values()))
    keys = []
    values = range(0,len(parties))
    for party in parties:
        keys.append(party)
        print(party)
    return dict(zip(keys,values))

#creates one-hot encoding for a given party using the partyToNumber dictionary
def partyToArray(party):
    resultArray = np.zeros(len(dictPartyToNumber.keys()))
    
    resultArray[dictPartyToNumber.get(party)] = 1
    return resultArray

#returns the index which is set to 1 of the one-hot encoding      
def getIndexOfParty(array):
    for i in range(0,len(array)):
        if(array[i] == 1):
            return i
    return -1    

#counts the number of tweets per party and prints the result on the console
def showTweetcountPerParty(parties, data):
    data = np.transpose(data)
    counts = np.zeros(len(parties))
 
    for d in data:
        counts[getIndexOfParty(d[1])] += 1
    for i in range(0, len(parties)):
        print(parties[i] + " " + str(counts[i]))

#Saves the data of the LKR party
def saveDataOfLKRparty():
    resultData = []
    data = readCSVfileOfOnePolitician("MieruchMario.csv")        
    for tweetData in data:
        resultData.append(tweetData[3])

    numpyArray = np.array(resultData)
    np.save("LKR", numpyArray)

#Generates a txt-file containing the name of a politician and all tweets with the time when they were sent
def generateFileWithNamePartyTimeAndTweet():
    csvData = readAllCSVfilesOfAllPoliticans()
    nameTimeTweetList = []
    #tweetTimesAndNames is a set which is used to store unique keys for eachTweet. This is used because
    #the data contains also tweets which were retweeted. These tweets will removed in the data.  
    #Also we made two runs to get our data from twitter. This is the reason, why we got much duplicates, which
    #will be removed
    tweetTimesAndNames = set()
    counter = 0
    for tweetsOfOnePolitician in csvData:
        counter += 1
        for tweetData in tweetsOfOnePolitician:
            twitterName = tweetData[7] 
            name = dictTwitterAccountToRealName.get(twitterName)
            time = tweetData[2]
            party = dictPoliticianToParty.get(twitterName)
            if party =="GRï¿½NE":
                party = "GRÜN"
            timeAndName = twitterName + time
            if(not (timeAndName in tweetTimesAndNames)):    #removes duplicates      
                tweet = tweetData[3]  
                #All tweets should have a mimimum size of 8
                if len(tweet) >= 8: 
                    tweet = cleanUpTweet(tweet)              
                    nameTimeTweetList.append((name, party, time, tweet))     
          
    #nameTimeTweetList = sorted(nameTimeTweetList, key=lambda tup: tup[0])
    #nameTimeTweetList.sort(key=lambda 
    # tup: tup[0])
    nameTimeTweetList = [x for x in nameTimeTweetList if x[0] is not None]
    nameTimeTweetList.sort(key=itemgetter(0))
    lastName = ""
    with open('Party_Classifier/Data_Generation/NamePartyTimeTweet.txt', 'w', encoding='utf-8') as f:
        for data in nameTimeTweetList:
            (name, party, time, tweet) = data
            if name != lastName:
                f.write("----------")
                f.write("\n")
                f.write(name)
                f.write('\n')
                f.write(party)
                f.write('\n')
                lastName = name

            f.write(time + " " + tweet)
            f.write('\n')


def createDictionaryTwitterAcountNameToRealName():
    accountsNamesParties = readMdBcsvFile()    
    keys = []
    values = []
    for (twitterNames, realNames, _) in accountsNamesParties:
        keys.append(twitterNames)
        values.append(realNames)

    dictionary = dict(zip(keys, values))
    return dictionary

def cleanUpTweet(tweet):
    #Remove \n \r
    #tweet = tweet.rstrip()
    tweet = tweet.replace('\r', '')
    tweet = tweet.replace('\n', '')
    tweet = tweet.replace('\t', '')
    #Remove all characters after @ till next " "
    #indexAt = tweet.index("@")
    #indexSpace = tweet.index(" ", indexAt+1)
    #firstPart = tweet[:indexAt]
    #secondPart = tweet[indexSpace:]
    #tweet = firstPart + secondPart
    return tweet   


#Global dictionaries which were used in several functions. We used these dictionaries to 
#increase the performance 
tweetIdToParty = {}
dictPoliticianToParty = createDictionaryPoliticiansToParty()
dictPartyToNumber = createPartyToNumberDict()
dictTwitterAccountToRealName = createDictionaryTwitterAcountNameToRealName()

generateFileWithNamePartyTimeAndTweet()
print("Fertig")