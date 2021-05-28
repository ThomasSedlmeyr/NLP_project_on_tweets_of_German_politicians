from os import listdir
import numpy as np

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
    with open('MdB.csv', newline='',encoding='cp1252') as f:
        reader = csv.reader(f)
        data = list(reader)
        #print(data)
    
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
    for tweetsOfOnePolitician in csvData:   
        for tweetData in tweetsOfOnePolitician:
            #trainData list of length 2 containing tweet text and coresponding number of party
            trainData = [] 
            trainData.append(tweetData[3])
            party = dictPoliticianToParty.get(tweetData[len(tweetData)-1])
            trainData.append(dictPartyToNumber.get(party))
            numpyList.append(trainData)

    numpyArray = np.array(numpyList)
    numpyArray = np.transpose(numpyArray)
    np.save("TweetAndParty", numpyArray)

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
    fileNames = listdir("AlleTweets/")
    for fileName in fileNames:
        data = readCSVfileOfOnePolitician(fileName)        
        #The name of the politician is added to the data        
        resultData.append(data)
    return resultData

#reads the csv-file of a certain politician 
def readCSVfileOfOnePolitician(fileName):
    nameOfPolitician = fileName[0:len(fileName)-4] 
    filePath = "AlleTweets/"+fileName
    resultData = []
    
    with open(filePath, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        
        for d in data:            
            d.append(nameOfPolitician)
            resultData.append(d)
            
    return resultData[1:]

#creates dictionary that maps every party to a different number
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
        
def collectTwitterData():
    userList = getTwitterAccountNames()
    for user in userList:
        
        collectTwitterDataForUser(user)

def showTweetcountPerParty(parties, data):
    data = np.transpose(data)
    counts = np.zeros(len(parties))
 
    for d in data:
        counts[d[1]] += 1
    for i in range(0, len(parties)):
        print(parties[i] + " " + str(counts[i]))


dictPoliticianToParty = createDictionaryPoliticiansToParty()
dictPartyToNumber = createPartyToNumberDict()
generateNumpyArrayForTraining()

data = np.load("TweetAndParty.npy", allow_pickle=True)

print(dictPartyToNumber)
showTweetcountPerParty(list(dictPartyToNumber.keys()), data)