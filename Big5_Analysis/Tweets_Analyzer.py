from itertools import groupby
from operator import itemgetter
import numpy as np
import timestring   

def readTweets():
    tweets = []
    politician = ""
    partyOfPolitician = ""
    counter = 0
    with open('Big5_Analysis/Data_Generation/Name_Party_Time_Tweet_translated_complete.txt') as f:
        line = f.readline()
        while line:
            #if a new politician 
            if(line.find("------") == 0):
                line = f.readline()
                politicianName = line[:-1]
                line = f.readline()
                partyOfPolitician = line[:-1]
            line = f.readline()
            parsedLine = parseOneLine(line[:-1])
            if parsedLine is not None:
                (time, tweetText) = parsedLine
                tweets.append((time, politicianName, partyOfPolitician, tweetText))
                counter += 1
                if counter == 10000:
                    break
    return tweets

def parseOneLine(line):
    indexOfSpace = line.find(" ")
    if indexOfSpace == -1:
        return None
    try: 
        timeText = line[:indexOfSpace]
        tweetText = line[indexOfSpace+1:]

        time = timestring.Date(timeText)
    except:
        return None

    return (time, tweetText)

def evaluateAllTweets(model):
    tweets = readTweets()
    tweetsText = [x[3] for x in tweets]
    print(tweetsText)
    outputs = model.predict(tweetsText)
    tweetsWithOutput = zip(tweets, outputs)
    return tweetsWithOutput

def big5percentagePerparty(tweetsWithOutput):
    it = itertools.groupby(tweetsWithOutput, operator.itemgetter(2))
    resultList = []
    for outputsOfOneParty in it:
        sum = [x[4] for x in outputsOfOneParty]
        percentage = np.sum(sum) / len(outputsOfOneParty)
        resultOfOneParty = (outputsOfOneParty[2], percentage) 
        resultList.append(resultOfOneParty)
    print("['CDU', 'LINKE', 'FDP', 'GRÜNE','SPD', 'CSU', 'AFD']")
    print(resultList)
    return resultList


#tweets = readTweets()
#evaluateAllTweets(None)
#print("Finished")


    


