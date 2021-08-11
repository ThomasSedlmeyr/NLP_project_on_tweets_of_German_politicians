from itertools import groupby
from operator import itemgetter
import numpy as np
import timestring   

def readTweets():
    tweets = []
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
                if counter == 10000000:
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
    #print(tweetsText)
    print("Prediction startet")
    outputs = model.predict(tweetsText)
    print("Prediction finished")
    tweetsWithOutputs = []
    for i in range(0, len(outputs)):
        tweetsWithOutputs.append((tweets[i][0], tweets[i][1], tweets[i][2], outputs[i]))
    np.save('Big5_Analysis/tweetsWithOutput.npy', np.array(tweetsWithOutputs))
    return tweetsWithOutputs


def big5percentagePerParty(tweetsWithOutput):
    tweetsList = list(tweetsWithOutput)
    tweetsList.sort(key=itemgetter(2))
    groupedListByParty = [list(group) for key, group in groupby(tweetsList, itemgetter(2))]
    #it = groupby(tweetsWithOutput, itemgetter(2))
    #it = list(it)
    resultList = []
    for outputsOfOneParty in groupedListByParty:
        values = [x[3] for x in outputsOfOneParty]
        sumList = np.sum(values, axis = 0)
        percentage = sumList / len(outputsOfOneParty)
        resultOfOneParty = (outputsOfOneParty[0][2], percentage) 
        resultList.append(resultOfOneParty)
    #print("['CDU', 'LINKE', 'FDP', 'GRÜNE','SPD', 'CSU', 'AFD']")
    #,cEXT,cNEU,cAGR,cCON,cOPN
    print(resultList)
    return resultList
    
def big5percentagePerPolitician(tweetsWithOutput, threshhold):
    #sort by name of the politician
    tweetsWithOutput.sort(key=itemgetter(1))
    groupedListByPolitician = [list(group) for key, group in groupby(tweetsWithOutput, itemgetter(1))]
    #it = groupby(tweetsWithOutput, itemgetter(2))
    #it = list(it)
    resultList = []
    for outputsOfOnePolitician in groupedListByPolitician:
        if len(outputsOfOnePolitician) > threshhold:
            values = [x[3] for x in outputsOfOnePolitician]
            sumList = np.sum(values, axis = 0)
            percentage = sumList / len(outputsOfOnePolitician)
            resultOfOnePolitician = (outputsOfOnePolitician[0][1], percentage) 
            resultList.append(resultOfOnePolitician)
    #print("['CDU', 'LINKE', 'FDP', 'GRÜNE','SPD', 'CSU', 'AFD']")
    #,cEXT,cNEU,cAGR,cCON,cOPN
    print(resultList)
    return resultList


    
#tweets = readTweets()
#evaluateAllTweets(None)
#print("Finished")