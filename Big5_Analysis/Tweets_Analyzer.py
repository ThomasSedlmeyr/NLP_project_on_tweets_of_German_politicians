from itertools import groupby
from operator import itemgetter
import numpy as np
import timestring   

#reads all translated tweets
#returns list of tuples with (time, name, party and text) for every tweet
def readTweets():
    tweets = []
    partyOfPolitician = ""
    counter = 0
    with open('Big5_Analysis/Data_Generation/Name_Party_Time_Tweet_translated_complete.txt') as f:
        line = f.readline()
        while line:
            #if there is a new politician
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

#loads all translated tweets
#predicts out for them with given model
#saves result as numpy array containing
#(time, name, party, model prediction)
def evaluateAllTweets(model):
    tweets = readTweets()
    tweetsText = [x[3] for x in tweets]
    print("Prediction startet")
    outputs = model.predict(tweetsText)
    print("Prediction finished")
    tweetsWithOutputs = []
    for i in range(0, len(outputs)):
        tweetsWithOutputs.append((tweets[i][0], tweets[i][1], tweets[i][2], outputs[i]))
    np.save('Big5_Analysis/tweetsWithOutput.npy', np.array(tweetsWithOutputs))
    return tweetsWithOutputs

#groups given numpy array by parties
def groupResultsByParties(tweetsWithOutput):
    tweetsList = list(tweetsWithOutput)
    tweetsList.sort(key=itemgetter(2))
    groupedListByParty = [list(group) for key, group in groupby(tweetsList, itemgetter(2))]
    return groupedListByParty

def groupResultsByPoliticians(tweetsWithOutput):
    tweetsWithOutput.sort(key=itemgetter(1))
    groupedListByPolitician = [list(group) for key, group in groupby(tweetsWithOutput, itemgetter(1))]
    return groupedListByPolitician

#returns model predictions for all tweets grouped by parties
def big5valuesPerParties(tweetsWithOutput):
    #sort by party name, is necessary for grouping later
    groupedListByParty = groupResultsByParties(tweetsWithOutput)
    result = []
    parties = []
    #group by party name
    for outputsOfOneParty in groupedListByParty:
        values = [x[3] for x in outputsOfOneParty]
        parties.append(outputsOfOneParty[0][2])
        result.append(values)
    return (parties, result)

#returns average model prediction grouped by parties
def big5percentagePerParty(tweetsWithOutput):
    groupedListByParty = groupResultsByParties(tweetsWithOutput)
    resultList = []
    for outputsOfOneParty in groupedListByParty:
        values = [x[3] for x in outputsOfOneParty]
        sumList = np.sum(values, axis = 0)
        percentage = sumList / len(outputsOfOneParty)
        resultOfOneParty = (outputsOfOneParty[0][2], percentage) 
        resultList.append(resultOfOneParty)
    #print("['CDU', 'LINKE', 'FDP', 'GRÜNE','SPD', 'CSU', 'AFD']")
    #The order of big5 traits is:
    #Extraversion,Neuroticism,Agreeableness,Conscientiousness,Openness
    #print(resultList)
    return resultList
    

#returns average model prediction grouped by politicians
#only uses politicians with more tweets than threshhold
def big5percentagePerPolitician(tweetsWithOutput, threshhold):
    groupedListByPolitician = groupResultsByPoliticians(tweetsWithOutput)
    resultList = []
    for outputsOfOnePolitician in groupedListByPolitician:
        if len(outputsOfOnePolitician) > threshhold:
            values = [x[3] for x in outputsOfOnePolitician]
            sumList = np.sum(values, axis = 0)
            percentage = sumList / len(outputsOfOnePolitician)
            resultOfOnePolitician = (outputsOfOnePolitician[0][1], percentage) 
            resultList.append(resultOfOnePolitician)
    #print("['CDU', 'LINKE', 'FDP', 'GRÜNE','SPD', 'CSU', 'AFD']")
    #print(resultList)
    return resultList
    