from itertools import groupby
from operator import itemgetter, ne
import numpy as np
import timestring   
import seaborn as sns
import Tweets_Analyzer as ta
import matplotlib.pyplot as plt

def readTweets():
    return np.load('Big5_Analysis/tweetsWithOutput.npy', allow_pickle=True)

def plot(title, path, xData, yData, palette, rotate):
    ax = sns.barplot(x=xData, y=yData)
    if rotate:
        plt.xticks(rotation=90, fontsize=12)
    if palette != None:
        sns.set_palette(palette)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()


def calculateOutliers(values, labels, threshhold):
    average = sum(values) / len(values)
    resultValues = []
    resultLabels = []
    for i in range(len(values)):
        v = values[i]
        if abs(average - v) > threshhold:
            resultValues.append(v)
            resultLabels.append(labels[i])
    return (resultValues, resultLabels)


def visualizeEachPolitician(tweetsData):
    tweetsData = list(tweetsData)
    big5percentagePerParty = ta.big5percentagePerPolitician(tweetsData, 500)
    extraversion = []
    agreeableness = []
    openness = []
    conscientiousness = []
    neuroticism = []
    politicianNames = []


    #,cEXT,cNEU,cAGR,cCON,cOPN
    for entry in big5percentagePerParty:
        extraversion.append(entry[1][0])
        agreeableness.append(entry[1][2])
        openness.append(entry[1][4])
        conscientiousness.append(entry[1][3])
        neuroticism.append(entry[1][1])
        politicianNames.append(entry[0])

    extraversion, filteredNames = calculateOutliers(extraversion, politicianNames, 0.05)
    plot("Extraversion", "Big5_Analysis/Visualized_data/Politicians_Extraversion.png", filteredNames, extraversion, None, True)

    agreeableness, filteredNames = calculateOutliers(agreeableness, politicianNames, 0.05)
    plot("Agreeableness", "Big5_Analysis/Visualized_data/Politicians_Agreeableness.png", filteredNames, agreeableness, None, True)

    openness, filteredNames = calculateOutliers(openness, politicianNames, 0.05)
    plot("Openness", "Big5_Analysis/Visualized_data/Politicians_Openness.png", filteredNames, openness, None, True)

    conscientiousness, filteredNames = calculateOutliers(conscientiousness, politicianNames, 0.07)
    plot("Conscientiousness", "Big5_Analysis/Visualized_data/Politicians_Conscientiousness.png", filteredNames, conscientiousness, None, True)

    neuroticism, filteredNames = calculateOutliers(neuroticism, politicianNames, 0.08)
    plot("Neuroticism", "Big5_Analysis/Visualized_data/Politicians_Neuroticism.png", filteredNames, neuroticism, None, True)


def visualizeBig5PerParty(tweetsData):
    #extraversion (outgoing/energetic vs. solitary/reserved)
    #agreeableness (friendly/compassionate vs. critical/rational)
    #openness to experience (inventive/curious vs. consistent/cautious)
    #conscientiousness (efficient/organized vs. extravagant/careless)
    #neuroticism (sensitive/nervous vs. resilient/confident)
    big5percentagePerParty = ta.big5percentagePerParty(tweetsData)
    extraversion = []
    agreeableness = []
    openness = []
    conscientiousness = []
    neuroticism = []
    parties = []
        #,cEXT,cNEU,cAGR,cCON,cOPN
    for entry in big5percentagePerParty:
        extraversion.append(entry[1][0])
        agreeableness.append(entry[1][2])
        openness.append(entry[1][4])
        conscientiousness.append(entry[1][3])
        neuroticism.append(entry[1][1])
        parties.append(entry[0])
    colors = ["blue", "black", "black", "yellow", "green", "violet", "red"] 
    customPalette = sns.color_palette(colors)
    sns.set_theme(style="whitegrid")
    plot("Extraversion", "Big5_Analysis/Visualized_data/Parties_Extraversion.png", parties, extraversion, customPalette, False)
    plot("Extraversion", "Big5_Analysis/Visualized_data/Parties_Extraversion.png", parties, extraversion, customPalette, False)
    #plt.show()
    plot("Agreeableness", "Big5_Analysis/Visualized_data/Parties_Agreeableness.png", parties, agreeableness, customPalette, False)
    #plt.show()
    plot("Openness", "Big5_Analysis/Visualized_data/Parties_Openness.png", parties, openness, customPalette, False)
    #plt.show()
    plot("Conscientiousness", "Big5_Analysis/Visualized_data/Parties_Conscientiousness.png", parties, conscientiousness, customPalette, False)
    #plt.show()
    plot("Neuroticism", "Big5_Analysis/Visualized_data/Parties_Neuroticism.png", parties, neuroticism, customPalette, False)
    #plt.show()
    #sns.palplot(customPalette)
    #plt.show()

tweetsData = readTweets()
visualizeBig5PerParty(tweetsData)
visualizeEachPolitician(tweetsData)
#tweets = readTweets()
#evaluateAllTweets(None)
#print("Finished")


    


