from itertools import groupby
from operator import itemgetter, ne
import numpy as np
import timestring   
import seaborn as sns
import Tweets_Analyzer as ta
import matplotlib.pyplot as plt

#loads numpy array with tuples: 
#(time, name, party, model prediction)
def readTweets():
    return np.load('Big5_Analysis/tweetsWithOutput.npy', allow_pickle=True)

#
def boxPlot(title, path, xData, yData, palette, rotate): 
    ax = sns.boxplot(x=[1,2,3], y=yData)
    if rotate:
        plt.xticks(rotation=90, fontsize=12)
    if palette != None:
        sns.set_palette(palette)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()

#creates barplot for given x and y data and saves it in path
def plot(title, path, xData, yData, palette, rotate):
    ax = sns.barplot(x=xData, y=yData)
    if rotate:
        #rotate x labels by 90 degrees
        plt.xticks(rotation=90, fontsize=12)
    if palette != None:
        #change color of bars to match given palette
        sns.set_palette(palette)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()


#returns only pairs of label and value where the value differs more than threshhold from the mean
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


#creates barplots for each of the big5 categories for politicians predicted mean values are outliers
def visualizeEachPolitician(tweetsData):
    tweetsData = list(tweetsData)
    big5percentagePerParty = ta.big5percentagePerPolitician(tweetsData, 500)
    extraversion = []
    agreeableness = []
    openness = []
    conscientiousness = []
    neuroticism = []
    politicianNames = []

    #The order of big5 traits is:
    #Extraversion,Neuroticism,Agreeableness,Conscientiousness,Openness
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


#creates barplots for each of the big5 categories for the mean values of every party
def visualizeBig5PerParty(tweetsData):
    big5percentagePerParty = ta.big5percentagePerParty(tweetsData)
    extraversion = []
    agreeableness = []
    openness = []
    conscientiousness = []
    neuroticism = []
    parties = []
    for entry in big5percentagePerParty:
        extraversion.append(entry[1][0])
        agreeableness.append(entry[1][2])
        openness.append(entry[1][4])
        conscientiousness.append(entry[1][3])
        neuroticism.append(entry[1][1])
        parties.append(entry[0])
        
    #use colors to match the parties
    colors = ["blue", "black", "black", "yellow", "green", "violet", "red"] 
    customPalette = sns.color_palette(colors)
    sns.set_theme(style="whitegrid")
    plot("Extraversion", "Big5_Analysis/Visualized_data/Parties_Extraversion.png", parties, extraversion, customPalette, False)
    plot("Extraversion", "Big5_Analysis/Visualized_data/Parties_Extraversion.png", parties, extraversion, customPalette, False)
    plot("Agreeableness", "Big5_Analysis/Visualized_data/Parties_Agreeableness.png", parties, agreeableness, customPalette, False)
    plot("Openness", "Big5_Analysis/Visualized_data/Parties_Openness.png", parties, openness, customPalette, False)
    plot("Conscientiousness", "Big5_Analysis/Visualized_data/Parties_Conscientiousness.png", parties, conscientiousness, customPalette, False)
    plot("Neuroticism", "Big5_Analysis/Visualized_data/Parties_Neuroticism.png", parties, neuroticism, customPalette, False)

def visualizeBig5PerPartyBoxPlot(tweetsData):
    (parties, big5valuesPerParty) = ta.big5valuesPerParty(tweetsData)
    extraversion = []
    agreeableness = []
    openness = []
    conscientiousness = []
    neuroticism = []
        #,cEXT,cNEU,cAGR,cCON,cOPN
    for i in range(0, len(big5valuesPerParty)):
        extraversion.append([])
        agreeableness.append([])
        openness.append([])
        neuroticism.append([])
        conscientiousness.append([])
        for j in range(0, len(big5valuesPerParty[i])):
            extraversion[i].append(big5valuesPerParty[i][j][0])
            agreeableness.append(big5valuesPerParty[i][j][2])
            openness.append(big5valuesPerParty[i][j][4])
            conscientiousness.append(big5valuesPerParty[i][j][3])
            neuroticism.append(big5valuesPerParty[i][j][1])

    colors = ["blue", "black", "black", "yellow", "green", "violet", "red"] 
    customPalette = sns.color_palette(colors)
    sns.set_theme(style="whitegrid")
    boxPlot("Extraversion", "Big5_Analysis/Visualized_data/Parties_Extraversion_AfD.png", parties[0], extraversion[0], customPalette, False)
    boxPlot("Extraversion", "Big5_Analysis/Visualized_data/Parties_Extraversion.png", parties, extraversion, customPalette, False)
    boxPlot("Agreeableness", "Big5_Analysis/Visualized_data/Parties_Agreeableness.png", parties, agreeableness, customPalette, False)
    boxPlot("Openness", "Big5_Analysis/Visualized_data/Parties_Openness.png", parties, openness, customPalette, False)
    boxPlot("Conscientiousness", "Big5_Analysis/Visualized_data/Parties_Conscientiousness.png", parties, conscientiousness, customPalette, False)
    boxPlot("Neuroticism", "Big5_Analysis/Visualized_data/Parties_Neuroticism.png", parties, neuroticism, customPalette, False)

def calculateStdDeviation(tweetsData):
    (parties, valuesPerParty) = ta.big5valuesPerParty(tweetsData)
    for i in range(0, len(valuesPerParty)):
        extr = np.transpose(valuesPerParty[i])
        std_dev = np.std(extr[4])
        print(parties[i] + " " + str(std_dev))


tweetsData = readTweets()
visualizeEachPolitician(tweetsData)
visualizeBig5PerParty(tweetsData)


    


