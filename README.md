This project is about the analysis of tweets from politicians who are members of the German Bundestag.

As a first step, we created a database containing more than 240k tweets by over 500 German politicians from different parties.

We then trained a Keras model, which uses a political tweet as input and predicts to which party (CDU, LINKE, FDP, GRÜNE, SPD, CSU, AfD) the creator of the tweet belongs to. We reached an accuracy of more than 55% on our dataset using this model.

In another step we carried out an automated Big5 personality traits analysis of the politicians. For that purpose we first trained a BERT model on the essays dataset by Pennebaker and King, which we then used to evaluate the personalities of the politicians based on their tweets in our dataset.

Since this dataset is in English and our tweets are in German, we first had to translate them to English. Because these translated texts could contain less information, we had to check if the Big5 analysis is possible on machine translated text. We found out that using google translate for translating the texts is possible and even improved the model accuracy for the dataset. We used this improved model to analyse all the tweets of german politicians we collected and evaluated the results.
