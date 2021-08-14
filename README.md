This project is about the analysis of tweets from politicians who are members of the German Bundestag.

As a first step, we created a database containing more than 200k tweets by over 500 German politicians from different parties.

We then trained a Keras model, which uses a political tweet as input and predicts to which party (CDU, LINKE, FDP, GRÜNE, SPD, CSU, AfD) the creator of the tweet belongs to. We reached an accuracy of more than 60% on our dataset using this model.

In another step we wanted to carry out an automated Big5 personality traits analysis of the politicians. Herefore we first trained a BERT model on the essays dataset by Pennebaker and King, which we then used to evaluate the personalities of the politicians based on their tweets in our dataset.
Since our dataset is in German, we will also experiment, whether such a Big5 personality traits analysis even works on texts which were translated by a machine translation algorithm. 

To test this, we implemented our own transformer model to translate from English to German texts. With our hardware setup however, it was not possible to train a model which satisfied our requirements.
We therefore decided to use the Google translate application to translate the essays of our Big5 test dataset into German and retranslate them into English. Using this, we will compare the outputs of the Big5 analysis on the twice translated texts with the original outputs.

We strongly expect that the model will also work on that translated data and we can use it for our own analysis.
As another step we will analyse the personality development of every politician in our dataset over time.
