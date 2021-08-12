import csv
import Data_Generation.Data_Generation as dg
import datetime
from time import sleep
from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.common import exceptions
from os import listdir

#adapted from: https://youtu.be/3KaffTIZ5II and https://github.com/israel-dryer/Twitter-Scraper

#returns the appropriate webdriver, which is chrome in our case
def create_webdriver_instance():
    driver = Chrome()
    return driver

#tries to open the given url
def gotoUrl(url, driver):
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(expected_conditions.url_to_be(url))
    except exceptions.TimeoutException:
        print("Timeout while waiting for website")
    return True

#returns the time of the tweet which we use as an unique identifier
def generate_tweet_id(tweet):
    return tweet[2]

def scroll_down_page(driver, last_position, scroll_attempt, num_seconds_to_load=0.2, max_attempts=10):
    """The function will try to scroll down the page and will check the current
    and last positions as an indicator. If the current and last positions are the same after `max_attempts`
    the assumption is that the end of the scroll region has been reached and the `end_of_scroll_region`
    flag will be returned as `True`"""
    end_of_scroll_region = False
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    sleep(num_seconds_to_load)
    curr_position = driver.execute_script("return window.pageYOffset;")
    if curr_position == last_position:
        if scroll_attempt == max_attempts:
            end_of_scroll_region = True
        else:
            return scroll_down_page(driver, curr_position, scroll_attempt + 1)
    last_position = curr_position
    #print(end_of_scroll_region)
    return last_position, end_of_scroll_region

#creates header for the file if the mode argument is 'w'
#appends collected data to the given file otherwise
def save_tweet_data_to_csv(records, filepath, mode='a+'):
    header = ['User', 'Handle', 'PostDate', 'TweetText', 'ReplyCount', 'RetweetCount', 'LikeCount']
    with open(filepath, mode=mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(header)
        if records:
            writer.writerow(records)

#returns all tweets that are currently loaded as WebElements
def collect_all_tweets_from_current_view(driver, lookback_limit=100000):
    """The page is continously loaded, so as you scroll down the number of tweets returned by this function will
     continue to grow. To limit the risk of 're-processing' the same tweet over and over again, you can set the
     `lookback_limit` to only process the last `x` number of tweets extracted from the page in each iteration.
     You may need to play around with this number to get something that works for you. I've set the default
     based on my computer settings and internet speed, etc..."""
    page_cards = driver.find_elements_by_xpath('//div[@data-testid="tweet"]')
    print(len(page_cards))
    if len(page_cards) <= lookback_limit:
        return page_cards
    else:
        return page_cards[-lookback_limit:]

#extracts the relevant information our of a single WebElement and returns it as tuple
def extract_data_from_current_tweet_card(card):
    try:
        user = card.find_element_by_xpath('.//span').text
    except exceptions.NoSuchElementException:
        user = ""
    except exceptions.StaleElementReferenceException:
        return
    try:
        handle = card.find_element_by_xpath('.//span[contains(text(), "@")]').text
    except exceptions.NoSuchElementException:
        handle = ""
    try:
        """
        If there is no post date here, there it is usually sponsored content, or some
        other form of content where post dates do not apply. You can set a default value
        for the postdate on Exception if you which to keep this record. By default I am
        excluding these.
        """
        postdate = card.find_element_by_xpath('.//time').get_attribute('datetime')
    except exceptions.NoSuchElementException:
        return
    try:
        _comment = card.find_element_by_xpath('.//div[2]/div[2]/div[1]').text
    except exceptions.NoSuchElementException:
        _comment = ""
    try:
        _responding = card.find_element_by_xpath('.//div[2]/div[2]/div[2]').text
    except exceptions.NoSuchElementException:
        _responding = ""
    tweet_text = _comment + _responding
    try:
        reply_count = card.find_element_by_xpath('.//div[@data-testid="reply"]').text
    except exceptions.NoSuchElementException:
        reply_count = ""
    try:
        retweet_count = card.find_element_by_xpath('.//div[@data-testid="retweet"]').text
    except exceptions.NoSuchElementException:
        retweet_count = ""
    try:
        like_count = card.find_element_by_xpath('.//div[@data-testid="like"]').text
    except exceptions.NoSuchElementException:
        like_count = ""

    tweet = (user, handle, postdate, tweet_text, reply_count, retweet_count, like_count)
    return tweet

#opens the url for the given user
#creates a csv file for storing the tweets of the user
#extracts the data from the loaded view and saves it
#scrolls down the page to load new tweets
#repeats until it can't scroll any further
def collectTwitterDataForUser(user):
    url = 'https://twitter.com/' + user
    filepath = user + '.csv'
    #save_tweet_data_to_csv(None, filepath, 'w')  # create file for saving records
    last_position = None
    end_of_scroll_region = False
    unique_tweets = set()
    driver = create_webdriver_instance()
    pageLoaded = gotoUrl(url, driver)
    if not pageLoaded:
        return
    sleep(4)
    while not end_of_scroll_region:
        cards = collect_all_tweets_from_current_view(driver)
        for card in cards:
            try:
                tweet = extract_data_from_current_tweet_card(card)
            except exceptions.StaleElementReferenceException:
                continue
            if not tweet:
                continue
            tweet_id = generate_tweet_id(tweet)
            if tweet_id not in unique_tweets:
                unique_tweets.add(tweet_id)
                #save_tweet_data_to_csv(tweet, filepath)
            else:
                continue
            #indexOfTime = tweet[2].find("T")
            #dateArr = tweet[2][:indexOfTime].split("-")
            #date = datetime.datetime(int(dateArr[0]), int(dateArr[1]), int(dateArr[2]))
            #if(date < minDate):
                #return
        last_position, end_of_scroll_region = scroll_down_page(driver, last_position, 0)
    driver.quit()


#collects the data and saves it for all users in our dataset
def collectTwitterData():
    userList = dg.getTwitterAccountNames()
    for user in userList:
        collectTwitterDataForUser(user)

collectTwitterData()