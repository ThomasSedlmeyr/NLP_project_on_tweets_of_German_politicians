import csv
import datetime
from time import sleep
from selenium.webdriver import Chrome
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.common import exceptions

csvPath = "twitterData.csv"

def create_webdriver_instance():
    driver = Chrome()
    return driver

def gotoUrl(url, driver):
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(expected_conditions.url_to_be(url))
    except exceptions.TimeoutException:
        print("Timeout while waiting for website")
    return True


def find_search_input_and_enter_criteria(search_term, driver):
    xpath_search = '//input[@aria-label="Search query"]'
    search_input = driver.find_element_by_xpath(xpath_search)
    search_input.send_keys(search_term)
    search_input.send_keys(Keys.RETURN)
    return True


def change_page_sort(tab_name, driver):
    """Options for this program are `Latest` and `Top`"""
    tab = driver.find_element_by_link_text(tab_name)
    tab.click()
    xpath_tab_state = f'//a[contains(text(),\"{tab_name}\") and @aria-selected=\"true\"]'


def generate_tweet_id(tweet):
    return ''.join(tweet)

def scroll_completely(driver, scroll_attempt, num_seconds_to_load=0.2 , max_attempts=10):
    curr_position = driver.execute_script("return window.pageYOffset;")
    while True:
        last_position = curr_position
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        curr_position = driver.execute_script("return window.pageYOffset;")
        sleep(num_seconds_to_load)
        if curr_position == last_position:
            if scroll_attempt == max_attempts:
                break
            else:
                scroll_completely(driver, scroll_attempt + 1)
        last_position = curr_position

def scroll_down_page(driver, last_position, num_seconds_to_load=0.3, scroll_attempt=0, max_attempts=100):
    """The function will try to scroll down the page and will check the current
    and last positions as an indicator. If the current and last positions are the same after `max_attempts`
    the assumption is that the end of the scroll region has been reached and the `end_of_scroll_region`
    flag will be returned as `True`"""
    end_of_scroll_region = False
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    sleep(num_seconds_to_load)
    curr_position = driver.execute_script("return window.pageYOffset;")
    if curr_position == last_position:
        if scroll_attempt < max_attempts:
            end_of_scroll_region = True
        else:
            scroll_down_page(last_position, curr_position, scroll_attempt + 1)
    last_position = curr_position
    print(end_of_scroll_region)
    return last_position, end_of_scroll_region


def save_tweet_data_to_csv(records, filepath, mode='a+'):
    header = ['User', 'Handle', 'PostDate', 'TweetText', 'ReplyCount', 'RetweetCount', 'LikeCount']
    with open(filepath, mode=mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(header)
        if records:
            writer.writerow(records)


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


def main(url, filepath, minDate, age_sort='Latest'):
    last_position = None
    end_of_scroll_region = False
    unique_tweets = set()
    driver = create_webdriver_instance()
    pageLoaded = gotoUrl(url, driver)
    if not pageLoaded:
        return
    sleep(4)
    #search_found = find_search_input_and_enter_criteria(search_term, driver)
    #if not search_found:
        #return

    #change_page_sort(page_sort, driver)

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
                save_tweet_data_to_csv(tweet, filepath)
            else:
                print("double processing")
                continue
            indexOfTime = tweet[2].find("T")
            dateArr = tweet[2][:indexOfTime].split("-")
            date = datetime.datetime(int(dateArr[0]), int(dateArr[1]), int(dateArr[2]))
            if(date < minDate):
                return
        last_position, end_of_scroll_region = scroll_down_page(driver, last_position)
    driver.quit()

def main2(url, filepath, minDate, age_sort='Latest'):
    driver = create_webdriver_instance()
    pageLoaded = gotoUrl(url, driver)
    if not pageLoaded:
        return
    sleep(4)
    scroll_completely(driver, 0)
    print("Start collection of tweets")
    cards = collect_all_tweets_from_current_view(driver)
    print("Finished collection of tweets")
    counter = 1
    for card in cards:
            try:
                tweet = extract_data_from_current_tweet_card(card)
            except exceptions.StaleElementReferenceException:
                continue
            if not tweet:
                continue
            #tweet_id = generate_tweet_id(tweet)
            #if tweet_id not in unique_tweets:
                #unique_tweets.add(tweet_id)
                #save_tweet_data_to_csv(tweet, filepath)
            #else:
                #print("double processing")
                #continue
            save_tweet_data_to_csv(tweet, filepath)
            print("saving tweet: " + str(counter))
            counter += 1
            #indexOfTime = tweet[2].find("T")
            #dateArr = tweet[2][:indexOfTime].split("-")
            #date = datetime.datetime(int(dateArr[0]), int(dateArr[1]), int(dateArr[2]))
            #if(date < minDate):
                #return
    driver.quit()

def getTwitterAcounts():
    with open('MdB.csv', newline='',encoding='cp1252') as f:
        reader = csv.reader(f)
        data = list(reader)
        #print(data)
    
    resultData = []
    for line in data:
        resultData.append(line[0])
    return resultData

if __name__ == '__main__':
    save_tweet_data_to_csv(None, csvPath, 'w')  # create file for saving records
    #userList = ["arminLaschet", "_FriedrichMerz", "Markus_Soeder"]
    userList = ["Th_Seitz_AfD"]
    for user in userList:
        url = 'https://twitter.com/' + user
        main2(url, csvPath, datetime.datetime(2010,1,1))

