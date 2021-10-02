from selenium import webdriver
from time import sleep

#empty list for store data
import pandas as pd
person = []
review = []
Date=[]
data = pd.DataFrame()

for page_no in range(1,256):
    #URL for 1st page
    url = "https://www.etsy.com/in-en/c/jewelry/earrings/ear-jackets-and-climbers?ref=catcard-2900-171816901&explicit=1&page="+str(page_no)
    browser = webdriver.Chrome('C:/Users/win/chromedriver')
    browser.get(url)
    
    #this one change time by time have to copy new in every new session
    try:
        try:
            PATH_1 = '//*[@id="content"]/div/div[1]/div/div[5]/div[2]/div[2]/div[4]/div/div/div/ul'
            items = browser.find_element_by_xpath(PATH_1)
        except:
            PATH_1 = '//*[@id="content"]/div/div[1]/div/div[5]/div[2]/div[2]/div[5]/div/div/div/ul'
            items = browser.find_element_by_xpath(PATH_1)
    except:
         try:
            PATH_1 = '//*[@id="content"]/div/div[1]/div/div[3]/div[2]/div[2]/div[6]/div/div/div/ul'
            items = browser.find_element_by_xpath(PATH_1)
         except:
            PATH_1 = '//*[@id="content"]/div/div[1]/div/div[5]/div[2]/div[2]/div[6]/div/div/div/ul'
            items = browser.find_element_by_xpath(PATH_1)
    
    items = items.find_elements_by_tag_name('li')
    #available items in page
    end_product = len(items)
    
    for i in range(0,end_product):
        try:
            items[i].find_element_by_tag_name('a').click()
            windows = browser.window_handles
            browser.switch_to.window(windows[1])
        except:
            continue
        #sleep(3)
        
        try:
            try:
                try:
                    count = browser.find_element_by_id('same-listing-reviews-panel')
                except:
                    count = browser.find_element_by_xpath('//*[@id="reviews"]/div[2]/div[2]/div[1]')
            except:
                try:
                    count = browser.find_element_by_xpath('//*[@id="same-listing-reviews-panel"]/div')
                except:
                    count = browser.find_element_by_xpath('//*[@id="same-listing-reviews-panel"]/div')
                
            count = count.find_elements_by_class_name('wt-grid__item-xs-12')
        except:
            count = 0
            
        if count != 0:
            for r2 in range(0,len(count)):
              
                try:
                    path = '//*[@id="review-preview-toggle-'+ str(r2) +'"]'
                    data3 = count[r2].find_element_by_xpath(path).text
                    review.append(data3)
                    
                    data1 = count[r2].find_element_by_tag_name('p').text
                    data2 = data1.split()
                    data2 = data2[-3] + ' ' + data2[-2] + ' ' + data2[-1]
                    data1 = data1.replace(data2,'')[:-1]
                    
                    Date.append(data2)
                    person.append(data1)
                except:
                    pass
              
        browser.close()
        browser.switch_to.window(windows[0])
    browser.close()
            
data["Date"] = Date
data["Person"] = person
data["Reviews"] = review
data.to_csv('scrappedReviews.csv',index=False)