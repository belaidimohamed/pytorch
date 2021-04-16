from selenium import webdriver
import time
import urllib.request
import os
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm

os.chdir(r'C:\Users\mohamed\Desktop\tensor')


browser = webdriver.Chrome(r"C:\Users\mohamed\Desktop\tensor\chromedriver.exe") #incase you are chrome
browser.get("https://www.google.com/")

search = browser.find_element_by_name("q")
search.send_keys("random images",Keys.ENTER)
print('hi')
elem = browser.find_element_by_link_text("Images")
elem.get_attribute("href")
elem.click()

value = 0
number_of_images = 12800
sub = []
while len(sub) < number_of_images :
    browser.execute_script("scrollBy("+ str(value) +",+1000);")
    value += 50
    time.sleep(1)
    print(len(sub))
    elem1 = browser.find_element_by_id("islmp")
    sub += elem1.find_elements_by_tag_name("img")
if  len(sub) - number_of_images > 0 :
    sub = sub[:number_of_images - len(sub)]

try:
    os.mkdir("downloads")
except FileExistsError:
    pass

count = 0
error = 0
print(len(sub))
for i in tqdm(range(len(sub))):
    src = sub[i].get_attribute('src')
    try:
        if src != None:
            src  = str(src)
            count+=1
            urllib.request.urlretrieve(src, os.path.join('downloads','image'+str(count)+'.jpg'))
        else:
            raise TypeError
    except TypeError:
        error += 1
print(error)