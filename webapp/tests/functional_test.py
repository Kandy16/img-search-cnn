import time
import unittest
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


class TestHomePage(unittest.TestCase):
    def setUp(self):
        self.browser = webdriver.Firefox()

    def tearDown(self):
        self.browser.close()

    def test_home_page_will_show_inputfield(self):
        self.browser.get('http://localhost:5000')
        self.assertIn('DeepLearning', self.browser.title)

    def test_home_page_should_have_search_field(self):
        self.browser.get('http://localhost:5000')

        search_box = self.browser.find_element_by_id('searchbox')
        self.assertTrue(search_box)
        search_button = self.browser.find_element_by_id('searchbtn')

        self.assertTrue(search_button)

    def test_searching_image_will_show_search_page(self):
        self.browser.get('http://localhost:5000')

        search_box = self.browser.find_element_by_id('searchbox')
        search_box.send_keys('cat')
        search_button = self.browser.find_element_by_id('searchbtn')
        search_button.send_keys(Keys.ENTER)

        time.sleep(2)
        result_text = self.browser.find_element_by_id('search_text')
        print(result_text)
        self.assertIn('cat', result_text)



if __name__ == '__main__':
    unittest.main(warnings=False)
