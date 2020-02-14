import unittest
from appium import webdriver
from selenium.webdriver.common.keys import Keys

class SimpleCalculatorTests(unittest.TestCase):




    @classmethod
    def setUpClass(self):
        #set up appium
        desired_caps = {}
        desired_caps["app"] = r"D:/work/MikuMikuDance 10th Anniversary Version/MikuMikudance.exe"
        self.target_dir = r"D:\work"
        self.pairs = [[r'D:\work\OpenMMD1.0\examples\SourClassicMiku\SourClassicMiku.Pmx', r'D:\work\OpenMMD1.0\examples\SourClassicMiku\SourClassicMiku.pmd', 30, 25]]
        self.driver = webdriver.Remote(
            command_executor='http://127.0.0.1:4723',
            desired_capabilities= desired_caps)

    @classmethod
    def tearDownClass(self):
        self.driver.quit()

    # def getresults(self):
    #     displaytext = self.driver.find_element_by_accessibility_id("CalculatorResults").text
    #     displaytext = displaytext.strip("Display is " )
    #     displaytext = displaytext.rstrip(' ')
    #     displaytext = displaytext.lstrip(' ')
    #     return displaytext

    def test_initialize(self):
        for pmx in self.pairs:
            # for each video, costing 224s
            pmxfile,pmdfile,frame_end,fr = pmx
            #pmx
            self.driver.find_element_by_name("載　入").click()
            self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'文件名(N)')]").send_keys(pmxfile+Keys.ENTER)
            self.driver.find_element_by_xpath("//Button[starts-with(@Name,'确定')]").click()
            #pmd
            self.driver.find_element_by_name(" 文件 (F)  |").click()
            self.driver.find_element_by_xpath('//MenuItem[starts-with(@Name, "导入动作数据")]').click()
            self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'文件名(N)')]").send_keys(pmdfile + Keys.ENTER)
            self.driver.find_element_by_xpath("//Button[starts-with(@Name,'确定')]").click()
            #output
            aviname = pmxfile.split("\\")[-1][:-4]
            self.driver.find_element_by_name(" 文件 (F)  |").click()
            self.driver.find_element_by_xpath('//MenuItem[starts-with(@Name, "导出视频 ")]').click()

            # self.driver.find_element_by_xpath("ToolBar[@ClassName='ToolbarWindow32']").click()
            # self.driver.find_element_by_xpath(
            #     '//Edit[starts-with(@Name, "地址")]').send_keys(
            #     self.target_dir + Keys.ENTER)
            # self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'地址')]").send_keys(self.target_dir + Keys.ENTER)
            self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'文件名')]").send_keys(aviname + Keys.ENTER)
            self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'FPS')]").send_keys(Keys.BACKSPACE+Keys.BACKSPACE+str(fr))
            self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'录制范围')]").send_keys('0')
            self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'至')]").send_keys(str(frame_end))
            self.driver.find_element_by_xpath("//Button[starts-with(@Name,'确认')]").click()
            # delete
            self.driver.find_elements_by_xpath("//Button[starts-with(@Name,'刪　除')]")[1].click()
            self.driver.find_element_by_xpath("//Button[starts-with(@Name,'确定')]").click()









if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SimpleCalculatorTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
