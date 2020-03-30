import unittest
from appium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
import os
import time
import random,datetime
def r_string():#生成随机字符串
    #用时间来做随机播种
    random.seed(time.time())
    #随机选取数据
    if random.randint(0,100)<67:
        return True
    else:
        return False


class SimpleCalculatorTests(unittest.TestCase):
    '''
    Hyperparams is in setUpClass
    '''
    @classmethod
    def setUpClass(self):
        #Hyperparams
        desired_caps = {}
        desired_caps["app"] = r"D:\work\MikuMikuDance 10th Anniversary Version\MikuMikudance.exe"
        self.index = r"D:\download_cache\PMXmodel\index.csv"
        self.process_range = [0,3358] # start_num, num_of_read
        self.base_dir = "D:\download_cache\PMXmodel"
        self.clip_index_dir = r"D:\download_cache\PMXmodel\CLIPIndex"
        # self.pairs = [[r'D:\work\OpenMMD1.0\examples\SourClassicMiku\SourClassicMiku.pmx', r'D:\work\OpenMMD1.0\examples\SourClassicMiku\SourClassicMiku.pmd', 30, 25]]
        # df = pd.read_csv(self.index, header=None,skiprows=self.process_range[0],nrows=self.process_range[1])
        df = pd.read_csv(self.index, header=None,skiprows=301)
        self.pairs = df.values.tolist()
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
            videofile_,pmxfile_,fr,frames = pmx
            pmxfile_ = str(pmxfile_)
            if pmxfile_=="pmdfile1":
                pmxfile = self.base_dir+"\PMXfile\\"+pmxfile_+"\\"+pmxfile_+".pmd"
            else:
                pmxfile = self.base_dir+"\PMXfile\\"+pmxfile_+"\\"+pmxfile_+".pmx"
            vmdfile = self.base_dir + "\VMDfile\\" + videofile_+"_"+pmxfile_+".vmd"
            if os.path.exists(os.path.join(self.clip_index_dir,videofile_+".csv")):
                if not os.path.exists(vmdfile):
                    continue
                if not r_string():
                    time.sleep(1)
                    continue
                print(videofile_)
                clipindex = pd.read_csv(os.path.join(self.clip_index_dir,videofile_+".csv"),header=None)

                #pmx
                self.driver.find_element_by_name("載　入").click()
                self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'文件名(N)')]").send_keys(pmxfile+Keys.ENTER)
                self.driver.find_element_by_xpath("//Button[starts-with(@Name,'确定')]").click()
                #pmd
                self.driver.find_element_by_name(" 文件 (F)  |").click()
                self.driver.find_element_by_xpath('//MenuItem[starts-with(@Name, "导入动作数据")]').click()
                time.sleep(1)
                self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'文件名(N)')]").send_keys(vmdfile + Keys.ENTER)
                self.driver.find_element_by_xpath("//Button[starts-with(@Name,'确定')]").click()
                #output

                for num,clip in enumerate(clipindex.values.tolist()):
                    aviname = videofile_+"_"+str(num)+"_"+pmxfile_+".avi"
                    start_frame, end_frame = clip
                    self.driver.find_element_by_name(" 文件 (F)  |").click()
                    self.driver.find_element_by_xpath('//MenuItem[starts-with(@Name, "导出视频 ")]').click()

                    # self.driver.find_element_by_xpath("ToolBar[@ClassName='ToolbarWindow32']").click()
                    # self.driver.find_element_by_xpath(
                    #     '//Edit[starts-with(@Name, "地址")]').send_keys(
                    #     self.target_dir + Keys.ENTER)
                    # self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'地址')]").send_keys(self.target_dir + Keys.ENTER)
                    time.sleep(1)
                    self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'文件名')]").send_keys(aviname + Keys.ENTER)
                    # self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'FPS')]").send_keys(Keys.BACKSPACE+Keys.BACKSPACE+str(fr))
                    self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'录制范围')]").send_keys(str(start_frame))
                    self.driver.find_element_by_xpath("//Edit[starts-with(@Name,'至')]").send_keys(str(end_frame))
                    self.driver.find_element_by_xpath("//ComboBox[starts-with(@Name,'视频压缩编码')]").send_keys(Keys.ARROW_UP)


                    self.driver.find_element_by_xpath("//Button[starts-with(@Name,'确认')]").click()
                    sleeptime = int((int(end_frame)-int(start_frame))*0.055)+5
                    time.sleep(sleeptime)
                # delete
                self.driver.find_elements_by_xpath("//Button[starts-with(@Name,'刪　除')]")[1].click()
                self.driver.find_element_by_xpath("//Button[starts-with(@Name,'确定')]").click()









if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SimpleCalculatorTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
