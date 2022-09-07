import selenium
import json
from selenium import webdriver
from selenium.common.exceptions import JavascriptException, WebDriverException
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
import geckodriver_autoinstaller
import os
import argparse



URL = "http://localhost:8000/"
OUTPUT_DIR = "./results"

def get_driver(browser_s):
    ''' Create a driver instance and returns it.
    We use it to waste less time starting and closing the browser.

    Parameters:
    browser_s(string): The name of the browser. Either firefox or chrome

    Returns:
    selenium.webdriver: A handle to the browser
    '''
    if (browser_s == 'firefox'):
        options = FirefoxOptions()
        options.add_argument("-devtools") # Opening the console somehow makes things better
        options.add_argument("-headless") # Opening the console somehow makes things better

        driver = webdriver.Firefox(options=options)
    elif (browser == 'chrome'):
        options = ChromeOptions()
        options.add_argument("auto-open-devtools-for-tabs") # Opening the console somehow makes things better
        driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    driver.set_script_timeout(10000000000) # Sometimes the computations are too long so we set a high timeout ot be sure
    driver.get(URL)
    return driver





def framework(iterations, output_dir):
    ''' Runs the framework evaluating all pairs of wasm instructions for potential distinguishers.
    Outputs several json files, one per iteration, containing the measurement times of both grouped and interleaved experiments for each pair of instructions.
    The actual measurement is made in JavaScript, this framework is simply a Selenium handle over the actual web-based experiment.

    Parameters:
    iterations(int): The number of iteration of the framwork, i.e., the number of times we test each pair of operations. We restart the browser between each iteration.
    output_dir(str): The directory where the result files will be stored.
    '''
    geckodriver_autoinstaller.install()
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for i in range(iterations):
        print("Running experiment {}/{}.".format(i+1,iterations))
        output = output_dir+"/results_{}.json".format(i)
        driver = get_driver('firefox')
        results = driver.execute_script("""return testAll()""" )
        with open(output,'w') as file:
            json.dump(results,file)
        driver.quit()
    print("done")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', help='Number of experiment iterations (hence number of result files)', type=int, default = 100)
    parser.add_argument('-o', '--output', help='Output directory', type=str, default = OUTPUT_DIR)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    framework(args.iterations,args.output)
