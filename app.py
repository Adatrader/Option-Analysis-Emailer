# Filename: app.py
# Author: Adam VanRiper
# Description: Email Notification Service for market making activity in S&P 500 option chain with ARIMA machine learning graph

# Utility Dependencies: 
import requests
import json
from threading import Timer
from time import sleep
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Data Analysis Dependencies:
import pandas as pd
from pandas.io.json import json_normalize
from pandas.core.base import NoNewAttributesMixin
from pandas.core.frame import DataFrame
from requests.api import put

# Email Service Dependencies:
from IPython.display import display
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Machine Learning
from statsmodels.tsa.arima_model import ARIMA
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('bmh')

# -------------------------------------------------------------------------------------------
# Globals
# -------------------------------------------------------------------------------------------
# Update your TD ameritrade consumer key: https://developer.tdameritrade.com/apis
api_key = '<Put TD Ameritrade API key here>'

# Update the mon,wed,friday SPY 0-day contract date (Year, Month, Day)
todays_date = "####-##-##"

# Emailer
outbound_email_username = '<Sender email username here>'
outbound_email_password = '<Sender email password here>'
reciever_email = '<Reciever email here>'

# -------------------------------------------------------------------------------------------
# API TICKER QUOTE CLASS
# -------------------------------------------------------------------------------------------

class td_quote_client(object):
    # Information
    def __init__(self):
        self.td_consumer_key = api_key
        self.price_quote_endpoint = 'https://api.tdameritrade.com/v1/marketdata/{symbol}/quotes?'
        self.tickerJson = {}  # Response Json ticker
        self.ticker_quote(self.price_quote_endpoint, self.td_consumer_key)
        self.price_history_json = {}
        self.price_history_df = DataFrame

    # Get Equity Price Quote
    def ticker_quote(self, endpoint, key):
        # Set ticker
        full_endpoint = endpoint.format(symbol='SPY')
        # Call API
        page = requests.get(url=full_endpoint, params={
                            'apikey': key})
        self.tickerJson = json.loads(page.content)  # Update ticker Json

    # Get Price history
    def price_history(self):
        key = self.td_consumer_key
        endpoint = 'https://api.tdameritrade.com/v1/marketdata/{stock_ticker}/pricehistory?periodType={periodType}&period={period}&frequencyType={frequencyType}&frequency={frequency}'
        full_url = endpoint.format(
            stock_ticker='SPY', periodType='day', period=10, frequencyType='minute', frequency=5)
        page = requests.get(url=full_url, params={
                            'apikey': key, 'needExtendedHoursData': 'true'})
        self.price_history_json = page.json()
        self.price_history_df = json_normalize(
            self.price_history_json['candles'])
        print(self.price_history_df)

# -------------------------------------------------------------------------------------------
# ARIMA -MACHINE LEARNING GRAPH OUTPUT
# -------------------------------------------------------------------------------------------

    def machine_learning(self):  # Arima Machine Learning
        generatedDf = self.price_history_df[['datetime', 'close']]
        generatedDf.datetime = pd.to_datetime(
            self.price_history_df.datetime, unit='ms')
        generatedDf = generatedDf.set_index("datetime")
        plt.figure(figsize=(8, 4))

        a = b = c = range(0, 3)  # a, b, c random values
        genCombos = list(itertools.product(a, b, c))  # Generate combinations
        warnings.filterwarnings("ignore")
        modList = []
        params = []
        for element in genCombos:
            try:
                mod = sm.tsa.statespace.SARIMAX(generatedDf, order=element,
                                                enforce_stationarity=True, enforce_invertibility=True)
                results = mod.fit()
                modList.append(results.aic)  # save results in list
                params.append(element)  # save element in list
            except:
                continue
        # find lowest in modList
        index_min = min(range(len(modList)), key=modList.__getitem__)
        model = ARIMA(generatedDf, order=params[index_min])
        model_fit = model.fit(disp=0)
        print(model_fit.summary())
        model_fit.plot_predict(start=2, end=len(generatedDf)+200)
        plt.title('10-Day SPY Price (5 min)')
        plt.savefig('spy10day.png')

    def printQuote(self):  # Print quote

        lp = self.tickerJson['SPY']['lastPrice']
        nc = self.tickerJson['SPY']['netChange']
        pc = self.tickerJson['SPY']['netPercentChangeInDouble']
        vol = self.tickerJson['SPY']['totalVolume']
        low = self.tickerJson['SPY']['lowPrice']
        high = self.tickerJson['SPY']['highPrice']

        stringSent = """    
        <table>
        <tr>
        <th>Last Price</th>
        <th>Net Change</th>
        <th>% Change</th>
        <th>Volume</th>
        <th>Low Price</th>
        <th>High Price</th>
        </tr>
        """
        dataString = """<tr><td>""" + str(lp) + """</td>""" + """<td>""" + str(nc) + """</td>""" + """<td>""" + str(pc) + """</td>""" + """<td>""" + str(
            vol) + """</td>""" + """<td>""" + str(low) + """</td>""" + """<td>""" + str(high) + """</td></tr></table>"""
        stringSent += dataString
        return stringSent

# -------------------------------------------------------------------------------------------
# API OPTION SERVICE 
# -------------------------------------------------------------------------------------------

class td_option_clent(object):
    # Information
    def __init__(self, current_date):
        self.current_date = current_date
        self.td_consumer_key = api_key
        self.option_chain_endpoint = 'https://api.tdameritrade.com/v1/marketdata/chains?&symbol={symbol}'
        self.optionJson = {}  # Response Json option chain
        self.putsSplitJson = {}  # Break up the json response
        self.callsSplitJson = {}  # Into puts and calls
        self.putsDict = {}  # Store strike : index pairs
        self.callsDict = {}  # Store strike : index pairs
        self.lastTime = None  # Store the last time for comparing volume change
        self.newTime = None  # Store new time

        # Dataframe
        self.putsDataframe = pd.DataFrame(columns=['Description', 'Strike'])
        self.callsDataframe = pd.DataFrame(columns=['Description', 'Strike'])

        self.initializeDataframe()  # Load Dataframe

    # Get Option Chain
    def option_chain(self, endpoint, key):
        # Set ticker
        full_endpoint = endpoint.format(symbol='SPY')
        # Call API
        # stikeCount = 20 (Get +-10 contracts around strike)
        page = requests.get(url=full_endpoint, params={
                            'apikey': key, 'strategy': "SINGLE", 'strikeCount': 20,
                            'fromDate': self.current_date, 'toDate': self.current_date})
        self.optionJson = json.loads(page.content)  # Update option Json

    # Set up pandas dataframe
    def initializeDataframe(self):
        self.option_chain(self.option_chain_endpoint, self.td_consumer_key)
        date = list(self.optionJson['putExpDateMap'].keys())[0]
        self.putsSplitJson = self.optionJson['putExpDateMap'][date]
        self.callsSplitJson = self.optionJson['callExpDateMap'][date]

        self.lastTime = self.getTime()  # Get time
        # Add new column with title as current time for puts df
        self.putsDataframe[self.getTime()] = None
        # Add new column with title as current time for calls df
        self.callsDataframe[self.getTime()] = None

        countPuts = int(0)
        countCalls = int(0)
        for contract in self.putsSplitJson:  # Loop through puts json
            description = self.putsSplitJson.get(contract)[0]['description']
            strike = contract  # Strike
            des = description  # Description
            volume = self.putsSplitJson.get(contract)[0]['totalVolume']
            # Update dictionary with strike : index
            self.putsDict.update({strike: countPuts})
            # Insert new row into dataframe
            self.putsDataframe.loc[len(self.putsDataframe.index)] = [
                des, strike, volume]
            countPuts += 1

        for contract in self.callsSplitJson:  # Loop through calls json
            description = self.callsSplitJson.get(contract)[0]['description']
            strike = contract  # Strike
            des = description  # Description
            volume = self.callsSplitJson.get(contract)[0]['totalVolume']
            # Update dictionary with strike : index
            self.callsDict.update({strike: countCalls})
            # Insert new row into dataframe
            self.callsDataframe.loc[len(self.callsDataframe.index)] = [
                des, strike, volume]
            countCalls += 1
        # Print
        display(self.putsDataframe)
        display(self.callsDataframe)
        # print(json.dumps(self.optionJson, sort_keys=True, indent=4))

    # update dataframe every set interval
    def updateDataframe(self):
        self.option_chain(self.option_chain_endpoint, self.td_consumer_key)
        date = list(self.optionJson['putExpDateMap'].keys())[0]
        self.putsSplitJson = self.optionJson['putExpDateMap'][date]
        self.callsSplitJson = self.optionJson['callExpDateMap'][date]

        self.newTime = self.getTime()  # Get time
        # Add new column with title as current time for puts df
        self.putsDataframe[self.getTime()] = None
        # Add new column with title as current time for calls df
        self.callsDataframe[self.getTime()] = None
        callsTriggered = {}
        putsTriggered = {}
        htmlSpread = """
        <br>
        <h3>Multi-Leg Activity:</h3>
        <br>
        <table>
        <tr>
        <th>Contract</th>
        <th>Volume</th>
        </tr>"""
        htmlSingle = """    
        <br>
        <h3>Single-Leg Activity:</h3>
        <br>
        <table>
        <tr>
        <th>Contract</th>
        <th>Volume</th>
        </tr>"""

# -------------------------------------------------------------------------------------------
# OPTION PUTS ANALYSIS
# -------------------------------------------------------------------------------------------
        
        for contract in self.putsSplitJson:  # Loop through puts json
            strike = contract  # Strike
            volume = self.putsSplitJson.get(contract)[0]['totalVolume']
            if(self.putsDataframe.isin([strike]).any().any()):
                self.putsDataframe.at[self.putsDict.get(
                    strike), self.newTime] = volume
                if self.putsDataframe.at[self.putsDict.get(strike), self.lastTime] is not None and volume is not None:
                    increasedVolume = volume - self.putsDataframe.at[self.putsDict.get(
                        strike), self.lastTime]
                    if(increasedVolume > 2000):
                        putsTriggered.update({self.putsSplitJson.get(contract)[
                            0]['description']: increasedVolume})

        # Check for credit screads
        if putsTriggered:  # Check if empty
            putSpreadsArr = []
            for element in putsTriggered:  # Loop through dict
                if element in putSpreadsArr:
                    continue
                for element2 in putsTriggered:  # dict element minus outer loop element
                    if element2 != element and abs(putsTriggered.get(element) - putsTriggered.get(element2)) < 400:
                        temp = """<tr><td>""" + element + """</td>""" + """<td>""" + str(putsTriggered.get(
                            element)) + """</td></tr>""" + """<tr><td>""" + element2 + """</td>""" + """<td>""" + str(putsTriggered.get(element2)) + """</td></tr>"""
                        htmlSpread += temp
                        putSpreadsArr.append(element)
                        putSpreadsArr.append(element2)
            # Check for single legs
            for element in putsTriggered:
                if element in putSpreadsArr:
                    continue
                temp = """<tr><td>""" + element + """</td>""" + """<td>""" + \
                    str(putsTriggered.get(element)) + """</td></tr>"""
                htmlSingle += temp

# -------------------------------------------------------------------------------------------
# OPTION CALLS ANALYSIS
# -------------------------------------------------------------------------------------------

        for contract in self.callsSplitJson:  # Loop through calls json
            strike = contract  # Strike
            volume = self.callsSplitJson.get(contract)[0]['totalVolume']
            if(self.callsDataframe.isin([strike]).any().any()):
                self.callsDataframe.at[self.callsDict.get(
                    strike), self.newTime] = volume
                if self.callsDataframe.at[self.callsDict.get(strike), self.lastTime] is not None and volume is not None:
                    increasedVolume = volume - self.callsDataframe.at[self.callsDict.get(
                        strike), self.lastTime]
                    if(increasedVolume > 2000):
                        callsTriggered.update({self.callsSplitJson.get(contract)[
                            0]['description']: increasedVolume})

        # Check for credit screads
        if callsTriggered:  # Check if empty
            callSpreadsArr = []
            for element in callsTriggered:  # Loop through dict
                if element in callSpreadsArr:
                    continue
                for element2 in callsTriggered:  # dict element minus outer loop element
                    if element2 != element and abs(callsTriggered.get(element) - callsTriggered.get(element2)) < 400:
                        temp = """<tr><td>""" + element + """</td>""" + """<td>""" + str(callsTriggered.get(
                            element)) + """</td></tr>""" + """<tr><td>""" + element2 + """</td>""" + """<td>""" + str(callsTriggered.get(element2)) + """</td></tr>"""
                        htmlSpread += temp
                        callSpreadsArr.append(element)
                        callSpreadsArr.append(element2)
            # Check for single legs
            for element in callsTriggered:
                if element in callSpreadsArr:
                    continue
                temp = """<tr><td>""" + element + """</td>""" + """<td>""" + \
                    str(callsTriggered.get(element)) + """</td></tr>"""
                htmlSingle += temp

        # If significant volume detected, call emailer service
        if callsTriggered or putsTriggered:
            # Add table end html
            htmlSpread += """</table>"""
            htmlSingle += """</table>"""

            # Check if elements have been updated to html string
            stringGen = ""
            # Combine spread and single leg tables
            if len(htmlSingle) != 169:
                stringGen += htmlSingle
            if len(htmlSpread) != 164:
                stringGen += htmlSpread
            # Call Email
            self.emailUpdate(stringGen)

        self.lastTime = self.newTime  # Update last time
        # Print
        display(self.putsDataframe)
        display(self.callsDataframe)

# -------------------------------------------------------------------------------------------
# SMTP EMAILER SERVICE
# -------------------------------------------------------------------------------------------

    def emailUpdate(self, stringGiven):
        # Combine html
        html = """ 
        <html>
        <head>
        <style>
        table,
        th,
        td {
            padding: 10px;
            align: center;
            border: 2px solid black;
            border-collapse: collapse;
        }
        </style>
        </head>
        <body> 
        <img src="cid:logoImage">
        <h2>S&P 500 ARIMA Forecast:</h2>
        <img src="cid:pricePrediction">
        <h2>S&P 500 Information:</h2>

        """

        htmlend = """
        <br>
        <br>
        <h5>The above references an opinion and is for information purposes only. 
        It is not intended to be investment advice. Seek a duly licensed professional for investment advice.</h5>
        </body>
        </html>"""

        spyQuote = td_quote_client()
        spyInfo = spyQuote.printQuote()
        html += spyInfo + stringGiven + htmlend

        sender = outbound_email_username
        receivers = [reciever_email]

        port = 587
        msg = MIMEMultipart()

        msg['Subject'] = 'Daily SPY Unusual Volume'
        msg['From'] = outbound_email_username
        msg['To'] = reciever_email

        htmlCreated = MIMEText(html, "html")
        msg.attach(htmlCreated)

        # Attach Logo
        fp = open('Banner_Logo.png', 'rb')
        image = MIMEImage(fp.read())
        fp.close()
        # ID tag for img src in HTML
        image.add_header('Content-ID', '<logoImage>')
        msg.attach(image)

        # Attach Prediction
        fp = open('spy10day.png', 'rb')
        image = MIMEImage(fp.read())
        fp.close()
        # ID tag for img src in HTML
        image.add_header('Content-ID', '<pricePrediction>')
        msg.attach(image)

        with smtplib.SMTP('mail.gmx.com', port) as server:
            server.starttls()
            server.login(outbound_email_username, outbound_email_password)
            server.sendmail(sender, receivers, msg.as_string())
            print("Successfully sent email")
            server.quit()

    def getTime(self):
        now = datetime.now()
        dtFormat = now.strftime("%H:%M")
        return dtFormat

# -------------------------------------------------------------------------------------------
# REPEAT CALL TO API
# -------------------------------------------------------------------------------------------


class RepeatTimer(object):  # Timer to repeat option call
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


# -------------------------------------------------------------------------------------------
# RUN PROGRAM
# -------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Spy Option Chain
    spyChain = td_option_clent(todays_date)
    spyHistory = td_quote_client()
    spyHistory.price_history()
    spyHistory.machine_learning()

    # Loop over updateDataframe
    print("Starting..")
    # First param is the wait interval (ex. 60 sec)
    repeat = RepeatTimer(60, spyChain.updateDataframe)
    try:
        # 23400 is full trading 6:30 - 1 pm
        sleep(23400)  # Number of seconds it should run (6.5 hours)
    finally:
        repeat.stop()  # try/finally block to make sure the program ends
