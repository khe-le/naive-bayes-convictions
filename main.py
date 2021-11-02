import pandas as pd
import csv
from pprint import pprint
import json

# Returns JSON object as a dictionary - Get keys for 2018 Crime Statistics
f = open('keys.json')
keys = json.load(f)

# Calculate the probabilities of exonerated and non-exonerated convictions
with open('data/2018CrimeStats.csv', newline='') as csvfile:
    csvReader = csv.reader(csvfile)
    non_exn = len(list(csvReader)) # non-exonerations

exn = pd.read_excel('data/publicspreadsheet.xlsx').shape[0]
total_convictions = exn + non_exn # total number of convictions
probE = exn/total_convictions # exoneration probability
probN = 1 - probE # non-exoneration probability

# Non-exonerated convictions from 2018 Crime Statistics file, assuming that they're not exonerated (explained in write-up)
class NonExonerations():
    def __init__(self):
        self.varList = ["OFFGUIDE", "DISTRICT", "MONRACE"]
        self.df = pd.read_csv("data/2018CrimeStats.csv", usecols=self.varList)
        self.offenses = {}
        self.locations = {}
        self.races = {}
        
    # Convert keys to corresponding values    
    def translateData(self):
        self.df.fillna(-999, inplace = True)
        dfLen = self.df.shape[0]
        for i in range(dfLen):
            # replace offguide keys with values
            offguideKey = self.df.at[i,self.varList[0]]
            if offguideKey != -999:
                self.df.loc[i,self.varList[0]] = keys['offguide'][str(offguideKey)]
            else:
                self.df.loc[i,self.varList[0]] = keys['offguide']['na']

            # replace district keys with values
            districtKey = self.df.at[i,self.varList[1]]
            if districtKey != -999:
                self.df.loc[i,self.varList[1]] = keys['district'][str(districtKey)]
            else:
                self.df.loc[i,self.varList[1]] = keys['offguide']['na']

            # replace monrace keys with values
            monraceKey = self.df.at[i,self.varList[2]]
            if monraceKey != -999:
                self.df.loc[i,self.varList[2]] = keys['monrace'][str(monraceKey)]
            else:
                self.df.loc[i,self.varList[2]] = keys['monrace']['na']

    # Hepler function
    def addToDict(self, value, dict):
        if value not in dict:
            dict[value] = 1
        else:
            dict[value] += 1

    # Calculate probability of each var
    def computeVarsProb(self):
        self.translateData()
        # Read vars in dataframe
        for index, row in self.df.iterrows():
            self.addToDict(row[self.varList[0]],self.offenses)
            self.addToDict(row[self.varList[1]],self.locations)
            self.addToDict(row[self.varList[2]],self.races)

        # Calculate probability of each var
        for dict in [self.offenses, self.locations, self.races]:
            for k in dict:
                prob = dict[k]/total_convictions
                dict[k] = prob

    # The resulting machine learning model
    def computeNonExnProb(self, offense, loc, race):
        if offense not in self.offenses or loc not in self.locations or race not in self.races:
            print('Invalid inputs')
        else:
            prob = probN * self.offenses[offense] * self.locations[loc] * self.races[race] 
            print("Probability that this conviction is NOT exonerated given offense = {} | location = {} | race = {}: {}".format(offense, loc, race, prob))


# Read exonerated convictions from Excel file (50 yrs)
class Exonerations():
    def __init__(self):
        self.df = pd.read_excel('data/publicspreadsheet.xlsx')
        self.total_exonerations = self.df.shape[0]
        self.varList = ["Worst Crime Display", "State", "Race"]
        self.offenses = {}
        self.locations = {}
        self.races = {}

    # Hepler function
    def addToDict(self, value, dict):
        if value not in dict:
            dict[value] = 1
        else:
            dict[value] += 1

    def computeVarsProb(self):
        # Read vars in dataframe
        for index, row in self.df.iterrows():
            self.addToDict(row[self.varList[0]],self.offenses)
            self.addToDict(row[self.varList[1]],self.locations)
            self.addToDict(row[self.varList[2]],self.races)

        # Calculate probability of each var
        for dict in [self.offenses, self.locations, self.races]:
            for k in dict:
                prob = dict[k]/self.total_exonerations
                dict[k] = prob

    # The resulting machine learning model
    def computeExnProb(self, offense, loc, race):
        if offense not in self.offenses or loc not in self.locations or race not in self.races:
            print('Invalid inputs')
        else:
            prob = probE * self.offenses[offense] * self.locations[loc] * self.races[race]
            print("Probability that this conviction is exonerated given offense = {} | location = {} | race = {}: {}".format(offense, loc, race, prob))


if __name__ == "__main__": 
    exonerations = Exonerations()
    exonerations.computeVarsProb()

    # with open('exnVarsProb.log', 'w') as f:
    #     pprint("----------EXONERATED CONVICTIONS----------", stream=f)
    #     pprint("----------OFFENSES----------", stream=f)
    #     pprint(exonerations.offenses, stream=f)
    #     pprint("----------LOCATIONS----------", stream=f)
    #     pprint(exonerations.locations, stream=f)
    #     pprint("----------RACES----------", stream=f)
    #     pprint(exonerations.races, stream=f)

    nonExonerations = NonExonerations()
    nonExonerations.computeVarsProb()

    # with open('nonexnVarsProb.log', 'w') as f:
    #     pprint("----------NONEXONERATED CONVICTIONS----------", stream=f)
    #     pprint("----------OFFENSES----------", stream=f)
    #     pprint(nonExonerations.offenses, stream=f)
    #     pprint("----------LOCATIONS----------", stream=f)
    #     pprint(nonExonerations.locations, stream=f)
    #     pprint("----------RACES----------", stream=f)
    #     pprint(nonExonerations.races, stream=f)

    # Testing 2 models HERE
    print('Testing...')
    exonerations.computeExnProb('Murder','New Jersey', 'Black')
    nonExonerations.computeNonExnProb('Murder','New Jersey', 'Black/African American')

    exonerations.computeExnProb('Bribery','Connecticut', 'White')
    nonExonerations.computeNonExnProb('Bribery/Corruption','Connecticut', 'White/Caucasian')
