import pandas as pd
import sys
import numpy as np

def parse_input_file(filename, features):
    #Getthe original csv
    df = pd.read_csv(filename)
   
    #Organize into a multi-level dictionary
    results = {}
    for (state, year), group in df.groupby(["State", "Year"]):
        if state not in results:
            results[state] = {}

        #There'z only one element here, but just in case take the first
        results[state][year] = group[features].values[0]

    #Convert the dataframe to a list of dictionaries
    old_records = df.to_dict(orient="records")

    #Name the new features like this
    new_features = [elem + " prev" for elem in features]

    #Create a new list of dict by looking back into the table
    #we just made, iterating over the list of dict
    new_records = []
    for record in old_records:
        #Start from the existing
        new_record = record

        #Get the state and year for convenience
        state = record["State"]
        year = record["Year"]

        #Shouldn't ever happen, but best to be safe
        if state not in results:
            continue

        #Skip years that aren't usable -- could also 
        #fill with some sort of default value
        if year - 1 not in results[state]:
            continue

        #Above the features were put in in order of the 'features'
        #list and the new_features list was also in that same order
        #so we're just accessing by index here. could probably be cleaner
        for i, elem in enumerate(new_features):
            new_record[elem] = results[state][year-1][i]

        #Finally add in the constructed record
        new_records.append(new_record)

    #Building dataframe from this format is easy from constructor
    df_new = pd.DataFrame(new_records)
    return df_new

if __name__ == '__main__':
    #If we wanted we could use this for a couple features too
    print(parse_input_file(sys.argv[1], ["bank account"]).head())
