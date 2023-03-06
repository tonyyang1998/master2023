import pandas as pd
import json




def main(book, sheet_name):
    
    df2 = pd.read_excel(book, sheet_name=sheet_name)

    capacity = df2['Capacity']

    df3 = df2[['Rider','Origin location',  "Destination location", 'Max ride time']]

    df1 = df3.dropna()

    df = df1.join(capacity)

    pd.set_option('display.max_rows', None, 'display.max_columns', None)

    riders = df['Rider']
    numbers = []


    for number in riders.index:
        numbers.append(number)

    sample_passenger = {}
    sample_driver = {}
    
    for i in numbers:
        object = {}
        object['id'] = i

        object["origin_location"] = df["Origin location"][i]
        object["destination_location"] = df["Destination location"][i]
        object['lower_tw'] = 120
        object['upper_tw'] = 170
        object['max_ride_time'] = df['Max ride time'][i]
        if riders[i] == 'Passenger':
            id_string = "P" +str(i) + ""
            sample_passenger[id_string] = object
        else:
            id_string = "D" +str(i) + ""
            object['max_capacity'] = df['Capacity'][i]
            sample_driver[id_string] = object

    with open('sample_passenger.json', 'w') as fp:
        json.dump(sample_passenger, fp, indent=2)

    with open('sample_driver.json', 'w') as fp:
        json.dump(sample_driver, fp, indent=2)



