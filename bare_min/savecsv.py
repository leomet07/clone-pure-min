import csv

def savetocsv(fname, row, header):
    # add row to CSV file
    append = False
    import csv
    try:
        with open(fname,'r') as userFile:
            userFileReader = csv.reader(userFile)
            #read the first row
            for row in userFileReader:
                if row == header:
                    append = True
    except FileNotFoundError:
        print("File not found")

    if append:
        print("Appending")
        with open(fname, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
    else:
        print("Writing..")
        with open(fname, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)
