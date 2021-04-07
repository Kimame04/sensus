import re

def cleanString(string):
    string = re.sub('[\(\[].*?[\)\]]','',string)
    string = string.replace('.',' .\n')
    return string

if __name__ == "__main__":
    print(cleanString(input("Enter text: ")))
    print("done")