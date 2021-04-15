import re

def readFromTxt(filename):
    with open(filename,'r') as file:
        content = cleanString(file.read())
    return content

def cleanString(string):
    string = re.sub('[\(\[].*?[\)\]]','',string)
    string = string.replace('.',' .\n')
    return string

if __name__ == "__main__":
    print(readFromTxt('inputtext.txt'))
    print("done")