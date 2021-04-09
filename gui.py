import pickle
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import cleanString
import naive_conversion
from conversion import convert


class Application(tk.Frame):
    global model_direct
    model_direct = pickle.load(open('objectivity-detection-direct.sav','rb'))

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.title = tk.Label(self, text='Text Objectivity Detector and Text Voice Converter', font=('Roboto',30))
        self.title.grid(row=1,column=0,padx=10,pady=10,sticky=W)
        
        self.field = tk.Text(self)
        self.field.grid(row=2,column=0,padx=10,pady=5)
               
        self.upload = tk.Button(self, text = 'Upload .txt File', width = 60, command = self.uploadTxt)
        self.upload.grid(row=3,column=0,padx=10,pady=3)
        
        self.btn = tk.Button(self, text='Detect Objectivity', command = self.detectObjectivity)
        self.btn.grid(row=4,column=0,padx=50,pady=3,sticky=W)
        
        self.btn2 = tk.Button(self, text='Make More Objective', command = self.naive_obj)
        self.btn2.grid(row=4,column=0,padx=40,pady=3)
        
        self.btn3 = tk.Button(self, text='Make More Subjective', command = self.naive_subj)
        self.btn3.grid(row=4,column=0,padx=50,pady=3,sticky=E)

        self.res = tk.Label(self, text='', font=('Roboto',13))
        self.res.grid(row=5,column=0,padx=10,pady=10)  
        
        
    def detectObjectivity(self):
        arr = model_direct.predict_proba([self.field.get(1.0,'end')])
        self.res.config(text = 'Objective: ' + str(round(arr[0][1]*100,2)) + 
                        '%\nSubjective: ' + str(round(arr[0][0]*100,2)) + '%')
    def uploadTxt(self):
        tf = filedialog.askopenfilename(
            title="Open Text file", 
            filetypes=(("Text Files", "*.txt"),)
            )
        with open(tf, 'r') as file:
            data = file.read()
        self.field.insert(1.0,data)
    
    def makeMoreSubjective(self):
        t=self.field.get(1.0,'end')
        file = open('subjectified.txt','w')
        file.write(t)
        file.close()
        cleaned_text = cleanString.readFromTxt('subjectified.txt')
        text=convert(t.split("\n"),False)
        file = open('subjectified.txt','w')
        file.write(text)
        file.close()
        self.res.config(text = 'Text converted to be more subjective.\nPlease check your file system.')

    def makeMoreObjective(self):
        t=self.field.get(1.0,'end')
        file = open('objectified.txt','w')
        file.write(t)
        file.close()
        cleaned_text = cleanString.readFromTxt('objectified.txt')
        text=convert(t.split("\n"),True)
        file = open('objectified.txt','w')
        file.write(text)
        file.close()
        self.res.config(text = 'Text converted to be more objective.\nPlease check your file system.')
        
    def naive_obj(self):
        self.naive_convert(False)
    
    def naive_subj(self):
        self.naive_convert(True)
        
    def naive_convert(self,voice):
        temp = naive_conversion.test(self.field.get(1.0,'end'),voice)
        self.res.config(text = 'Text converted.\nPlease check your file system.')
        file = open('converted.txt','w')
        file.write(temp)
        file.close()

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Not a Fact Checker')

    app = Application(master=root)
    root.mainloop()