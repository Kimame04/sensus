from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter as tk 
import pickle
import numpy as np

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
        
        self.btn = tk.Button(self, text='Detect Objectivity', command = self.detectObjectivity)
        self.btn.grid(row=3,column=0,padx=40,pady=3,sticky=W)

        self.btn2 = tk.Button(self, text='Change Text Voice', command = self.generateConverted)
        self.btn2.grid(row=3,column=0,padx=40,pady=3,sticky=E)
        
        self.upload = tk.Button(self, text = 'Upload .txt File', command = self.uploadTxt)
        self.upload.grid(row=4,column=0,padx=10,pady=3)
        
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
    
    def generateConverted(self):
        result = model_direct.predict([self.field.get(1.0,'end')])
        if result[0] == 0:
            voice = 'subjective'
            transformed = 'objective'
        else: 
            voice = 'objective'
            transformed = 'subjective'
        self.res.config(text = 'Your text is detected to be ' + voice + '. We will therefore transform it to be more '
                        + transformed + '.')
        
        

root = tk.Tk()
root.title('Not a Fact Checker')
'''
tabparent = ttk.Notebook(root)

tab1 = ttk.Frame(tabparent)
tab2 = ttk.Frame(tabparent)

tabparent.add(tab1, text ='Tab 1')
tabparent.add(tab2, text = 'Tab 2')
tabparent.pack(expand=1,fill='both')
'''
        
app = Application(master=root)
#app['bg'] = '#545AA7'

root.mainloop()
