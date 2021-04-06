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
        self.title = tk.Label(self, text='Objectivity Detector', font=('Roboto',30))
        self.title.grid(row=1,column=0,padx=10,pady=10,sticky=W)
        
        self.field = tk.Text(self)
        self.field.grid(row=2,column=0,padx=10,pady=5)
        
        self.btn = tk.Button(self, text='Detect Probability', command = self.detectObjectivity)
        self.btn.grid(row=3,column=0,padx=10,pady=3)
        
        self.upload = tk.Button(self, text = 'Upload .txt', command = self.uploadTxt)
        self.upload.grid(row=4,column=0,padx=10,pady=3)
        
        #TODO
        self.res = tk.Label(self, text='Results go here', font=('Consolas',13))
        self.res.grid(row=5,column=0,padx=10,pady=10,sticky=W)
        
        self.title2 = tk.Label(self, text='Objectivity-Subjectivity Conversion', font=('Consolas',30))
        self.title2.grid(row=1,column=1,padx=10,pady=10)
        
        self.field2 = tk.Text(self)
        self.field2.grid(row=2,column=1,padx=10,pady=5)
        
        self.btn2 = tk.Button(self, text='Convert text', command = self.detectObjectivity)
        self.btn2.grid(row=3,column=1,padx=10,pady=3)
        
        self.upload2 = tk.Button(self, text = 'Upload .txt', command = self.detectObjectivity)
        self.upload2.grid(row=4,column=1,padx=10,pady=3)
        
        
    def detectObjectivity(self):
        arr = model_direct.predict_proba([self.field.get(1.0,'end')])
        print(np.array(arr))
        self.res.config(text = 'Objective: ' + str(round(arr[0][0]*100,2)) + 
                        '%\nSubjective: ' + str(round(arr[0][1]*100,2)) + '%')
    def uploadTxt(self):
        tf = filedialog.askopenfilename(
            title="Open Text file", 
            filetypes=(("Text Files", "*.txt"),)
            )
        with open(tf, 'r') as file:
            data = file.read()
        self.field.insert(1.0,data)
        

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
