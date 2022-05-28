# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:23:19 2022

@author: vedhs
"""
import tkinter as tk
import iproc
import threading
'''
Class to make Graphical User Interface

Credits:
    GUI frame logic     :   Stevoisiak [https://stackoverflow.com/users/3357935/stevoisiak]
    
GUI:    Makes parent GUI and opens train test split tab
    None
run:    Runs the mainloop  
    Args:
        None
    Return:
        None
        
'''
class GUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame=None

        #Init for UI elements to show on tk window
        self.UIe = {}
        
        #Label to show status
        self.status = tk.StringVar()
        self.UIe['status'] = tk.Label(self,textvariable= self.status)
        self.UIe['status'].grid(row=0,column=0)
        self.status.set("Choose what to do")
        
        #Buttons to switch frames
        self.UIe['setTrainTestFrame']=tk.Button(self,text='Split Dataset',command=lambda:self.switch_frame(TrainTestFrame)).grid(row=1,column=0)
        
        #Set initial frame to Train Test Frame
        self.switch_frame(TrainTestFrame)
        
    #Run the mainloop
    def run(self):
        self.mainloop()
        
    #Destroy the previous frame and adds new frame to the UI
    def switch_frame(self,frame):
        if self._frame is not None:
            self._frame.destroy()
        self._frame=frame(self)
        self._frame.grid(row=2,column=0)
        
        
'''
TrainTestFrame:     Makes the frame for splitting dataset into train and test dataset
    None
'''      
class TrainTestFrame(tk.Frame):
    #Constructor
    def __init__(self,master):        
        tk.Frame.__init__(self, master) 
        self.status = master.status
        #Init for UI elements to show on tk window
        self.UIe = {}
        
        self.px,self.py=0,0       
                        
        #Getting the data path from the user
        self.UIe['pathLabel'] = tk.Label(self, text = 'Path:')   
        self.UIe['trainPathLabel'] = tk.Label(self, text = 'Train path:')   
        self.UIe['testPathLabel'] = tk.Label(self, text = 'Test path:')
        self.UIe['splitRatioLabel'] = tk.Label(self, text = 'Split ratio:')        
          
        self.UIe['path'] = tk.Text(self,height = 1,width = 20)        
        self.UIe['trainPath'] = tk.Text(self,height = 1,width = 20)        
        self.UIe['testPath'] = tk.Text(self,height = 1,width = 20)
        self.UIe['splitRatio'] = tk.Text(self,height = 1,width = 20)
        
        self.UIe['pathLabel'].grid(row=0,column=0)
        self.UIe['path'].grid(row=0,column=1)
        self.UIe['trainPathLabel'].grid(row=1,column=0)
        self.UIe['trainPath'].grid(row=1,column=1)
        self.UIe['testPathLabel'].grid(row=2,column=0)
        self.UIe['testPath'].grid(row=2,column=1)                
        self.UIe['splitRatioLabel'].grid(row=3,column=0)
        self.UIe['splitRatio'].grid(row=3,column=1)
        
        #UI button to train test split button
        self.UIe['ttsplit'] = tk.Button(self,text = "Train Test Split",command = self.__runTrainTestSplit)
        self.UIe['ttsplit'].grid(row=4,column=0,columnspan=2)
                
        
    def __runTrainTestSplit(self):
        #Get the text from the text fields
        path=self.UIe['path'].get(1.0, "end-1c")
        trainPath=self.UIe['trainPath'].get(1.0, "end-1c")
        testPath=self.UIe['testPath'].get(1.0, "end-1c")
        splitRatio=self.UIe['splitRatio'].get(1.0, "end-1c")
        
        try:
            #validating the data
            splitRatio=float(splitRatio)
            if len(path)>0 and len(trainPath)>0 and len(testPath)>0:
                #Calling proc function with the given data and running it on different thread
                data=iproc.Data(info=True,path=path,changeColor=False)
                
                trainTestThread=threading.Thread(target=data.trainTestSplit,args=(trainPath, testPath, splitRatio,self.status,))
                trainTestThread.setDaemon(True)
                trainTestThread.start()                
            else:
                self.status.set('Fill path,train path and test path')
        except ValueError:
            self.status.set('Split ratio must be a number')        

if __name__=='__main__':
    gui=GUI()    
    gui.run()