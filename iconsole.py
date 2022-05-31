# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:19:14 2022

@author: vedhs
"""
from shutil import get_terminal_size
from threading import Thread
from time import sleep,time,strftime,gmtime

'''
Console loading script
 
Credits: 
    Basic logic:    ted [https://stackoverflow.com/users/3867406/ted]
'''
class Loader:
    def __init__(self, desc="Loading...", end="Done!", timeout=0.5):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout
        self.steps = ['\\','|','/','-']
                
        self.maxl = 0
        self.done = False
        self.startTime=time()
        self.elapsedTime=''
        self._thread = Thread(target=self._animate, daemon=True).start()
        

    

    def _animate(self):
        i=0
        while True:
            
            if self.done:
                break                        
                    
            prstr=f'\rLoading  {self.steps[i]} {self.desc}\tTime Elapsed:{self.elapsedTime}'
            
            print(prstr+' '*(self.maxl-len(prstr)), flush=True, end="")
            if len(prstr)>self.maxl:
                self.maxl=len(prstr)
            sleep(self.timeout)
            self.elapsedTime=strftime('%Hh %Mm %Ss',gmtime(time()-self.startTime))
            i=(i+1)%len(self.steps)
    

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}\tTime taken:{self.elapsedTime}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()