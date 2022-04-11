import tkinter as tk
import numpy as np
import random
import time
from numba import jit,njit,prange,cuda
from numba.typed import List
from math import cos,sin

class Demo:
    def __init__(self,root,t02=0,t03=0,t04=0,t05=0,t11=0,t12=0,t13=0,t14=0,t15=0,t21=0,t22=0,t23=0,t24=0,t25=0):
        self.root=root
        self.text=tk.Text(root)
        self.text.grid(row=0,column=0,padx=0,pady=0,ipadx=15,ipady=140,rowspan=8)
        self.text.insert(tk.INSERT,"Γειά σας! Πάτηστε πρώτα το κουμπί των οδηγιών 'Οδηγίες' για να ενημερωθείτε σχετικά με την χρήση αυτού του προγράμματος")
        self.text.grid()
        self.create_buttons()
        self.t02=t02        ##  t#1#2:Το πρώτο νούμερο (#1) παριστάνει τη στήλη και το δεύτερο (#2) τη γραμμή(για κάθε κουμπί)
        self.t03=t03
        self.t04=t04
        self.t05=t05
        self.t11=t11
        self.t12=t12
        self.t13=t13
        self.t14=t14
        self.t15=t15
        self.t21=t21
        self.t22=t22
        self.t23=t23
        self.t24=t24
        self.t25=t25

        
    #####CREATING BUTTONS#####
    def create_buttons(self):

        ##Πρώτη Στήλη
        self.label=tk.Label(root,text="Python-Math-Numpy")
        self.label.grid(row=0,column=1,ipadx=15,ipady=15)
        self.button=tk.Button(root,text="Math\n(n=10.000.000)",padx=1,pady=1,command=self.math)
        self.button.grid(row=1,column=1,ipadx=38,ipady=25)
        self.button=tk.Button(root,text="Numpy\n(n=10.000.000)",padx=1,pady=1,command=self.numpy)
        self.button.grid(row=2,column=1,ipadx=38,ipady=25)
        self.button=tk.Button(root,text="Numpy-Numba\nWith Compilation",padx=1,pady=1,command=self.numpy_numba_with_compilation)
        self.button.grid(row=3,column=1,ipadx=30,ipady=25)
        self.button=tk.Button(root,text="Numpy-Numba\nAfter Compilation",padx=1,pady=1,command=self.numpy_numba_after_compilation)
        self.button.grid(row=4,column=1,ipadx=30,ipady=25)
        self.label=tk.Label(root,text="")
        self.label.grid(row=5,column=1,ipadx=15,ipady=15)
        self.label=tk.Label(root,text="")
        self.label.grid(row=6,column=1,ipadx=15,ipady=15)
        self.button=tk.Button(root,text="Clear",padx=1,pady=1,fg="green",command=lambda: self.text.delete(1.0,tk.END))
        self.button.grid(row=7,column=1,ipadx=38,ipady=25)

        ##Δεύτερη Στήλη(Πρώτη Διαχωριστική Στήλη)
        self.label=tk.Label(root,bg="black")
        self.label.grid(row=0,column=2,rowspan=8,ipady=325)
        
        ##Τρίτη Στήλη
        self.label=tk.Label(root,text="Numba (Μικρό)")
        self.label.grid(row=0,column=3,ipadx=15,ipady=15)
        self.button=tk.Button(root,text="Python",padx=1,pady=1,command=self.python_1)
        self.button.grid(row=1,column=3,ipadx=72,ipady=32)
        self.button=tk.Button(root,text="With Compilation",padx=1,pady=1,command=self.with_compilation_1)
        self.button.grid(row=2,column=3,ipadx=44,ipady=32)
        self.button=tk.Button(root,text="After Compilation",padx=1,pady=1,command=self.after_compilation_1)
        self.button.grid(row=3,column=3,ipadx=43,ipady=32)
        self.button=tk.Button(root,text="With Parallel",padx=1,pady=1,command=self.with_parallel_1)
        self.button.grid(row=4,column=3,ipadx=58,ipady=32)
        self.button=tk.Button(root,text="After Parallel",padx=1,pady=1,command=self.after_parallel_1)
        self.button.grid(row=5,column=3,ipadx=57,ipady=32)
        self.button=tk.Button(root,text="Cuda",padx=1,pady=1,command=self.cuda_1)
        self.button.grid(row=6,column=3,ipadx=76,ipady=32)
        self.button=tk.Button(root,text="Οδηγίες",padx=1,pady=1,fg="blue",command=self.odigies)
        self.button.grid(row=7,column=3,ipadx=38,ipady=25)

        ##Τέταρτη Στήλη(Δεύτερη Διαχωριστική Στήλη)
        self.label=tk.Label(root,bg="black")
        self.label.grid(row=0,column=4,rowspan=8,ipady=325)
        
        ##Πέμπτη Στήλη    
        self.label=tk.Label(root,text="Numba (Μεγάλο)")
        self.label.grid(row=0,column=5,ipadx=15,ipady=15)
        self.button=tk.Button(root,text="Python",padx=1,pady=1,command=self.python_2)
        self.button.grid(row=1,column=5,ipadx=72,ipady=32)
        self.button=tk.Button(root,text="With Compilation",padx=1,pady=1,command=self.with_compilation_2)
        self.button.grid(row=2,column=5,ipadx=44,ipady=32)
        self.button=tk.Button(root,text="After Compilation",padx=1,pady=1,command=self.after_compilation_2)
        self.button.grid(row=3,column=5,ipadx=43,ipady=32)
        self.button=tk.Button(root,text="With Parallel",padx=1,pady=1,command=self.with_parallel_2)
        self.button.grid(row=4,column=5,ipadx=58,ipady=32)
        self.button=tk.Button(root,text="After Parallel",padx=1,pady=1,command=self.after_parallel_2)
        self.button.grid(row=5,column=5,ipadx=57,ipady=32)
        self.button=tk.Button(root,text="Cuda",padx=1,pady=1,command=self.cuda_2)
        self.button.grid(row=6,column=5,ipadx=76,ipady=32)
        self.button=tk.Button(root,text="Quit",padx=1,pady=1,fg="red",command=self.root.destroy)
        self.button.grid(row=7,column=5,ipadx=38,ipady=25)

    

    #####PYTHON-MATH-NUMPY#####                 #####ΕΝΤΟΣ_ΚΛΑΣΕΙΣ#####
    def math(self,n=10000000): ##n=Μέγεθος Λιστών                #####Αλλαγή n-πρώτη στήλη
        self.L1=[]
        self.L2=list(range(n))
        self.L3=list(range(n))

        self.start = time.time()

        for i in range(n):
            self.L1.append(random.randint(0,500))   ##γέμισμα της πρώτης λίστας με τυχαίους αριθμούς

        for i in range(n):
            self.L2[i] = sin(self.L1[i])    ##η 2η λίστα είναι τα ημίτονα των στοιχείων της 1ης

        for i in range(n):
            self.L3[i] = cos(self.L1[i])    ##η 3η λίστα είναι τα συνημίτονα των στοιχείων της 1ης
            
        self.end=time.time()
        self.time=self.end-self.start
        self.text.insert(tk.INSERT,"\nMath: ")
        self.text.insert(tk.END,self.time)
        self.text.insert(tk.INSERT," sec")
        self.text.grid()

    def numpy(self,n=10000000): ##n=Μέγεθος Λιστών                #####Αλλαγή n-πρώτη στήλη
        self.L1=np.random.randint(500,size=10000000)
        self.L2=list(range(n))
        self.L3=list(range(n))

        self.start = time.time()

        self.L2 =  np.sin(self.L1) ##η 2η λίστα είναι τα ημίτονα των στοιχείων της 1ης

        self.L3 = np.cos(self.L1) ##η 3η λίστα είναι τα συνημίτονα των στοιχείων της 1ης

        self.end=time.time()
        self.time=self.end-self.start
        self.text.insert(tk.INSERT,"\nNumPy: ")
        self.text.insert(tk.END,self.time)
        self.text.grid()

    def numpy_numba_with_compilation(self):
        self.text.insert(tk.INSERT,"\nNumPy-Numba: ")
        self.text.insert(tk.END,self.t02)
        self.text.grid()

    def numpy_numba_after_compilation(self):
        self.text.insert(tk.INSERT,"\nNumPy-Numba: ")
        self.text.insert(tk.END,self.t03)
        self.text.grid()

    def numpy_numba_with_parallel(self):
        self.text.insert(tk.INSERT,"\nNumPy-Numba: ")
        self.text.insert(tk.END,self.t04)
        self.text.grid()
        
    def numpy_numba_after_parallel(self):
        self.text.insert(tk.INSERT,"\nNumPy-Numba: ")
        self.text.insert(tk.END,self.t05)
        self.text.grid()


        
    #####PYTHON-NUMBA-COMPILE-PARALLEL-CUDA(ΜΙΚΡΌ)#####                #####ΕΝΤΟΣ_ΚΛΑΣΕΙΣ#####
    def python_1(self,n=1000): #n=Ποσότητα Αριθμών που θα κατανεμηθούν σε σειρά             #####Αλλαγή n-δεύτερη στήλη
        self.list=[]
        self.start=time.time()
        for i in range(n):
            self.list.append(random.randint(0,100))

        self.start=time.time()

        for i in range(len(self.list)):
            for j in range(0, (len(self.list))-i-1):
                if self.list[j] > self.list[j+1] :
                    self.list[j], self.list[j+1] = self.list[j+1], self.list[j]
            

        self.end=time.time()
        self.time=self.end-self.start                                                                        
        self.text.insert(tk.INSERT,"\nJust Python: ")
        self.text.insert(tk.END,self.time)
        self.text.grid()
        
    def with_compilation_1(self):
        self.text.insert(tk.INSERT,"\nWith Compilation: ")
        self.text.insert(tk.END,t11)
        self.text.grid()

    def after_compilation_1(self):
        self.text.insert(tk.INSERT,"\nAfter Compilation: ")
        self.text.insert(tk.END,t12)
        self.text.grid()
        
    def with_parallel_1(self):
        self.text.insert(tk.INSERT,"\nWith Compilation (Parallel): ")
        self.text.insert(tk.END,t13)
        self.text.grid()
        
    def after_parallel_1(self):
        self.text.insert(tk.INSERT,"\nAfter Compilation (Parallel): ")
        self.text.insert(tk.END,t14)
        self.text.grid()

    def cuda_1(self):
        self.text.insert(tk.INSERT,"\nUsing Cuda: ")
        self.text.insert(tk.END,t15)
        self.text.grid()

    def odigies(self):
        self.text.insert(tk.INSERT,"\n\nΠώς δουλεύει;")
        self.text.insert(tk.INSERT,"\nΥπάρχουν τρεις κατηγορίες (κάθε στήλη), στις οποίες κάθε μία συγκρίνει διαφορετικές τιμές.Η πρώτη στήλη φτιάχνει μία λίστα n-διάστασης την οποία γεμίζει με τυχαίους")
        self.text.insert(tk.INSERT,"\nαριθμούς,όπου n∈(0,500)(τυχαία όρια), της οποίας στη συνέχεια βρίσκει τα ημίτονα")
        self.text.insert(tk.INSERT,"\nκαι τα συνημίτονα σε δύο άλλες λίστες.Η δεύτερη και η τρίτη στήλη δημίουργουν μία")
        self.text.insert(tk.INSERT,"\nλίστα n-τυχαίων αριθμών, σε τυχαία σειρά την οποία στη συνέχεια κατανέμει με")
        self.text.insert(tk.INSERT,"\nαύξουσα σειρά.Η διαφορά μεταξύ δεύτερης-τρίτης στήλης είναι ότι η δεύτερη")
        self.text.insert(tk.INSERT,"\nχρησιμοποείται για μικρά n, σε αντίθεση με την τρίτη, της οποίας το n δεχέται")
        self.text.insert(tk.INSERT,"\nμεγάλες τιμές για είναι φανερή μεταξύ στα αποτελέσματα.")
        self.text.insert(tk.INSERT,"\n\nΠώς αλλάζουμε τις τιμές των 'n';")
        self.text.insert(tk.INSERT,"\nΓια την πρώτη στήλη αλλάζουμε τις τιμές στα δεξία τις ισότητας στις γραμμές #101,\n#124,#281")
        self.text.insert(tk.INSERT,"\nΓια την δεύτερη στήλη αλλάζουμε τις τιμές στα δεξία τις ισότητας στις γραμμές #167,#305")
        self.text.insert(tk.INSERT,"\nΓια την τρίτη στήλη αλλάζουμε τις τιμές στα δεξία τις ισότητας στις γραμμές #232,\n#371")
        self.text.insert(tk.INSERT,"\n\nΠΡΟΣΟΧΉ!!!")
        self.text.insert(tk.INSERT,"\nΤο πρόγραμμα καθυστερεί λίγα δευτερόλεπτα στην αρχή για να ανοίξει καθώς υπολογίζει κάποιες τιμές")
        self.text.insert(tk.INSERT,"\nΕπίσης, κάποια κουμπιά καθυστερούν να ανταποκριθούν καθώς υπολογίζουν εκείνη")
        self.text.insert(tk.INSERT,"\nτην στιγμή τις τιμές, οπότε παραμείνετε υπομονετικοί!")
        self.text.grid()

    #####PYTHON-NUMBA-COMPILE-PARALLEL-CUDA(ΜΕΓΑΛΟ)#####                #####ΕΝΤΟΣ_ΚΛΑΣΕΙΣ#####
    def python_2(self,n=10000): #n=Ποσότητα Αριθμών που θα κατανεμηθούν σε σειρά                #####Αλλαγή n-τρίτη στήλη
        self.list=[]
        self.start=time.time()
        for i in range(n):
            self.list.append(random.randint(0,100))


        for i in range(len(self.list)):
            for j in range(0, (len(self.list))-i-1):
                if self.list[j] > self.list[j+1] :
                    self.list[j], self.list[j+1] = self.list[j+1], self.list[j]
            

        self.end=time.time()
        self.time=self.end-self.start                                                                        
        self.text.insert(tk.INSERT,"\nJust Python: ")
        self.text.insert(tk.END,self.time)
        self.text.grid()
        
    def with_compilation_2(self):
        self.text.insert(tk.INSERT,"\nWith Compilation: ")
        self.text.insert(tk.END,t21)
        self.text.grid()

    def after_compilation_2(self):
        self.text.insert(tk.INSERT,"\nAfter Compilation: ")
        self.text.insert(tk.END,t22)
        self.text.grid()
        
    def with_parallel_2(self):
        self.text.insert(tk.INSERT,"\nWith Compilation (Parallel): ")
        self.text.insert(tk.END,t23)
        self.text.grid()
        
    def after_parallel_2(self):
        self.text.insert(tk.INSERT,"\nAfter Compilation (Parallel): ")
        self.text.insert(tk.END,t24)
        self.text.grid()

    def cuda_2(self):
        self.text.insert(tk.INSERT,"\nUsing Cuda: ")
        self.text.insert(tk.END,t25)
        self.text.grid()



#####PYTHON-MATH-NUMPY#####                 #####ΕΚΤΟΣ_ΚΛΑΣΕΙΣ#####

L1=np.random.randint(500,size=10000000)                         #####Αλλαγή n-πρώτη στήλη

@jit(nopython=True)
def numpy(): ##n=Μέγεθος Λιστών
    
        L2 = np.sin(L1) ##η 2η λίστα είναι τα ημίτονα των στοιχείων της 1ης

        L3 = np.cos(L1) ##η 3η λίστα είναι τα συνημίτονα των στοιχείων της 1ης

#Εύρεση Χρόνου t02       
start=time.time()
numpy()
end=time.time()
t02=end-start

#Εύρεση Χρόνου t03
start=time.time()
numpy()
end=time.time()
t03=end-start



#####PYTHON-NUMBA-COMPILE-PARALLEL-CUDA(ΜΙΚΡΟ)#####                #####ΕΚΤΟΣ_ΚΛΑΣΕΙΣ#####
mylist = np.random.randint(100, size=1000)                                  #####Αλλαγή n-δεύτερη στήλη
mylist1 = mylist
mylist2 = mylist
mylist3 = mylist
mylist4 = mylist
mylist5 = mylist


@jit(nopython=True)
def bubbleSort_fast(x):  
    for i in range(len(x)):
        for j in range(0, (len(x))-i-1):
            if x[j] > x[j+1] :
                x[j], x[j+1] = x[j+1], x[j]

#Εύρεση Χρόνου t11
start=time.time()
bubbleSort_fast(mylist)
end=time.time()
t11=end-start

#Εύρεση Χρόνου t12
start=time.time()
bubbleSort_fast(mylist1)
end=time.time()
t12=end-start



@njit(parallel=True)
def bubbleSort_fast_par(x):
    for i in prange(len(x)):
        for j in range(0, (len(x))-i-1):
            if x[j] > x[j+1] :
                x[j], x[j+1] = x[j+1], x[j]

#Εύρεση Χρόνου t13
start=time.time()
bubbleSort_fast_par(mylist2)
end=time.time()
t13=end-start

#Εύρεση Χρόνου t14
start=time.time()
bubbleSort_fast_par(mylist3)
end=time.time()
t14=end-start



@cuda.jit
def bubbleSort_fast_cuda(list):  
    for i in range(len(list)):
        for j in range(0, (len(list))-i-1):
            if list[j] > list[j+1] :
                list[j], list[j+1] = list[j+1], list[j]

#Εύρεση Χρόνου t15
start=time.time()
bubbleSort_fast_cuda(mylist5)
end=time.time()
t15=end-start



#####PYTHON-NUMBA-COMPILE-PARALLEL-CUDA(ΜΕΓΑΛΟ)#####                #####ΕΚΤΟΣ_ΚΛΑΣΕΙΣ#####
mylist = np.random.randint(100, size=10000)                                         #####Αλλαγή n-τρίτη στήλη
mylist1 = mylist
mylist2 = mylist
mylist3 = mylist
mylist4 = mylist
mylist5 = mylist


@jit(nopython=True)
def bubbleSort_fast(x):  
    for i in range(len(x)):
        for j in range(0, (len(x))-i-1):
            if x[j] > x[j+1] :
                x[j], x[j+1] = x[j+1], x[j]

#Εύρεση Χρόνου t21
start=time.time()
bubbleSort_fast(mylist)
end=time.time()
t21=end-start

#Εύρεση Χρόνου t22
start=time.time()
bubbleSort_fast(mylist1)
end=time.time()
t22=end-start



@njit(parallel=True)
def bubbleSort_fast_par(x):
    for i in prange(len(x)):
        for j in range(0, (len(x))-i-1):
            if x[j] > x[j+1] :
                x[j], x[j+1] = x[j+1], x[j]

#Εύρεση Χρόνου t23
start=time.time()
bubbleSort_fast_par(mylist2)
end=time.time()
t23=end-start

#Εύρεση Χρόνου t24
start=time.time()
bubbleSort_fast_par(mylist3)
end=time.time()
t24=end-start



@cuda.jit
def bubbleSort_fast_cuda(list):  
    for i in range(len(list)):
        for j in range(0, (len(list))-i-1):
            if list[j] > list[j+1] :
                list[j], list[j+1] = list[j+1], list[j]

#Εύρεση Χρόνου t25
start=time.time()
bubbleSort_fast_cuda(mylist5)
end=time.time()
t25=end-start



root=tk.Tk()
Demo(root,t02=t02,t03=t03,t11=t11,t12=t12,t13=t13,t14=t14,t15=t15,t21=t21,t22=t22,t23=t23,t24=t24,t25=t25)
root.mainloop()
