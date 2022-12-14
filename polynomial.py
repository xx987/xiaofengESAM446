#!/usr/bin/env python
# coding: utf-8

# In[435]:


import numpy as np
import math
from sympy import *
import pprint
import re


# In[436]:


def str_split (s):
    
    first_split = s.split('+') # first split by '+'
    second_split = [  split.split('-')   for split in first_split] # then split by '-'
    for secon in second_split:
        length = len(secon)
        for i in range(1,length):
            secon[i] = '-'+secon[i]   # put back the '-'
    new_list = []
    for i in second_split:
        for j in i: 
            if j!='': # discard the first empty split
                new_list.append(j)
                
    
    j = 0
    while j<len(new_list):
        new_list[j] = new_list[j].replace(" ","")
        if new_list[j][0:2] == '-x':
            new_list[j] = "-1*" + new_list[j][1:]
        j = j+1
    
    
    return new_list




class Polynomial:
    def __init__(self,coefs):
        self.coefs = coefs
        self.order = len(self.coefs)-1
    @ staticmethod
    def from_string (s):
        
        monomial = str_split(s)
        order_coef = []
        for m in monomial:
            if ('x' not in m): # this is the constant term
                order_coef.append((0,(int(m))))
            else:
                search = re.search("\s?(-?\d?)\*?(x\^\d+|x)\s?",m) # look for something like (-7)*x^(5); search[1] returns -1, search[2] resturns 5
                coef,order = search[1],search[2]
                if order == 'x': order = 1
                else: order = order[-1]
                if coef == '': coef = 1
                order_coef.append(((int(order)),(int(coef))))
        order_coef = np.array(order_coef)
        order = max(order_coef[:,0])
        coefs = [0]*(int(order)+1)
        
        for oc in order_coef:
            order,coef = int(oc[0]),int(oc[1])
            coefs[order] = coef
        ##后加
        
            

        return Polynomial(coefs)
# addition, subtraction, multiplication, and equality.   
    def __add__(self,P2):
        coef = []
        order1,coefs1 = self.order, self.coefs
        order2,coefs2 = P2.order, P2.coefs
        P2_high = 0
        if order2> order1: P2_high = 1
        elif (P2_high):
            coef = [sum(x) for x in zip(coefs2, (coefs1+[0]*(order2 - order1)))]
        else:                      
            coef = [sum(x) for x in zip(coefs2+[0]*(order1 - order2), (coefs1))]
         
        if coef[-1] == 0:
            coef.remove(coef[-1])
        
        return Polynomial(list(coef))
    
    def __sub__(self, P2):
        co = [-1*x for x in P2.coefs]
        P3 = Polynomial(co)
        
        
        
        
        return self + P3
    
    
    def __eq__(self, P2):

        return self.coefs == P2.coefs
 
        
    
    def __mul__(self, P2):
        order1,coefs1 = self.order, self.coefs
        order2,coefs2 = P2.order, P2.coefs
        orderlist1 = list(range(0, order1 + 1))
        orderlist2 = list(range(0, order2 + 1))
        corl = []
        orrl = []
        for i in range(len(coefs1)):
            for j in range(len(coefs2)):
                cor = coefs1[i] * coefs2[j]
                ordr = orderlist1[i] + orderlist2[j]
        
                corl.append(cor)
                orrl.append(ordr)
        Coeff = corl
        order = orrl
        

        highest_ord = max(order)
        new_coeff = [0]*(highest_ord+1)
        for coef,orde in zip(Coeff,order):
            
            new_coeff[orde]+= coef
        coef = new_coeff
        
        

        return Polynomial(coef)
                
        
    def __truediv__(self, P2):
        order1,coefs1 = self.order, self.coefs
        order2,coefs2 = P2.order, P2.coefs
        numerator = Polynomial(coefs1)
        denominator = Polynomial(coefs2)
        return RationalPolynomial(numerator, denominator)
    
    #def div(self, P2):
        
        
    
    
    def __repr__(self):
        s = ''
        for i in range(len(self.coefs)):
            if i==0:
                s+=str(self.coefs[i] if self.coefs[i] != 1 else '' )+'+'
            else:
                s+= str(self.coefs[i] if self.coefs[i] != 1 else '')+'x'+'^'+str(i)+'+'
        return (s[:-1])   
    
            


# In[437]:


class RationalPolynomial:
        def __init__(self, numerator, denominator):
            self.numerator = numerator
            self.denominator = denominator
            #self._reduce()
        
        @staticmethod
        def from_string(string):
            spst = string.split(")/(")
            numerator = spst[0].replace(spst[0][0],'')
            denominator = spst[1].replace(spst[1][-1],'')
            return RationalPolynomial(Polynomial.from_string(numerator), Polynomial.from_string(denominator))
        
        def __add__(self, other):
            
            selfnu = self.numerator
            othnu =  other.numerator 
            selfde = self.denominator
            othde =  other.denominator
            hafnumo = othnu*(selfde) 
            hafnumt = selfnu*(othde)
            numerator = hafnumo + (hafnumt)
            denominator = selfde*(othde)
            return RationalPolynomial(numerator, denominator)
        
        
        def __sub__(self, other):
            nume = Polynomial.from_string(other.numerator)
            co = [-1*x for x in nume.coefs]
            upnumer = Polynomial(co)
            newpol = RationalPolynomial(upnumer, other.denominator)
            #return newpol.numerator.order
            
            
        def __eq__(self, other):
            
            return self.numerator*other.denominator == other.numerator*self.denominator
            
            
            
            

        
        def __mul__(self, other):
            selfnu = self.numerator
            othnu =  other.numerator 
            selfde = self.denominator
            othde =  other.denominator
            numerator = selfnu*(othernu)
            denominator = selfde*(othde)
            return RationalPolynomial(numerator, denominator)
        
        def __truediv__(self, other):
            selfnu = self.numerator
            othnu =  other.numerator 
            selfde = self.denominator
            othde =  other.denominator
            numerator = selfnu*(othde)
            denominator = selfde*(othernu)
            return RationalPolynomial(numerator, denominator)
            
                  
            


# In[438]:


a = RationalPolynomial(Polynomial.from_string('2+x'), Polynomial.from_string('-1+x+2*x^3'))
b = RationalPolynomial(Polynomial.from_string('x'), Polynomial.from_string('-1+x'))
a + b


# In[439]:


st1 = '-1+x+2*x^3'
st2 = '-1+x+2*x^3'
a = Polynomial.from_string(st1)
b = Polynomial.from_string(st2)


# In[440]:


a = Polynomial.from_string("-4 + x^2")
b = Polynomial.from_string("x^2 - 4")


# In[441]:


b = Polynomial.from_string("-3 - x^2 + 2*x^3")


# In[442]:


a = Polynomial.from_string("-3 -x -7*x^2")
b = Polynomial.from_string("1 + 7*x^2")
c = Polynomial.from_string("-2 - x")
t = a+b
t.coefs


# In[443]:


a = Polynomial.from_string("1 + 7*x^2")
b = Polynomial.from_string("-3 - x + 2*x^2")
c = Polynomial.from_string("-2 - x + 9*x^2")
print(a+b)
c


# In[444]:


a = Polynomial.from_string("-2 + 6*x^2 + 2*x^3")
b = Polynomial.from_string("-3 - x^2 + 2*x^3")
c = Polynomial.from_string("1 + 7*x^2")
print(a.coefs)
print(b.coefs)

co = [-1*x for x in b.coefs]
print(co)


# In[446]:


a = Polynomial.from_string("4")
b = Polynomial.from_string("2 - x + 3*x^2")
c = Polynomial.from_string("8 - 4*x + 12*x^2")
print(a*b)
print(c.coefs)


# In[ ]:





# In[ ]:




