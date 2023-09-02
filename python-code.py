#breaking each dataset into different excel files
import os
import lasio
file = 1 # file number
f = open('well-'+str(file)+'.LAS')
f2 = open('well-'+str(file)+'-modified.LAS', 'a+')
isHeader = False
i = 0
for line in f.readlines():
    if line == '~Parameter Information Block\n':
        print("YES")
        i += 1
        f2.close()
        f2 = open('well-'+str(file)+'-modified-'+str(i)+'.LAS', 'a+')
    f2.write(line)
f2.close()
  
    
while i > 0:
    las = lasio.read('well-'+str(file)+'-modified-'+str(i)+'.LAS')
    las.df().to_excel('well-'+str(file)+'-modified-'+str(i)+'.xlsx')
    #os.remove('well-'+str(file)+'-modified-'+str(i)+'.LAS')
    i -= 1
#os.remove('well-'+str(file)+'-modified.LAS')


#removing extra lines
file = open("well-1.LAS", 'r')
start = False
end = True
for i in file.readlines():
    #print()
    i=i[:-1] # ---deletion of extra blank lines
    if ("~A" in i and (start == False)):
        start = True
        end = False
        continue
    if("~" in i and (start == True)):
        start = False 
        end = True
    if((start == True) and (end == False)):
        print(i)


#original 
file = open("well-1.LAS", 'r')
start = False
end = True
for i in file.readlines():
    #print()
    if ("~A" in i and (start == False)):
        start = True
        end = False
        continue
    if("~" in i and (start == True)):
        start = False 
        end = True
    if((start == True) and (end == False)):
        print(i)      



#headers/attributes added
file = open("well-1.LAS", 'r')
start = False
end = True
for i in file.readlines():
    #print()
    #i=i[:-1]
    if ("~A" in i and (start == False)):
        start = True
        end = False
    elif("~" in i and (start == True)):
        start = False 
        end = True
    if((start == True) and (end == False)):
        a=i.strip()
        a1=a.replace("~A", "")
        print(a1) 



#converting to integer and list
file = open("well-1.LAS", 'r')
start = False
end = True
for i in file.readlines():
    #print()
    #i=i[:-1]
    if ("~A" in i and (start == False)):
        start = True
        end = False
    elif("~" in i and (start == True)):
        start = False 
        end = True
    if((start == True) and (end == False)):
        a=i.strip()
        if ("~A" in a):
            a=a.replace("~A", "")
            a1=a.split()
        else :
            a=a.split()
            a1=[eval(i) for i in a]
        print(a1)



import pandas as pd
import numpy as np
file = open("well-1.LAS", 'r')
start = False
end = True
dt = {}
k = False
for i in file.readlines():
    #print()
    #i=i[:-1]
    if ("~A" in i and (start == False)):
        start = True
        end = False
    elif("~" in i and (start == True)):
        start = False 
        end = True
        if k == True:
            dt[temp] = lt
            temp = 0  
        
    if((start == True) and (end == False)):
        a = i.strip()
        if ("~A" in a):
            a = a.replace("~A", "")
            a1 = a.split()   
            temp = a1[1]
            lt = []
            k = True
            #list = list.append(a)
        else :
            a = a.split()
            a1 = [eval(i) for i in a] 
            #print(a1)
            lt.append(a1)
            #print(lt)
        
print(dt)




#modified
import pandas as pd
import numpy as np
file = open("well-1.LAS", 'r')
start = False
end = True
dt = {}
#lt = []
k = False
for i in file.readlines():
    #print()
    #i=i[:-1]
    if ("~A" in i and (start == False)):
        start = True
        end = False
    elif("~" in i and (start == True)):
        start = False 
        end = True
        if k == True:
            a = i.strip()
            a1 = a.split()
            if(len(a1)>2):
                for j in range(1,len(a1)):
                    lt = [a1[0], a1[j]]
                    dt[a1[j]].append(lt)
            else:
                lt = [a1[0], a1[1]]
                dt[a1[1]].append(lt)
            #dt[temp] = lt
            #temp = 0  
        
    if((start == True) and (end == False)):
        a = i.strip()
        if ("~A" in a):
            a = a.replace("~A", "")
            a1 = a.split()   
            #temp = a1[1]
            for j in range(1,len(a1)):
                dt[a1[j]] = []
            k = True
            lt = []
            #list = list.append(a)
        else :
            a = a.split()
            a1 = [eval(i) for i in a] 
            #print(a1)
            if(len(a1)>2):
                for j in range(1,len(a1)):
                    lt.extend([a1[0], a1[j]])
                    #lt = [a1[0], a1[j]]
                    #dt[a1[j]].append(lt)
            else:
                lt.append(a1)
            #print(lt)
        
print(dt)



print(dt.keys())


print(dt['LLD'])
DataFrame=pd.dataframe()


print(dt['RHOB'])


df = pd.DataFrame(data=dt)



#FINAL code including a single file
import pandas as pd
import numpy as np
files = ["well-1.LAS"]
Dict = {}
List = []
flag = 0
for filename in files:
    #Dict["well Number"] = {filename:[]}
    Dict[filename] = {}
    file = open(filename, 'r')
    for line in file.readlines():
        if("~A" in line and flag == 0):
            head = line.strip()
            head = head.replace("~A", "")
            head = head.split()
            flag = 1
            for i in range(1, len(head), 1):
                Dict[filename][head[i]]=[]
        elif(flag == 1):
            if("~" in line):
                flag = 0
            else:
                data = line.strip()
                data = data.split()
                if(len(head)>2):
                    for i in range(1, len(head), 1):
                        List = [data[0], data[i]]
                        Dict[filename][head[i]].append(List)
                else:
                    Dict[filename][head[1]].append(data)
                    
print(Dict.keys())
print(Dict) 
    


#1  *FINAL code including all well files*
import pandas as pd
import numpy as np
files = ["well-1.LAS","well-2.LAS","well-3.LAS","well-4.LAS","well-5.LAS"]
Dict = {}
List = []
flag = 0
for filename in files:
    #Dict["well Number"] = {filename:[]}
    Dict[filename] = {}
    file = open(filename, 'r')
    for line in file.readlines():
        if("~A" in line and flag == 0):
            head = line.strip()
            head = head.replace("~A", "")
            head = head.split()
            flag = 1
            for i in range(1, len(head), 1):
                Dict[filename][head[i]]=[]
        elif(flag == 1):
            if("~" in line):
                flag = 0
            else:
                data = line.strip()
                data = data.split()
                if(len(head)>2):
                    for i in range(1, len(head), 1):
                        List = [data[0], data[i]]
                        Dict[filename][head[i]].append(List)
                else:
                    Dict[filename][head[1]].append(data)
                    
print(Dict.keys())
print(Dict) 
    
# extra
import pandas as pd
df=pd.DataFrame.from_dict(Dict)
df

import pandas as pd
df=pd.DataFrame.from_dict(Dict["well-1.LAS"]["LLD"])
df
import pandas as pd
df=pd.DataFrame.from_dict(Dict["well-1.LAS"]["DT"])
df



#2
print(Dict["well-5.LAS"].keys())
df = pd.DataFrame(Dict["well-5.LAS"]["DT"], columns = ["DEPTH","DT"])
print(df)



#3
print(df.shape)
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(type(x))
print(x.shape)
x = x.astype(np.float64)
print(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



#4
print(x_train)
print(y_train)


#5
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


#6 predicting values
y_pred= regressor.predict([[2972.4081]])
print(y_pred)


#7
import numpy as np
import matplotlib.pyplot as plt
x_test.astype(np.float64)
y_test.astype(np.float64)
X_test=X_test[:100]
y_test=y_test[:100]
#X_test=X_test.transpose()
plt.figure(figsize=(20,20))
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.title('Depth Vs DT')
plt.xlabel("Depth")
plt.ylabel("DT")
plt.show()


#idk 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
files = ["well-1.LAS","well-2.LAS","well-3.LAS","well-4.LAS","well-5.LAS"]
Dict = {}
List = []
flag = 0
for filename in files:
    #Dict["well Number"] = {filename:[]}
    Dict[filename] = {}
    file = open(filename, 'r')
    for line in file.readlines():
        if("~A" in line and flag == 0):
            head = line.strip()
            head = head.replace("~A", "")
            head = head.split()
            flag = 1
            for i in range(1, len(head), 1):
                Dict[filename][head[i]]=[]
        elif(flag == 1):
            if("~" in line):
                flag = 0
            else:
                data = line.strip()
                data = data.split()
                if(len(head)>2):
                    for i in range(1, len(head), 1):
                        List = [data[0], data[i]]
                        Dict[filename][head[i]].append(List)
                else:
                    Dict[filename][head[1]].append(data)
                    
#print(Dict.keys())
#print(Dict) 

data = pd.DataFrame.from_dict(Dict)

for column in data.index.values:
    well_1_data = data.loc[column]['well-1.LAS']
    if type(well_1_data) is not list:
        well_1_data = []
    well_2_data = data.loc[column]['well-2.LAS']
    if type(well_2_data) is not list:
        well_2_data = []
    well_3_data = data.loc[column]['well-3.LAS']
    if type(well_3_data) is not list:
        well_3_data = []
    well_4_data = data.loc[column]['well-4.LAS']
    if type(well_4_data) is not list:
        well_4_data = []
    well_5_data = data.loc[column]['well-5.LAS']
    if type(well_5_data) is not list:
        well_5_data = []
    column_data = well_1_data + well_2_data + well_3_data + well_4_data + well_5_data
    column_data = np.array(column_data).astype(float)

    X = column_data[:, 0].reshape(-1, 1)
    Y = column_data[:, 1]

    model = LinearRegression()
    model.fit(X, Y)
    print(f"Predicting for depth {(X[-1] + 1)[0]}")
    print(f"{column} = {model.predict((X[-1] + 1).reshape(-1, 1))[0]}")
    print(f"RMSE = {mean_squared_error(Y, model.predict(X), squared=False)}")
    print("\n")




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
files=["well-1.LAS","well-2.LAS","well-3.LAS","well-4.LAS","well-5.LAS"]
dict={}
dd=[]
flg=0
for filename in files:
    #dict["WELL No"]={filename:[]}
    dict[filename]={}
    file=open(filename, 'r') 
    for line in file.readlines(): 
        if("~A" in line and flg==0): 
            head=line.strip() 
            head=head.replace("~A","") 
            head=head.split() 
            flg=1         
            for a in range(1,len(head),1): 
                dict[filename][head[a]]=[]           
        elif(flg==1): 
            if("~" in line):
                flg=0 
            else: 
                data=line.strip() 
                data=data.split()             
                if(len(head)>2):
                    for a in range(1,len(head),1): 
                        dd=[data[0], data[a]]                        
                        dict[filename][head[a]].append(dd)           
                else:
                    dict[filename][head[1]].append(data)
            
print(dict["well-5.LAS"].keys())   
df = pd.DataFrame(dict["well-1.LAS"]["DT"], columns=["DEPTH","DT"])
#print(df)
print(df.shape)
X=df.iloc[:,:-1].values
y=df.iloc[:, -1].values
print(type(X))
print(X.shape)
X=X.astype(np.float)
#print(X)
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict([[2000.156]])
print(y_pred)
X_test=X_test[:100]
y_test=y_test[:100]
#X_test=X_test.transpose()
plt.figure(figsize=(20,20))
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.title('Depth Vs DT')
plt.xlabel("Depth")
plt.ylabel("DT")
plt.show()

X_train=X_train[:100]
y_train=y_train[:100]
#X_test=X_test.transpose()
plt.figure(figsize=(20,20))
plt.scatter(X_train,y_train,color='black')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title('Depth Vs DT')
plt.xlabel("Depth")
plt.ylabel("DT")
plt.show()



data = pd.DataFrame.from_dict(Dict)
for column in data.index.values:
    well_1_data = data.loc[column]['well-1.LAS']
    if type(well_1_data) is not list:
        well_1_data = []
    well_2_data = data.loc[column]['well-2.LAS']
    if type(well_2_data) is not list:
        well_2_data = []
    well_3_data = data.loc[column]['well-3.LAS']
    if type(well_3_data) is not list:
        well_3_data = []
    well_4_data = data.loc[column]['well-4.LAS']
    if type(well_4_data) is not list:
        well_4_data = []
    well_5_data = data.loc[column]['well-5.LAS']
    if type(well_5_data) is not list:
        well_5_data = []
    column_data = well_1_data + well_2_data + well_3_data + well_4_data + well_5_data
    column_data = np.array(column_data).astype(float)

    X = column_data[:, 0].reshape(-1, 1)
    Y = column_data[:, 1]

    model = LinearRegression()
    model.fit(X, Y)
    print(f"Predicting for depth {(X[-1] + 1)[0]}")
    print(f"{column} = {model.predict((X[-1] + 1).reshape(-1, 1))[0]}")
    rmse = mean_squared_error(Y, model.predict(X), squared=False)
    print(f"RMSE = {rmse}")
    std = np.std(Y)
    print(f"STD = {std}")
    rmse_norm = rmse/std
    print(f"RMSE normalized = {rmse_norm}")
    print("\n")
    


    

