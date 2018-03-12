# coding:UTF-8


dic=[]
filepath="/media/hadoop/娱乐/A2/train.txt"
file = open(filepath) 
filename=[]
file_and_id={}

for line in file:
    #pass # do something
    #print(line)


    s_t=line.strip().split(" ")
    print(s_t[0])
    filename.append(s_t[0])
    s_t.pop(0)


#print(s_t)
    for s in s_t:
        #print(s)
        if s not in dic:
       
            dic.append(s)


file.close()

#add blank tag b
dic.append("B")
w=" "
for word in dic:
#     w=w+word
     print(dic.index(word))
     print(word)



file = open(filepath) 

print("dic length",len(dic))
#print(dic)
sentence_to_id=[]

s_to_id=[]

#s_to_id.append(dic.index("B"))

for line in file:
    #pass # do something
    #print(line)


    s_t=line.strip().split(" ")
    f_name=s_t[0]
    s_t.pop(0)
    sentence_to_id.append(dic.index("B"))
    for s in s_t:
    
          sentence_to_id.append(dic.index(s))
          #print(s)


   
    sentence_to_id.append(dic.index("B"))
    l=len(sentence_to_id)
    dis=0
    if l<32:
       dis=32-l
    for i in range(1,dis):
       sentence_to_id.append(dic.index("B"))
    #print(sentence_to_id)
    s_to_id.append(sentence_to_id)
    file_and_id[f_name]=sentence_to_id
    
    sentence_to_id=[]

file.close()

#s_to_id.append(dic.index("B"))
maxlength=0
minlength=100
for id_l in s_to_id:
 
    print(id_l)
    length=len(id_l)
    if length >maxlength:
        maxlength=length
    if minlength > length:
        minlength=length
print(maxlength)
print(minlength)
for id_l in s_to_id:
     length=len(id_l)
     if length<32:
        id_l.append(dic.index("B"))
for id_l in s_to_id:
     print(id_l)

print(file_and_id)

