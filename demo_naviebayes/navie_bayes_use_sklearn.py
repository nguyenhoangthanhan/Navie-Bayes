import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from builtins import list
from gensim.parsing.preprocessing import strip_non_alphanum, split_alphanum,\
    strip_short, strip_numeric
from scipy.sparse.coo import coo_matrix
from sklearn.metrics.classification import accuracy_score

lemmatizer = PorterStemmer() #Hàm xóa các stopworld

'''
    Dữ liệu chưa xử l lấy từ http://www2.aueb.gr/users/ion/data/enron-spam lấy 900 ham và 900 spam trong Enron1/ham và enron1/spam cho train
    và 900 ham tiếp theo trong thư mục enron1/ham và 600 spam tiếp theo trong enron1/spam
'''
tr = "train" # Tên của file tạo mới
te = "test"# Tên của file tạo mới
tr_raw_ham = "train/ham" #tên của thư mục train ham cần tiền xử lý
tr_raw_spam = "train/spam" #tên của thư mục train spam cần tiền xử lý
te_raw_ham = "test4/ham" # tên của thư mục test ham cần tiền xử lý
te_raw_spam = "test4/spam"# tên của thư mục test spam cần tiền xử lý
'''
    Trước tiên cần tạo thư mục tền là: training_Lemmatization và 2 thư mục con của nó là ham và spam để lưu 1800 thư mới sau khi tiền xử lý
    Tương tự cần tạo thư mục tền là: testing_Lemmatization và 2 thư mục con của nó là ham và spam để lưu 1500 thư mới sau khi tiền xử lý

'''
def raw_data_processing(file_path, folder_kind, str): # Hàm tiền xử lý dữ liệu
    emails = [os.path.join(file_path, f) for f in os.listdir(file_path)] #Hàm duyết tất cả các tệp trong cùng 1 thư mục
    i = 0
    for f in emails: # Duyệt từng tệp trong thư mục cần tiễn xử lý
        with open(f, "r+", encoding="latin1") as fi: #mở tệp
            text1 = fi.read() #Đọc tệp
            text1 = re.sub(r"http\S+", "", text1) #Xóa các đường dẫn (link)
            text1 = strip_non_alphanum(text1).lower().strip() #Xóa các từ không nằm trong bảng chữ cái đưa tất cả về chữ thường 
            text1 = split_alphanum(text1) #Tách biệt các từ vô nghĩa (chữa cái kết hợp số)
            text1 = strip_numeric(text1) #Xóa tất cả các số
            text1 = strip_short(text1, minsize=2) #Xóa tất cả các ký tự đứng 1 mình
        
            words = text1.split() #Tách tất cả các từ của 1 chuỗi thành từng phần tử của 1 mảng
            with open("%s/%s%d.txt"%(str, folder_kind, i), "w") as tr2: #Tạo thư mục và tệp mới đã qua xử lý nằm trong đường dẫn str/folder_kind.txt của project
                for r in words:
                    if r not in stopwords.words('english'): # Xóa tất cả các stopword Những từ xuất hiện thường xuyên như ‘and’, ‘the’, ‘of’, … được loại bỏ.
                        r = lemmatizer.stem(r) #Những từ có cùng ‘gốc’ được đưa về cùng loại. Ví dụ, ‘include’, ‘includes’, ‘included’ đều được đưa chung về ‘include’
                        tr2.write(r+" ") #Ghi từng từ đãng qua xử lý vào tệp mới
        i+=1


# raw_data_processing(tr_raw_ham, "ham", "training_Lemmatization/ham")
# raw_data_processing(tr_raw_spam, "spam", "training_Lemmatization/spam")
# raw_data_processing(te_raw_ham, "test_ham", "testing_Lemmatization/ham")
# raw_data_processing(te_raw_spam, "test_spam", "testing_Lemmatization/spam")

def make_Dictionary(path_ham, path_spam):
    email_hamss = [os.path.join(path_ham,f) for f in os.listdir(path_ham)]     
    all_words = []     #Mảng tất cả các từ   
    for f in email_hamss:    
        with open(f, "r") as m:
            textLine = m.read()
            textLine = textLine.split()
            for word in textLine: 
                all_words.append(word)
      
    email_spams = [os.path.join(path_spam,f) for f in os.listdir(path_spam)]   
    for f in email_spams:    
        with open(f, "r") as m:
            textLine = m.read()
            textLine = textLine.split()
            for word in textLine:
                all_words.append(word)
    
    vector_all_words = Counter(all_words)
    vector_all_words1 = vector_all_words.most_common(5000) #Lấy 5000 từ thường xuất hiện nhất
    new_vocabularys_common = [] #Tạo mảng từ vựng gồm tất cả các từ đểu khác nhau
    size_a = len(vector_all_words1)
    i = 0
    while i < size_a:
        new_vocabularys_common.append(list(vector_all_words1[i])[0])
        i += 1
    return new_vocabularys_common

list_vocabulary = make_Dictionary("training_Lemmatization/ham", "training_Lemmatization/spam")

'''
    (**) Tạo file với ma trận 3 dòng,1 dòng với 3 giá trị, thứ 1 là số thứ tự của thư, thứ 2 là vị trí từ trong danh sách từ vựng ở make_dictionary
, thứ 3 là số lần xuất hiện của từ đó 

    Đọc dữ liệu từ 1800 file đã qua tiền xử lý và được lưu lại trong thư mục "training_Lemmatization/ham" và "training_Lemmatization/spam"
để tạo dữ liệu cho file train_features.txt cấu trúc như (**) với 1800 file
    Đọc dữ liệu từ 1500 file đã qua tiền xử lý và được lưu lại trong thư mục "training_Lemmatization/ham" và "training_Lemmatization/spam"
để tạo dữ liệu cho file test_features.txt cấu trúc như (**) với 1500 file

'''
def create_features(path_ham, path_spam, matrix_output, labels_output): 
    email_hamss = [os.path.join(path_ham,f) for f in os.listdir(path_ham)]  
    i = 0
    for f in email_hamss:    
        words = []
        with open(f, "r") as m:
            textLine = m.read()
            textLine = textLine.split()
            
            for w in textLine:
                if w in list_vocabulary:
                    words.append(w)
            vetor_words = Counter(words)
            j = 0
            lst = list(vetor_words.keys())
            for w in list_vocabulary:
                if w in lst:
                    with open(matrix_output, "r+") as train_features:
                        train_features.read()
                        count = 0;
                        for k in range(0, len(words) - 1, 1):
                            if words[k] == w:
                                count += 1
                        train_features.write("%d %d %d\n" % (i,list_vocabulary.index(w), count))
                j += 1
                
            with open(labels_output, "r+") as train_labels:
                train_labels.read()
                train_labels.write("0 ")
            
        i += 1         
      
    email_spams = [os.path.join(path_spam,f) for f in os.listdir(path_spam)]   
    for f in email_spams:    
        words = []
        with open(f, "r") as m:
            textLine = m.read()
            textLine = textLine.split()
            
            for w in textLine:
                if w in list_vocabulary:
                    words.append(w)
            vetor_words = Counter(words)
            j = 0
            lst = list(vetor_words.keys())
            for w in list_vocabulary:
                if w in lst:
                    with open(matrix_output, "r+") as train_features:
                        train_features.read()
                        count = 0;
                        for k in range(0, len(words) - 1, 1):
                            if words[k] == w:
                                count += 1
                        train_features.write("%d %d %d\n" % (i,list_vocabulary.index(w), count))
                j += 1
                
            with open(labels_output, "r+") as train_labels:
                train_labels.read()
                train_labels.write("1 ")
            
        i += 1

# create_features("training_Lemmatization/ham", "training_Lemmatization/spam","train-features.txt","train-labels.txt")
# create_features("testing_Lemmatization/ham", "testing_Lemmatization/spam", "test-features.txt","test-labels.txt")

'''
    1800 thư train với 900 thư đầu tiên là thư thường gán nhãn là 0, và 900 thư sau là thư rác gán nhãn là 1
    Tương tự với 1500 thư test với 900 thư đầu tiên là thư thường gán nhãn là 0, và 600 thư sau là thư rác gán nhãn là 1
'''
        
labels_train = []
i = 0
for i in range(1800):
    if i < 900:
        labels_train.append(0)
    else:
        labels_train.append(1)
  
labels_test = []
i = 0
for i in range(1500):
    if i < 900:
        labels_test.append(0)
    else:
        labels_test.append(1) 
'''
    Đọc dữ liệu từ file sau khi xử lý và lưu trong Project
'''
def read_data(data_fn, labels_fn):
    with open(data_fn) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    
    dat = np.zeros((len(content), 3), dtype = int)
    
    for i, line in enumerate(content):
        a = line.split(' ')
        dat[i, :] = np.array([int(a[0]), int(a[1]), int(a[2])])
        
    data = coo_matrix((dat[:, 2], (dat[:, 0], dat[:, 1])),\
             shape=(len(labels_fn), 5000))
    return (data, labels_fn)
    
'''
    Test với các mô hình trong thư viện sklearn
'''
(train_data, train_label) = read_data("train-features.txt", labels_train)
(test_data, test_label) = read_data("test-features.txt", labels_test)
 
clf = MultinomialNB() 
clf.fit(train_data, train_label)
y_pred = clf.predict(test_data)
print('Training size = %d, accuracy = %.2f%%' % \
      (train_data.shape[0],accuracy_score(test_label, y_pred)*100))

clf = BernoulliNB(binarize = .5)
clf.fit(train_data, train_label)
y_pred = clf.predict(test_data)
print('Training size = %d, accuracy = %.2f%%' % \
      (train_data.shape[0],accuracy_score(test_label, y_pred)*100))
