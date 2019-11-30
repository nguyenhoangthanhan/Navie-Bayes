import os
from collections import Counter
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from builtins import list
from gensim.parsing.preprocessing import strip_non_alphanum, split_alphanum,\
    strip_short, strip_numeric

lemmatizer = PorterStemmer()

'''
    Thực ra bài này em làm theo cách tính trên trang web: https://machinelearningcoban.com/2017/08/08/nbc/
    phần Ví dụ. Do em chưa hiểu cách mô tả thuật toán và mô tả các đại lượng giá trị nên còn rất nhiều chỗ chú thích không rõ ràng và
xác suất thống kê em cũng không được giỏi và siêng năng học, học theo kiểu học vẹt chỉ giải được bài tập theo ví dụ chứ chưa hiểu được kiến thức và lý thuyết
nên việc code thuật toán này khá khó khăn và khó hiểu mong thầy thông cảm ạ!

'''

'''
    Dữ liệu chưa xử l lấy từ http://www2.aueb.gr/users/ion/data/enron-spam lấy 900 ham và 900 spam trong Enron1/ham và enron1/spam cho train
    và 900 ham tiếp theo trong thư mục enron1/ham và 600 spam tiếp theo trong enron1/spam
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
            with open("%s/%s%d.txt"%(str, folder_kind, i), "w") as tr2: #Tạo thư mục và tệp mới đã qua xử lý
                for r in words:
                    if r not in stopwords.words('english'): # Xóa tất cả các stopword Những từ xuất hiện thường xuyên như ‘and’, ‘the’, ‘of’, … được loại bỏ.
                        r = lemmatizer.stem(r) #Những từ có cùng ‘gốc’ được đưa về cùng loại. Ví dụ, ‘include’, ‘includes’, ‘included’ đều được đưa chung về ‘include’
                        tr2.write(r+" ") #Ghi từng từ đãng qua xử lý vào tệp mới
        i+=1

tr_raw_ham = "train/ham" #tên của thư mục train ham cần tiền xử lý
tr_raw_spam = "train/spam" #tên của thư mục train spam cần tiền xử lý
te_raw_ham = "test4/ham" # tên của thư mục test ham cần tiền xử lý
te_raw_spam = "test4/spam"# tên của thư mục test spam cần tiền xử lý

# raw_data_processing(tr_raw_ham, "ham", "training_Lemmatization/ham")
# raw_data_processing(tr_raw_spam, "spam", "training_Lemmatization/spam")
# raw_data_processing(te_raw_ham, "test", "testing_Lemmatization/ham")
# raw_data_processing(te_raw_spam, "test", "testing_Lemmatization/spam")
# raw_data_processing("test4_mix_ham_spam", "test", "testing2_Lemmatization_mix_ham_spam")



def make_Dictionary(path_ham, path_spam):
    emails = [os.path.join(path_ham,f) for f in os.listdir(path_ham)]    
    all_words = []     #Mảng tất cả các từ   
    words_spam = []   #Mảng tất cả các từ trong tất cả các tệp spam
    words_ham = []  #Mảng tất cả các từ trong tất cả các tệp ham
    for f in emails:    
        with open(f, "r") as m:
            textLine = m.read()
            textLine = textLine.split()
            for word in textLine: # Thêm các từ vào mảng tất cả các từ và mảng tất cả các từ ham
                all_words.append(word)
                words_ham.append(word)
    
    emails = [os.path.join(path_spam,f) for f in os.listdir(path_spam)]    
    for f in emails:    
        with open(f, "r") as m:
            textLine = m.read()
            textLine = textLine.split()
            for word in textLine:# Thêm các từ vào mảng tất cả các từ và mảng tất cả các từ ham
                all_words.append(word)
                words_spam.append(word)
    
    vector_all_words = Counter(all_words) #Gộp các từ giống nhau thành 
    vector_all_words1 = vector_all_words.most_common(3000) #Lấy 4500 từ thường xuất hiện nhất
    
    new_vocabularys_common = [] #Tạo mảng từ vựng gồm tất cả các từ đểu khác nhau
    size_a = len(vector_all_words1)
    i = 0
    while i < size_a:
        new_vocabularys_common.append(list(vector_all_words1[i])[0])
        i += 1
    
    i = 0
    while i < len(words_ham): #Xóa các từ không còn trong vector_all_words ở words_ham and words_spam
        if words_ham[i] not in new_vocabularys_common:
            words_ham.pop(i)
        else:
            i += 1
            
            
    i = 0
    while i < len(words_spam): #Xóa các từ không còn trong vector_all_words ở words_ham and words_spam
        if words_spam[i] not in new_vocabularys_common:
            words_spam.pop(i)
        else:
            i += 1 
        
    vector_words_ham = [0 for i2 in range(len(new_vocabularys_common))] #Tạo mảng gồm tất cả các từ đểu bằng 0 mảng này tương là tổng số lượng riêng từng từ của cả thư mục ham
    vector_words_spam = [0 for j2 in range(len(new_vocabularys_common))]#Tạo mảng gồm tất cả các từ đểu bằng 0 mảng này tương là tổng số lượng riêng từng từ của cả thư mục spam
    for w in words_ham: #Công doạn tính tổng
        if w in new_vocabularys_common:
            vector_words_ham[new_vocabularys_common.index(w)] += 1
    for w in words_spam:
        if w in new_vocabularys_common:
            vector_words_spam[new_vocabularys_common.index(w)] += 1
    
    size_of_ham_words_predict = len(words_ham) + len(new_vocabularys_common) + 0 #là mẫu số của λ trong lý thuyết thuật toán của ham
    size_of_spam_words_predict = len(words_spam) + len(new_vocabularys_common) + 0 #là mẫu số của λ trong lý thuyết thuật toán của spam
    
    train_predict_ham = [0 for i1 in range(len(new_vocabularys_common))]
    train_predict_spam = [0 for j1 in range(len(new_vocabularys_common))]
        
    for i in range(len(new_vocabularys_common)):
        train_predict_ham[i] = float((vector_words_ham[i]+1)*70000/size_of_ham_words_predict) # là λ của ham
        train_predict_spam[i] = float((vector_words_spam[i]+1)*70000/size_of_spam_words_predict) # là λ của spam
    
    return train_predict_ham, train_predict_spam, new_vocabularys_common


train_predict_ham, train_predict_spam, new_vocabularys_common = make_Dictionary("training_Lemmatization/ham", "training_Lemmatization/spam")

def test_data_processing(test_files):
    train_predict_ham, train_predict_spam, new_vocabularys_common = make_Dictionary("training_Lemmatization/ham", "training_Lemmatization/spam") #Gọi hàm
    size_vocabulary = len(new_vocabularys_common) #Kích cỡ của mảng tất cả các từ khác nhau
    size_of_labels = 1800 #Số lượng tệp train
    size_of_labels_ham = 900 #Số lượng tệp ham train
    size_of_labels_spam = 900 #Số lượng tệp spam train
    
    
    
    file_tests = [os.path.join(test_files, f_ham) for f_ham in os.listdir(test_files)] #Chạy tất cả các tệp của thư mục test 
    i = 0
    consequence_test = [] #Kết quả test và nhãn "ham" hoặc "spam" tương ứng
    
    for f in file_tests:
        text_test = [] #Mạng tất cả các từ trong 1 tệp
        p_ham = float(size_of_labels_ham/size_of_labels) #Tỉ lệ số thư ham trên tổng số thư 1/2
        p_spam = float(size_of_labels_spam/size_of_labels) #Tỉ lệ số thư spam trên tổng số thư 1/2
        with open(f, "r") as f_test:
            text = f_test.read()
            text_test = text.split()
        token_mail_test_in_voca = [] 
        for w in text_test:
            if (w in new_vocabularys_common):
                token_mail_test_in_voca.append(w)
        word_test_mail_vector = [0 for i in range(size_vocabulary)] #Mảng chứa số các từ trong 1 tệp mà các từ nằm trong vocabulary với vị trị ứng với vị trí từ của vocabulary
        for w in token_mail_test_in_voca:#Công doạn tính tổng
            word_test_mail_vector[new_vocabularys_common.index(w)] += 1
        
        for i in range(size_vocabulary): #Tìm 2 đại lượng tỉ lệ thuận với xác xuất
            if word_test_mail_vector[i] != 0:
                try:
                    p_ham *= float(train_predict_ham[i]**word_test_mail_vector[i])
                    p_spam *= float(train_predict_spam[i]**word_test_mail_vector[i])    
                except OverflowError:
                    pass
             
        predict_ham = float(p_ham/(p_ham + p_spam)) #Tỉ lệ email là ham
        predict_spam = float(p_spam/(p_ham + p_spam)) #Tỉ lệ email là spam
        print("Predict email is ham : %0.9f and is spam : %0.9f" % (predict_ham, predict_spam))
        if predict_ham >= predict_spam:
            consequence_test.append("ham")
        else:
            consequence_test.append("spam")    
        
        i += 1
    
    consequence_test_set = Counter(consequence_test)
    print(consequence_test_set) 

test_data_processing("testing2_Lemmatization_mix_ham_spam")