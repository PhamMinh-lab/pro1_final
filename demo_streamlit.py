import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Using menu
st.title("Trung Tâm Tin Học")
menu = ["Home", "Capstone Project", "Sử dụng các điều khiển", "Gợi ý điều khiển project 1", "Gợi ý điều khiển project 2"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':    
    st.subheader("[Trang chủ](https://csc.edu.vn)")  
elif choice == 'Capstone Project':    
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")
    st.write("""### Có 2 chủ đề trong khóa học:    
    - Topic 1: Sentiment Analysis & Clustering
    - Topic 2: Recommender System  
    - ...""")
    # hiển thị các hình ảnh liên quan đến đồ án
    st.image("Sentiment_p1.png", width=300, caption="Sentiment Analysis & Clustering")
    st.image("recommend.png", width=300, caption="Recommender System")
    
elif choice == 'Sử dụng các điều khiển':
    # Sử dụng các điều khiển nhập
    # 1. Text
    st.subheader("1. Text")
    name = st.text_input("Enter your name")
    st.write("Your name is", name)
    # 2. Slider
    st.subheader("2. Slider")
    age = st.slider("How old are you?", 1, 100, 20)
    st.write("I'm", age, "years old.")
    # 3. Checkbox
    st.subheader("3. Checkbox")
    if st.checkbox("I agree"):
        st.write("Great!")
    # 4. Radio
    st.subheader("4. Radio")
    status = st.radio("What is your status?", ("Active", "Inactive"))
    st.write("You are", status)
    # 5. Selectbox
    st.subheader("5. Selectbox")
    occupation = st.selectbox("What is your occupation?", ["Student", "Teacher", "Others"])
    st.write("You are a", occupation)
    # 6. Multiselect
    st.subheader("6. Multiselect")
    location = st.multiselect("Where do you live?", ("Hanoi", "HCM", "Danang", "Hue"))
    st.write("You live in", location)
    # 7. File Uploader
    st.subheader("7. File Uploader")
    file = st.file_uploader("Upload your file", type=["csv", "txt"])
    if file is not None:
        st.write(file)    
    # 9. Date Input
    st.subheader("9. Date Input")
    date = st.date_input("Pick a date")
    st.write("You picked", date)
    # 10. Time Input
    st.subheader("10. Time Input")
    time = st.time_input("Pick a time")
    st.write("You picked", time)
    # 11. Display JSON
    st.subheader("11. Display JSON")
    json = st.text_input("Enter JSON", '{"name": "Alice", "age": 25}')
    st.write("You entered", json)
    # 12. Display Raw Code
    st.subheader("12. Display Raw Code")
    code = st.text_area("Enter code", "print('Hello, world!')")
    st.write("You entered", code)
    # Sử dụng điều khiển submit
    st.subheader("Submit")
    submitted = st.button("Submit")
    if submitted:
        st.write("You submitted the form.")
        # In các thông tin phía trên khi người dùng nhấn nút Submit
        st.write("Your name is", name)
        st.write("I'm", age, "years old.")
        st.write("You are", status)
        st.write("You are a", occupation)
        st.write("You live in", location)
        st.write("You picked", date)
        st.write("You picked", time)
        st.write("You entered", json)
        st.write("You entered", code)
          
elif choice == 'Gợi ý điều khiển project 1':
    st.write("Sentiment Analysis & Clustering")
    st.write("##### Dữ liệu mẫu")
    # Tạo dataframe có 3 cột là Id, Content, Sentiment
    data = {
        'Id': [1, 2, 3, 4, 5],
        'Content': ['Tôi yêu công việc của mình', 'Công ty này thật tuyệt vời', 'Tôi không thích sếp của mình', 'Mọi người ở đây rất thân thiện', 'Tôi cảm thấy không thoải mái khi làm việc ở đây'],
        'Sentiment': ['Positive', 'Positive', 'Negative', 'Positive', 'Negative']
    }
    df = pd.DataFrame(data)
    # In dataframe ra màn hình nhưng không in cột thứ tự tự động

    st.dataframe(df)
    # Tạo điều khiển cho phép người dùng nhập vào một câu để phân tích cảm xúc
    st.write("### 1. Phân tích cảm xúc")
    user_input = st.text_input("Nhập câu để phân tích cảm xúc")
    if user_input:
        # Giả sử phân tích cảm xúc đơn giản là kiểm tra xem câu có chứa từ "yêu" hay không
        if "yêu" in user_input or "tuyệt vời" in user_input or "thân thiện" in user_input:
            sentiment = "Positive"
        else:
            sentiment = "Negative"
        st.write("Cảm xúc của câu bạn nhập là:", sentiment)
        # Thêm câu đã nhập vào dataframe
        new_id = df['Id'].max() + 1
        new_row = {'Id': new_id, 'Content': user_input, 'Sentiment': sentiment}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.dataframe(df)        


elif choice=='Gợi ý điều khiển project 2':
    st.write("##### Gợi ý điều khiển project 2: Recommender System")
    st.write("##### Dữ liệu mẫu")
    # Tạo dataframe có 3 cột là Id, CompanyName, CompanyInfo
    data = {
        'Id': [1, 2, 3, 4, 5],
        'CompanyName': ['Công ty A', 'Công ty B', 'Công ty C', 'Công ty D', 'Công ty E'],
        'CompanyInfo': ['Công ty hóa mỹ phẩm ABC', 'Công ty mỹ phẩm BCD', 'Thông tin truyền thông C', 'Công ty điện tử D', 'Thông tin công nghệ và điện tử E']
    }
    df = pd.DataFrame(data)
    st.dataframe(df)
    st.write("### 1. Tìm kiếm công ty tương tự")
    # Tạo điều khiển để người dùng chọn công ty
    selected_company = st.selectbox("Chọn công ty", df['CompanyName'])
    st.write("Công ty đã chọn:", selected_company) 
    # Từ công ty đã chọn này, người dùng có thể xem thông tin chi tiết của công ty
    # hoặc thực hiện các xử lý khác
    # tạo điều khiển để người dùng tìm kiếm công ty dựa trên thông tin người dùng nhập
    search = st.text_input("Nhập thông tin tìm kiếm")
    # Tìm kiếm công ty dựa trên thông tin người dùng nhập vào search, chuyển thành chữ thường trước khi tìm kiếm
    result = df[df['CompanyInfo'].str.lower().str.contains(search.lower())]    
    # In danh sách công ty tìm được ra màn hình     
    st.write("Danh sách công ty tìm được:")
    st.dataframe(result)

    st.write("### 2. Recommend or Not")

    # Tạo một điều khiển combobox để người dùng chọn công ty, một lần chỉ chọn được một công ty
    selected_company_info = st.selectbox("Chọn công ty để xem thông tin chi tiết", df['CompanyName'])
    # Lấy thông tin chi tiết của công ty đã chọn
    company_info = df[df['CompanyName'] == selected_company_info]
    # Tạo 1 radio button list để người dùng chọn điểm số từ 1 đến 5 cho Salary & benefits    
    salary_benefits = st.radio("Chọn điểm số cho Salary & benefits", [1, 2, 3, 4, 5], horizontal=True)
  
    training_learning = st.radio("Chọn điểm số cho Training & learning", [1, 2, 3, 4, 5], horizontal=True)
    
    management_cares = st.radio("Chọn điểm số cho Management cares about me", [1, 2, 3, 4, 5], horizontal=True)

    culture_fun = st.radio("Chọn điểm số cho Culture & fun", [1, 2, 3, 4, 5], horizontal=True)

    office_workspace = st.radio("Chọn điểm số cho Office & workspace", [1, 2, 3, 4, 5], horizontal=True)

    # Tạo một button để người dùng submit các thông tin đã chọn
    submit = st.button("Submit")
    # Giả sử điểm trung bình của các lựa chọn >=4 thì sẽ hiển thị thông báo "Recommend", ngược lại sẽ hiển thị thông báo "Not Recommend"
    if submit:
        st.write("Bạn đã chọn công ty:", selected_company_info)
        st.write("Điểm số cho Salary & benefits:", salary_benefits)
        st.write("Điểm số cho Training & learning:", training_learning)
        st.write("Điểm số cho Management cares about me:", management_cares)
        st.write("Điểm số cho Culture & fun:", culture_fun)
        st.write("Điểm số cho Office & workspace:", office_workspace)
        # Tính điểm trung bình của các điểm số đã chọn
        average_score = (salary_benefits + training_learning + management_cares + culture_fun + office_workspace) / 5
        st.write("Điểm trung bình của các điểm số đã chọn là:", average_score)
        # Hiển thị thông báo Recommend or Not
        if average_score >= 3.5:
            st.write("#### --> Recommend")
        else:
            st.write("#### --> Not Recommend")
        
        # vẽ biểu đồ cột thể hiện điểm số của từng lựa chọn        
        scores = {
            'Salary & benefits': salary_benefits,
            'Training & learning': training_learning,
            'Management cares about me': management_cares,
            'Culture & fun': culture_fun,
            'Office & workspace': office_workspace
        }
        scores_df = pd.DataFrame(list(scores.items()), columns=['Category', 'Score'])
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Category', y='Score', data=scores_df, palette='viridis')
        plt.title('Scores for ' + selected_company_info)
        plt.xticks(rotation=45)
        st.pyplot(plt)
   
# Done
    
    
    
        

        
        

    



