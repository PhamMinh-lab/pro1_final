import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from underthesea import pos_tag, sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
import regex
from sklearn.cluster import KMeans
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.metrics import  classification_report
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

def safe_vectorizer_fit(texts, min_df=2, max_df=0.95):
    n_docs = len(texts)
    if n_docs < min_df:
        min_df = 1
        max_df = 1.0
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    return vectorizer.fit_transform(texts), vectorizer


# ----------------- Sidebar -----------------
st.sidebar.title("🔧 Settings")
option = st.sidebar.selectbox("Choose a category", ["Dashboard", "Project Process", "C"], key="main_selectbox")
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍🎓 Group Members")
st.sidebar.markdown("1. Pham Nhat Minh  \n2. Vo Quoc Hung")

# ----------------- Dashboard -----------------
if option == "Dashboard":
    st.title("📝 Company Feedback Sentiment Analyzer")
    st.image("OIP.jfif")

    # Load feedback data
    @st.cache_data
    def load_data():
        return pd.read_excel('Reviews.xlsx')

    df = load_data()

    # Preprocessing: drop rows with all NaNs and create combined text column
    df = df.dropna(how='all')
    df['text'] = (
        df['Title'].fillna('') + ' - ' +
        df['What I liked'].fillna('') + ' - ' +
        df['Suggestions for improvement'].fillna('')
    )
    # #     # ========== Load resources ==========
    # def load_dicts_and_lists():
    #     with open('emojicon.txt', 'r', encoding="utf8") as f:
    #         emoji_dict = dict(line.split('\t') for line in f.read().split('\n') if line.strip())

    #     with open('teencode.txt', 'r', encoding="utf8") as f:
    #         teen_dict = dict(line.split('\t') for line in f.read().split('\n') if line.strip())

    #     with open('english-vnmese.txt', 'r', encoding="utf8") as f:
    #         for line in f.read().split('\n'):
    #             if line.strip():
    #                 key, value = line.split('\t')
    #                 teen_dict[key] = value  # merge into teen_dict

    #     with open('wrong-word.txt', 'r', encoding="utf8") as f:
    #         wrong_lst = f.read().split('\n')

    #     with open('vietnamese-stopwords.txt', 'r', encoding="utf8") as f:
    #         stopwords_lst = f.read().split('\n')

    #     return emoji_dict, teen_dict, wrong_lst, stopwords_lst

    # emoji_dict, teen_dict, wrong_lst, stopwords_lst = load_dicts_and_lists()

    # # # ========== Preprocessing functions ==========
    # def loaddicchar():
    #     uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđ..."  # trimmed for brevity
    #     unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeed..."  # trimmed for brevity
    #     dic = {}
    #     char1252 = 'à|á|ả|...'.split('|')
    #     charutf8 = "à|á|ả|...".split('|')
    #     for i in range(len(char1252)):
    #         dic[char1252[i]] = charutf8[i]
    #     return dic

    # def covert_unicode(txt):
    #     dicchar = loaddicchar()
    #     return regex.sub(
    #         r'à|á|ả|...|Ỵ', lambda x: dicchar.get(x.group(), x.group()), txt
    #     )

    # def process_special_word(text):
    #     new_text = ''
    #     text_lst = text.split()
    #     i = 0
    #     while i < len(text_lst):
    #         word = text_lst[i]
    #         if word == 'không' and i + 1 < len(text_lst):
    #             word = word + '_' + text_lst[i + 1]
    #             i += 2
    #         else:
    #             i += 1
    #         new_text += word + ' '
    #     return new_text.strip()

    # def process_postag_thesea(text):
    #     new_document = ''
    #     for sentence in sent_tokenize(text):
    #         sentence = sentence.replace('.', '')
    #         lst_word_type = ['A', 'AB', 'V', 'VB', 'VY', 'R']
    #         words = pos_tag(process_special_word(word_tokenize(sentence, format="text")))
    #         sentence = ' '.join(word[0] if word[1].upper() in lst_word_type else '' for word in words)
    #         new_document += sentence + ' '
    #     return regex.sub(r'\s+', ' ', new_document).strip()

    # def remove_stopword(text, stopwords):
    #     return regex.sub(r'\s+', ' ', ' '.join('' if word in stopwords else word for word in text.split())).strip()

    # def process_text(text, dict_emoji, dict_teen, lst_wrong):
    #     document = text.lower()
    #     document = document.replace("’", '')
    #     document = regex.sub(r'\.+', ".", document)
    #     document = re.sub(r'(.)\1+', r'\1', text)
    #     new_sentence = ''
    #     for sentence in sent_tokenize(document):
    #         sentence = ''.join(dict_emoji.get(word, word) + ' ' if word in dict_emoji else word for word in list(sentence))
    #         sentence = ' '.join(dict_teen.get(word, word) for word in sentence.split())
    #         pattern = r'(?i)\b[\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
    #         sentence = ' '.join(regex.findall(pattern, sentence))
    #         sentence = ' '.join('' if word in lst_wrong else word for word in sentence.split())
    #         new_sentence += sentence + '. '
    #     return regex.sub(r'\s+', ' ', new_sentence).strip()

    # # ========== Apply preprocessing on the text column ==========
    # df['clean'] = df['text'].apply(lambda x: remove_stopword(
    #     process_postag_thesea(
    #         process_text(
    #             covert_unicode(x), emoji_dict, teen_dict, wrong_lst
    #         )
    #     ), stopwords_lst
    # ))



    positive_words = [
    "thích", "tốt", "xuất sắc", "tuyệt vời", "ổn",
    "hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "nhanh",
    "thân thiện", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp",
    "an tâm", "thúc đẩy", "cảm động", "nổi trội",
    "sáng tạo", "phù hợp", "tận tâm", "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận",
    "vui vẻ", "hào hứng", "đam mê", 'chuyên', 'cảm', 'dễ', 'giỏi', 'hay', 'hiệu', 'hài', 'hỗ trợ', 'nhiệt tình',
    'sáng tạo', 'thân', 'thích', 'tuyệt', 'tốt', 'vui', 'ổn'
    ]
    negative_words = [
        "kém", "tệ", "buồn", "chán", "không dễ chịu", "không thích", "không ổn", "áp lực", "chán", "mệt",
        "không hợp", "không đáng tin cậy", "không chuyên nghiệp",
        "không thân thiện", "không tốt", "chậm", "khó khăn", "phức tạp",
        "khó chịu", "gây khó dễ", "rườm rà", "tồi tệ", "khó xử", "không thể chấp nhận", "không rõ ràng",
        "rối rắm", 'không hài lòng', 'không đáng', 'quá tệ', 'rất tệ', "phiền phức",
        'thất vọng', 'tệ hại', 'kinh khủng', 'chán', 'drama', 'dramas', 'gáp', 'gắt',
        'kém', 'lỗi', 'mệt', 'ngắt', 'quái', 'quát', 'rối', 'thiếu', 'trễ', 'tệ', 'tệp', 'tồi', 'áp', 'đáp', "hách dịch",
        'khó', 'không', 'chê', 'áp_lực', 'phân_biệt', 'lâu', 'ít', 'cực',
        'ép', 'thiếu', 'mệt', 'nhiều_việc', 'không_hài_lòng', 'xếp', 'lạnh', 'chậm', 'bất_công'
    ]
    positive_emojis = [
        "😄", "😃", "😀", "😁", "😆",
        "😅", "🤣", "😂", "🙂", "🙃",
        "😉", "😊", "😇", "🥰", "😍",
        "🤩", "😘", "😗", "😚", "😙",
        "😋", "😛", "😜", "🤪", "😝",
        "🤗", "🤭", "🥳", "😌", "😎",
        "🤓", "🧐", "👍", "🤝", "🙌", "👏", "👋",
        "🤙", "✋", "🖐️", "👌", "🤞",
        "✌️", "🤟", "👈", "👉", "👆",
        "👇", "☝️", "💚", "💖"
    ]
    negative_emojis = [
        "😞", "😔", "🙁", "☹️", "😕",
        "😢", "😭", "😖", "😣", "😩",
        "😠", "😡", "🤬", "😤", "😰",
        "😨", "😱", "😪", "😓", "🥺",
        "😒", "🙄", "😑", "😬", "😶",
        "🤯", "😳", "🤢", "🤮", "🤕",
        "🥴", "🤔", "😷", "🙅‍♂️", "🙅‍♀️",
        "🙆‍♂️", "🙆‍♀️", "🙇‍♂️", "🙇‍♀️", "🤦‍♂️",
        "🤦‍♀️", "🤷‍♂️", "🤷‍♀️", "🤢", "🤧",
        "🤨", "🤫", "👎", "👊", "✊", "🤛", "🤜",
        "🤚", "🖕"
    ]

    def find_words(document, list_of_words):
        document_lower = document.lower()
        word_count = 0
        word_list = []

        for word in list_of_words:
            if word in document_lower:
                # print(word)
                word_count += document_lower.count(word)
                word_list.append(word)

        return word_count, word_list
    
    df['sentiment'] = np.where(df['Rating'] >= 4 ,2, np.where(df['Rating'] >= 3,1,0 ))

    # Tách feature và label
    X = df.drop(columns=['sentiment'])
    y = df['sentiment']

    # Khởi tạo oversampler – chỉ oversample lớp 0- 2000, 1 - 3000
    oversample = RandomOverSampler(sampling_strategy={0: 2000, 1: 3000}, random_state=42)

    # Tạo dữ liệu mới sau khi oversample
    X_resampled, y_resampled =  oversample.fit_resample(X, y)

    vectorizer1 = CountVectorizer(max_df=0.95, min_df=2)
    doc_term_matrix1 = vectorizer1.fit_transform(X_resampled["text"])

    X1 = doc_term_matrix1
    y1= y_resampled

    X_train1 , X_test1 , y_train1, y_test1 = train_test_split(X1, y1, random_state=42)
        
    nvmodel = MultinomialNB()
    nvmodel.fit(X_train1, y_train1)
    y_pred_nv = nvmodel.predict(X_test1)

    def analyze_company_topics(company_id=None, n_topics=5):
        # Lấy danh sách id hợp lệ
        valid_ids = sorted(df['id'].unique())
        print("Danh sách id công ty hợp lệ:", valid_ids)

        # Nếu chưa truyền vào, yêu cầu nhập
        if company_id is None:
            try:
                company_id = int(input("Nhập company id: "))
            except ValueError:
                print("Vui lòng nhập số nguyên cho company id.")
                return

        if company_id not in valid_ids:
            print(f"Không tìm thấy công ty với id: {company_id}")
            return

        data_cty = df[df['id'] == company_id].reset_index(drop=True)
        company_name = data_cty['Company Name'].iloc[0]
        print(f"Tên công ty: {company_name}")

        document = ' '.join(data_cty['text'].astype(str))

        # Negative words
        negative_count, negative_word_list = find_words(document, negative_words)
        if negative_word_list:
            print("\nCác chủ đề tiêu cực:")
            lda_neg = LatentDirichletAllocation(n_components=5, random_state=42)
            # Use a new vectorizer with min_df=1 for small lists
            from sklearn.feature_extraction.text import CountVectorizer
            temp_vectorizer = CountVectorizer(min_df=1)
            X_neg = temp_vectorizer.fit_transform(negative_word_list)
            lda_neg.fit(X_neg)
            feature_names = temp_vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(lda_neg.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-11:-1] if i < len(feature_names)]
                print(f"  Chủ đề #{topic_idx + 1}: {', '.join(top_words)}")
        else:
            print("\nKhông có từ tiêu cực.")

        # Positive words
        positive_count, positive_word_list = find_words(document, positive_words)
        if positive_word_list:
            print("\nCác chủ đề tích cực:")
            lda_pos = LatentDirichletAllocation(n_components=5, random_state=42)
            # Use a new vectorizer with min_df=1 for small lists
            from sklearn.feature_extraction.text import CountVectorizer
            temp_vectorizer = CountVectorizer(min_df=1)
            X_pos = temp_vectorizer.fit_transform(positive_word_list)
            lda_pos.fit(X_pos)
            feature_names = temp_vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(lda_pos.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-11:-1] if i < len(feature_names)]
                print(f"  Chủ đề #{topic_idx + 1}: {', '.join(top_words)}")
        else:
            print("\nKhông có từ tích cực.")

    

    st.subheader("🔍 Sentiment Analysis Result")

    def analyze_company_topics_dashboard(df, company_id=None, n_topics=5):
        st.subheader("📊 Topic Analysis for Company Feedback")

        # valid_ids = sorted(df['id'].unique())
        # st.markdown(f"**Valid Company IDs:** {valid_ids}")

        # if company_id is None:
        #     company_id = st.number_input("Enter a Company ID:", min_value=int(min(valid_ids)), 
        #                                 max_value=int(max(valid_ids)), step=1)

        # if company_id not in valid_ids:
        #     st.error(f"Company ID {company_id} not found.")
            # return

        data_cty = df[df['id'] == company_id].reset_index(drop=True)
        company_name = data_cty['Company Name'].iloc[0]
        st.markdown(f"### 🏢 Company Name: `{company_name}`")

        document = ' '.join(data_cty['text'].astype(str))

        # --- Negative Topics ---
        negative_count, negative_word_list = find_words(document, negative_words)
        st.markdown(f"#### 😡 Negative Topics ({negative_count} mentions)")
        if negative_word_list:
            lda_neg = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            temp_vectorizer = CountVectorizer(min_df=1)
            X_neg = temp_vectorizer.fit_transform([' '.join(negative_word_list)])

            lda_neg.fit(X_neg)
            feature_names = temp_vectorizer.get_feature_names_out()

            for topic_idx, topic in enumerate(lda_neg.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-11:-1] if i < len(feature_names)]
                st.markdown(f"- **Topic #{topic_idx + 1}:** {', '.join(top_words)}")
        else:
            st.info("No negative words found.")

        # --- Positive Topics ---
        positive_count, positive_word_list = find_words(document, positive_words)
        st.markdown(f"#### 😊 Positive Topics ({positive_count} mentions)")
        if positive_word_list:
            lda_pos = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            temp_vectorizer = CountVectorizer(min_df=1)
            X_pos = temp_vectorizer.fit_transform([' '.join(positive_word_list)])
            lda_pos.fit(X_pos)
            feature_names = temp_vectorizer.get_feature_names_out()

            for topic_idx, topic in enumerate(lda_pos.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-11:-1] if i < len(feature_names)]
                st.markdown(f"- **Topic #{topic_idx + 1}:** {', '.join(top_words)}")
        else:
            st.info("No positive words found.")
    # Ensure df is preprocessed and has 'clean' column (done earlier)

    # Select company from dropdown
    company_options = df[['id', 'Company Name']].drop_duplicates().reset_index(drop=True)
    selected_company_name = st.selectbox("🔍 Select a Company", company_options['Company Name'], key='company')

    # Map selected name to id
    selected_company_id = company_options[company_options['Company Name'] == selected_company_name]['id'].values[0]

    # Run topic analysis
    analyze_company_topics_dashboard(df, company_id=selected_company_id)


    # You can add sentiment analysis or charts here in the next steps

    st.subheader("✍️ Test Your Own Feedback")

    user_input = st.text_area("Enter your feedback text here:")

    if user_input:
        lower_text = user_input.lower()

        pos_hits = [word for word in positive_words if word in lower_text]
        neg_hits = [word for word in negative_words if word in lower_text]

        # Classification logic
        if len(pos_hits) > len(neg_hits):
            st.success(f"🟢 Positive feedback detected.")
        elif len(neg_hits) > len(pos_hits):
            st.error(f"🔴 Negative feedback detected.")
        elif len(pos_hits) == len(neg_hits) and len(pos_hits) > 0:
            st.warning(f"🟡 Neutral feedback.")

        else:
            # Fallback to TextBlob
            polarity = TextBlob(user_input).sentiment.polarity
            if polarity > 0:
                st.success("🟢 Positive feedback detected (via TextBlob)")
            elif polarity < 0:
                st.error("🔴 Negative feedback detected (via TextBlob)")
            else:
                st.info("⚪ Neutral feedback detected (via TextBlob)")

    df = df.dropna(subset=["Title", "What I liked", "Suggestions for improvement"])
    df['positive_text'] = df['What I liked']
    df['negative_text'] = df['Suggestions for improvement']
    df['sentiment'] = df['Rating'].apply(lambda x: 'negative' if x <= 2 else 'positive' if x >= 4 else 'neutral')

    st.subheader("Information Clustering about Positive and Negative Feedback")

    def show_cluster_topics_by_sentiment(df, company_id, n_clusters=3):
        st.subheader("📑 Cluster Topics by Sentiment")

    # Filter company data
        company_data = df[df['id'] == company_id].reset_index(drop=True)
    
        if company_data.empty:
            st.warning("⚠️ No feedback found for this company.")
            return

    # Positive data
        pos_data = company_data[company_data['sentiment'] == 'positive']
        st.markdown("### ✅ Positive Feedback Topics")
        if not pos_data.empty:
            try:
                pos_matrix, pos_vectorizer = safe_vectorizer_fit(pos_data['positive_text'].astype(str))
                lda_pos = LatentDirichletAllocation(n_components=n_clusters, random_state=42)
                lda_pos.fit(pos_matrix)
                pos_features = pos_vectorizer.get_feature_names_out()
                for topic_idx, topic in enumerate(lda_pos.components_):
                    top_words = [pos_features[i] for i in topic.argsort()[:-11:-1]]
                    st.markdown(f"- **Topic {topic_idx+1}:** {', '.join(top_words)}")
            except ValueError as e:
                st.warning(f"No positive feedback for this company.")
        else:
            st.info("No positive feedback for this company.")

        # Negative data
        neg_data = company_data[company_data['sentiment'] == 'negative']
        st.markdown("### ❌ Negative Feedback Topics")
        if not neg_data.empty:
            try:
                neg_matrix, neg_vectorizer = safe_vectorizer_fit(neg_data['negative_text'].astype(str))
                lda_neg = LatentDirichletAllocation(n_components=n_clusters, random_state=42)
                lda_neg.fit(neg_matrix)
                neg_features = neg_vectorizer.get_feature_names_out()
                for topic_idx, topic in enumerate(lda_neg.components_):
                    top_words = [neg_features[i] for i in topic.argsort()[:-11:-1]]
                    st.markdown(f"- **Topic {topic_idx+1}:** {', '.join(top_words)}")
            except ValueError as e:
                st.warning(f"No negative feedback for this company.")
        else:
            st.info("No negative feedback for this company.")

    # Run topic analysis
    analyze_company_topics_dashboard(df, company_id=selected_company_id)
# Run cluster analysis by sentiment
    show_cluster_topics_by_sentiment(df, selected_company_id)

        

# ----------------- Project Process -----------------
elif option == "Project Process":
    st.title("📈 Project Process Overview")
    st.write("""
    1. **Data Collection**: Gather feedback data from ITViec platform.
    """)

    @st.cache_data
    def load_data():
        return pd.read_excel('Reviews.xlsx')

    df = load_data()

    if st.checkbox("Show Raw Feedback Data"):
        st.write(df.head(10))

    st.write("""
    2. **Data Preprocessing**: Clean and prepare the data for analysis.
    """)
    st.write("""
    3. **Sentiment Analysis**: Use TextBlob to analyze sentiment of feedback.
    """)
    st.write("""
    4. **Visualization**: Create visualizations to summarize findings.
    """)
    st.write("""
    5. **Deployment**: Deploy the application for user interaction.
    """)
