# làm sạch, loại bỏ noise trong dữ liệu
def news_cleaning_lower(x):
    # loại bỏ url từ html
    url_pattern = re.compile(r'http\S+')
    x = url_pattern.sub(r'', x)
    # loại bỏ emoji
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002500-\U00002BEF"  # chinese char
                            u"\U00002702-\U000027B0"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\U0001f926-\U0001f937"
                            u"\U00010000-\U0010ffff"
                            u"\u2640-\u2642"
                            u"\u2600-\u2B55"
                            u"\u200d"
                            u"\u23cf"
                            u"\u23e9"
                            u"\u231a"
                            u"\ufe0f"  # dingbats
                            u"\u3030"
                            "]+", flags=re.UNICODE)
    x = emoji_pattern.sub(r'', x)
    x = x.lower()
    return (x)
# Lọc những tin tức trùng dựa trên title -> chưa
# Lọc những tin tức không ý nghĩa
def news_meaningless(df):
    # loại bỏ những tin chứa keywork giải thưởng
    test_list = ['giải thưởng']
    df['check'] = df['title'].apply(lambda x: any(ele in x for ele in test_list))
    df = df[df.check == False]
    df.drop(columns = 'check', inplace = True)
    # loại bỏ những thông tin không có summary
    df = df[(df.summary.notnull())|(df.source=='cafef')]
    return df
# phân loại news theo các phân nhóm
def news_category_title_base(x):
    if x.find('acb') > 0:
        return 'acb'
    else:
        return 'macro'
# sentiment score for news
def get_sentiment_score(x):
    sentence = x  
    input_ids = torch.tensor([tokenizer.encode(sentence)])
    with torch.no_grad():
        out = model(input_ids)
        a = out.logits.softmax(dim=-1).tolist()
    return a[0]
# sentiment_score_groupby_day
def get_sentiment_score_by_day(x):
    avg_score = statistics.mean(x)
    no_of_x = len(x)
    ln_no_of_x = math.log(1 + no_of_x)
    return avg_score/ln_no_of_x