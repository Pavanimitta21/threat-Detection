import streamlit as st
import joblib
import re
import nltk
import mysql.connector
import pandas as pd
from urllib.parse import urlparse
from risk_engine import calculate_risk
st.set_page_config(page_title="AI-Based Digital Threat Detection",page_icon="🛡️",layout="wide")
text_model=joblib.load("models/text_model.pkl")
tfidf=joblib.load("models/text_vectorizer.pkl")
url_model=joblib.load("models/url_model.pkl")
nltk.data.path.insert(0,r"C:\Users\NAREN\nltk_data")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words=set(stopwords.words("english"))
db=mysql.connector.connect(host="localhost",user="root",password="sony31",database="digital_threat_db")
cursor=db.cursor()
def save_result(input_type,input_data,prediction,category,risk_score,risk_level,indicators):
    query="""INSERT INTO analysis_history(input_type,input_data,prediction,category,risk_score,risk_level,indicators) VALUES (%s,%s,%s,%s,%s,%s,%s)"""
    values=(input_type,input_data,prediction,category,risk_score,risk_level,indicators)
    cursor.execute(query,values)
    db.commit()
def clean_text(text):
    text=str(text).lower()
    text=re.sub(r"[^a-zA-Z\s]","",text)
    words=word_tokenize(text)
    words=[word for word in words if word not in stop_words]
    return " ".join(words)
def extract_url_features(url):
    if not url.startswith(("http://","https://")):
        url="https://"+url
    parsed=urlparse(url)
    domain=parsed.netloc.split(":")[0]
    length=len(url)
    digits=len(re.findall(r"\d",url))
    special=len(re.findall(r"[^A-Za-z0-9]",url))
    subdomains=max(len(domain.split("."))-2,0)
    is_ip=int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$",domain)))
    return [length,len(domain),subdomains,digits,special,url.count("@"),url.count("?"),url.count("="),int(parsed.scheme=="https"),is_ip]
def find_indicators(text):
    keywords=["urgent","verify","password","otp","bank","account","payment","prize","winner","click","fee","refund","job","selected"]
    text=text.lower()
    found=[]
    for keyword in keywords:
        if keyword in text:
            found.append(keyword)
    return found
def get_category(text):
    text=text.lower()
    if any(word in text for word in ["job","selected","registration fee","vacancy"]):
        return "Fake Job Offer"
    if any(word in text for word in ["bank","otp","payment","refund","upi"]):
        return "Financial Scam"
    if any(word in text for word in ["password","login","verify","click","urgent"]):
        return "Phishing"
    return "Suspicious Message"
st.title("🛡️ AI-Based Digital Threat Detection")
st.write("Detect suspicious messages and URLs using Machine Learning and Risk Scoring.")
st.sidebar.title("Navigation")
page=st.sidebar.radio("Select Page",["Threat Analysis","Dashboard"])
if page=="Threat Analysis":
    st.header("Threat Analysis")
    input_type=st.radio("Select Analysis Type",["Text Analysis","URL Analysis"])
    if input_type=="Text Analysis":
        st.subheader("📩 Analyze Suspicious Message")
        text=st.text_area("Enter the message:",height=150,placeholder="Example: Your bank account has been blocked. Verify now.")
        if st.button("Analyze Text"):
            if text.strip()=="":
                st.warning("Please enter a message.")
            else:
                cleaned=clean_text(text)
                features=tfidf.transform([cleaned])
                prediction=text_model.predict(features)[0]
                probabilities=text_model.predict_proba(features)[0]
                confidence=max(probabilities)*100
                indicators=find_indicators(text)
                score,level=calculate_risk(prediction,confidence,len(indicators))
                if prediction=="threat":
                    category=get_category(text)
                else:
                    category="Safe Message"
                save_result("Text",text,prediction,category,score,level,", ".join(indicators))
                st.subheader("Analysis Result")
                col1,col2,col3=st.columns(3)
                with col1:
                    st.metric("Prediction",prediction.upper())
                with col2:
                    st.metric("Confidence",f"{confidence:.2f}%")
                with col3:
                    st.metric("Risk Score",f"{score}/100")
                st.write("**Risk Level:**",level)
                st.write("**Threat Category:**",category)
                if indicators:
                    st.write("**Threat Indicators:**")
                    st.write(", ".join(indicators))
                else:
                    st.write("**Threat Indicators:** None detected")
                if level=="Critical":
                    st.error("Critical risk detected. Avoid interacting with this message.")
                elif level=="High":
                    st.warning("High risk detected. Be very cautious.")
                elif level=="Medium":
                    st.warning("Medium risk detected. Verify the information before proceeding.")
                else:
                    st.success("Low risk detected.")
                st.success("Analysis saved to MySQL.")
    else:
        st.subheader("🔗 Analyze Suspicious URL")
        url=st.text_input("Enter the URL:",placeholder="Example: https://www.google.com")
        if st.button("Analyze URL"):
            if url.strip()=="":
                st.warning("Please enter a URL.")
            else:
                features=extract_url_features(url)
                prediction=url_model.predict([features])[0]
                probabilities=url_model.predict_proba([features])[0]
                if prediction==1:
                    result="Phishing"
                    confidence=probabilities[1]*100
                    indicators=1
                else:
                    result="Safe"
                    confidence=probabilities[0]*100
                    indicators=0
                score,level=calculate_risk(result,confidence,indicators)
                if result=="Phishing":
                    category="Suspicious URL"
                else:
                    category="Safe URL"
                save_result("URL",url,result,category,score,level,"Suspicious URL" if result=="Phishing" else "")
                st.subheader("Analysis Result")
                col1,col2,col3=st.columns(3)
                with col1:
                    st.metric("Prediction",result)
                with col2:
                    st.metric("Confidence",f"{confidence:.2f}%")
                with col3:
                    st.metric("Risk Score",f"{score}/100")
                st.write("**Risk Level:**",level)
                st.write("**Threat Category:**",category)
                if result=="Phishing":
                    st.error("This URL appears suspicious. Avoid entering personal or financial information.")
                else:
                    st.success("This URL appears safe based on the trained model.")
                st.success("Analysis saved to MySQL.")
else:
    st.header("📊 Threat Detection Dashboard")
    query="""SELECT id,input_type,input_data,prediction,category,risk_score,risk_level,indicators,timestamp FROM analysis_history ORDER BY timestamp DESC"""
    cursor.execute(query)
    rows=cursor.fetchall()
    columns=["id","input_type","input_data","prediction","category","risk_score","risk_level","indicators","timestamp"]
    df=pd.DataFrame(rows,columns=columns)
    if df.empty:
        st.info("No analysis records found. Analyze some text or URLs first.")
    else:
        total=len(df)
        threats=len(df[df["prediction"].isin(["threat","Threat","Phishing"])])
        safe=total-threats
        average_risk=round(df["risk_score"].mean(),2)
        col1,col2,col3,col4=st.columns(4)
        with col1:
            st.metric("Total Analyses",total)
        with col2:
            st.metric("Threats Detected",threats)
        with col3:
            st.metric("Safe Results",safe)
        with col4:
            st.metric("Average Risk",f"{average_risk}/100")
        st.divider()
        st.subheader("Risk Level Summary")
        risk_counts=df["risk_level"].value_counts().reindex(["Low","Medium","High","Critical"],fill_value=0)
        st.bar_chart(risk_counts)
        st.subheader("Threat Categories")
        category_counts=df["category"].value_counts()
        st.bar_chart(category_counts)
        st.subheader("Analysis Type")
        type_counts=df["input_type"].value_counts()
        st.bar_chart(type_counts)
        st.subheader("Analysis History")
        display_df=df.copy()
        display_df.columns=["ID","Input Type","Input","Prediction","Category","Risk Score","Risk Level","Indicators","Timestamp"]
        st.dataframe(display_df,use_container_width=True,hide_index=True)
        if st.button("🔄 Refresh Dashboard"):
            st.rerun()

