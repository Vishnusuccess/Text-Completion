import sqlite3
import pandas as pd
import re

# Connect to SQLite database
conn = sqlite3.connect('reddit.db')
cursor = conn.cursor()

# Query the comments table
cursor.execute("SELECT body FROM comment")  
rows = cursor.fetchall()

# Convert to DataFrame
df = pd.DataFrame(rows, columns=["comment"])

# Function to clean text
def clean_text(text):
    text = re.sub(r"[\n●]", " ", text) 
    text = re.sub(r"http\S+", "", text) 
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", text)  
    text = re.sub(r"\s+", " ", text).strip() 
    return text

# Apply cleaning
df["comment"] = df["comment"].apply(clean_text)

# Save cleaned dataset
df.to_csv("cleaned_reddit_comments.csv", index=False)

# Close DB connection
conn.close()


