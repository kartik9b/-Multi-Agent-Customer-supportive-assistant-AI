#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset

# 1. Best practice: Stream the data so it doesn't take up gigabytes on your disk
print("Connecting to MS MARCO dataset...")
dataset_stream = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)

# 2. Extract a manageable sample for your project development
# Taking the first 1,000 examples is perfect for building your RAG pipeline test cases
sample_data = list(dataset_stream.take(1000))

# 3. Check out the structure of a single query item
print("\nSample Item Data Structure:")
print(f"Query: {sample_data[0]['query']}")
print(f"Passages Found: {len(sample_data[0]['passages']['passage_text'])}")


# In[2]:


pip install chromadb tqdm


# In[4]:


pip install pandas scikit-learn


# In[2]:


pip install streamlit


# In[3]:


import os

print("--- Banking77 Folder Contents ---")
try:
    print(os.listdir("Banking77_Intent_Classification")) # Check if this is the exact folder name
except Exception as e:
    print(e)

print("\n--- Squad_Dataset Folder Contents ---")
try:
    print(os.listdir("Squad_Dataset"))
except Exception as e:
    print(e)


# In[4]:


import os

# Let's list everything in your current directory to see the real folder names
print("Current Directory Contents:")
print(os.listdir("."))


# In[5]:


pip install google-genai


# In[ ]:




