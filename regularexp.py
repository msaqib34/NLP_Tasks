import re


# Searching for a pattern in a string
input_string = input("Enter a string: ")
print(re.search("python",input_string,re.IGNORECASE))


# Use of IgnoreCase flag
text = "Cat,cat,CAT,CAt,dog"
print(re.findall("cat", text, re.IGNORECASE))

# Split on The Word "and"
for_split = "oneAndTwoAndThreeandFour"
print(re.split("and", for_split, flags=re.IGNORECASE))


# Split Paragraph into Sentences
paragraph = """This is a sample paragraph. 
It contains multiple sentences. 
Let's see how it works!"""
 
print(re.findall(r'\b\w+\b', paragraph,re.MULTILINE))
# # sentences = re.split(r'[.!?]+', paragraph) 
# # sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
# # print(sentences)
# print(paragraph[0:4])


# Replaceing Words in a String
greetings = "Hello, how are you? Hello again!"
matches = re.sub(r"Hello","Hi", greetings)
print(matches)

# qu = "quantitative and qualitative research"
# print(re.findall("quantitative", qu))

# Extracting words from a String

txt= "Code 87686 of Lahore city  42"
remove_number= re.sub(r"\d+", "", txt)
remove_space= re.sub(r"\s+", " ", remove_number)
print(remove_space)

# Extracting Numbers from a String and Extracting Two Digit Numbers

# print(re.findall(r"[a-z,A-Z]+", txt))
print(re.sub(r"[^\d]", " ",txt))
two_num=re.findall(r"\b\d{2}\b", txt )
print(int(two_num[0]))



