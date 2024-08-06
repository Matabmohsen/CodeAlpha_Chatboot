import pandas as pd
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import random


# Load intents file
with open('Maty_Intents.json') as file:
    intents = json.load(file)

# Prepare data for training
patterns = []
responses = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['tag'])  # Store the tag for each pattern

# Create a DataFrame
df = pd.DataFrame({'User_Input': patterns, 'Tag': responses})

# Tokenization and removing stopwords
stop_words = set(stopwords.words('english'))
df['User_Input_Tokens'] = df['User_Input'].apply(
    lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.isalnum() and word.lower() not in stop_words])
)

# Vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['User_Input_Tokens'])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Tag'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Create a dictionary of responses by tag
responses_dict = {}
for intent in intents['intents']:
    if isinstance(intent['responses'], str):
        responses_dict[intent['tag']] = [intent['responses']]
    else:
        responses_dict[intent['tag']] = intent['responses']

# Function to get chatbot response using the trained model
def get_chatbot_response(user_input):
    input_tokens = ' '.join([word.lower() for word in word_tokenize(user_input) if
                             word.isalnum() and word.lower() not in stopwords.words('english')])
    input_vector = vectorizer.transform([input_tokens])
    prediction = model.predict(input_vector)
    predicted_tag = label_encoder.inverse_transform(prediction)[0]

    # Get responses corresponding to the predicted tag
    response_options = responses_dict.get(predicted_tag, ["Sorry, I don't understand that."])

    # Randomly select one of the responses
    response = random.choice(response_options)

    return response

# Example interaction
print("Chatbot: Hello! How can I help you today?")

while True:
    # Get user input
    user_input = input("You: ")

    # Check for exit conditions
    if user_input.lower() in ['bye', 'goodbye', 'cya', 'see you later']:
        print("Chatbot: Goodbye! Have a great day.")
        break

    # Get the chatbot's response
    bot_response = get_chatbot_response(user_input)

    # Print the chatbot's response
    print(f"Chatbot: {bot_response}")
