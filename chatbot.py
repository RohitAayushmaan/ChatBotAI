from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
import json

CB = ChatBot('BerryChatBot')

# Loading data from JSON file
file_path = 'trainData.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Training with tainData.json
trainer = ListTrainer(CB)
for intent in data['intents']:
    patterns = intent['patterns']
    responses = intent['responses']

    for pattern in patterns:
        trainer.train([pattern] + responses)

trainer_corpus = ChatterBotCorpusTrainer(CB)
trainer.train('chatterbot.corpus.english')