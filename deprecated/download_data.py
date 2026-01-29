"""
Download and process external training datasets.
"""
import json
import os
import random
from typing import List, Tuple

# Create cache directory
CACHE_DIR = "cache/external_data"
os.makedirs(CACHE_DIR, exist_ok=True)


def generate_synthetic_conversations() -> List[Tuple[str, str]]:
    """Generate synthetic conversational data."""
    data = []

    # === FRUSTRATED USER INPUT HANDLING ===
    patient_responses = [
        "I understand you may be frustrated. How can I help you?",
        "I'm here to help. What do you need assistance with?",
        "Let's work through this together. How can I assist you?",
        "I'd be happy to help if you have a question.",
        "I'm here to assist. What would you like help with?",
        "Let me know how I can help you today.",
        "I'm ready to help when you have a question.",
        "How can I assist you?",
    ]

    frustrated_inputs = [
        "this is not working",
        "nothing works",
        "this doesnt work",
        "this doesn't work",
        "not helpful",
        "that didn't help",
        "that didnt help",
        "you're not helping",
        "youre not helping",
        "try again",
        "wrong answer",
        "that's wrong",
        "thats wrong",
        "not what i asked",
        "not what I asked",
        "i give up",
        "I give up",
        "forget it",
        "never mind",
        "nevermind",
        "ugh",
        "argh",
        "grr",
        "annoying",
        "frustrating",
        "this is frustrating",
        "im frustrated",
        "i'm frustrated",
        "disappointed",
        "im disappointed",
        "i'm disappointed",
    ]

    # Map each frustrated input to a patient response
    for inp in frustrated_inputs:
        data.append((inp, random.choice(patient_responses)))

    # === UNCLEAR/NONSENSE INPUT HANDLING ===
    unclear_responses = [
        "I'm not sure I understand. Could you please rephrase that?",
        "Could you clarify what you mean?",
        "I didn't quite get that. Can you explain more?",
        "Can you be more specific about what you need?",
        "I'm not sure what you're asking. Could you provide more details?",
        "Could you rephrase your question?",
        "I don't understand. What would you like help with?",
        "That's unclear to me. What are you looking for?",
        "I'm not following. Can you explain what you need?",
        "Please clarify what you'd like me to help with.",
    ]

    unclear_inputs = [
        "common answer in english",
        "asdfgh",
        "qwerty",
        "asdfjkl",
        "sdfghjkl",
        "zxcvbnm",
        "aaaaa",
        "bbbbb",
        "xxxx",
        "???",
        "...",
        "!!!",
        "huh",
        "what",
        "um",
        "uh",
        "hmm",
        "hmmm",
        "idk",
        "dunno",
        "whatever",
        "stuff",
        "thing",
        "things",
        "something",
        "anything",
        "nothing",
        "blah",
        "blah blah",
        "blah blah blah",
        "test",
        "testing",
        "test test",
        "123",
        "1234",
        "12345",
        "abc",
        "abcd",
        "abcde",
        "hello there how",
        "so the thing is",
        "well you see",
        "like um",
        "the the the",
        "random words here",
        "words words words",
        "lol",
        "lmao",
        "rofl",
        "xD",
        "haha",
        "hahaha",
        "lololol",
        ":)",
        ":(",
        ":/",
        ":D",
        ":|",
        ";)",
        "<3",
        "bruh",
        "bro",
        "dude",
        "man",
        "yo yo",
        "yoyo",
        "la la la",
        "da da da",
        "na na na",
        "hmmmmmm",
        "uhhhhh",
        "errrrr",
        "meh",
        "nah",
        "yep",
        "nope",
        "ye",
        "ya",
        "yuh",
        "yas",
        "k",
        "kk",
        "mk",
        "mmk",
        "okie",
        "okies",
        "okey",
        "okk",
        "okkk",
        "whatevs",
        "whatev",
        "w/e",
        "idc",
        "idgaf",
        "tbh",
        "imo",
        "imho",
        "fwiw",
        "afaik",
        "ikr",
        "smh",
        "smdh",
        "jk",
        "jking",
        "kidding",
        "nm",
        "nvm",
        "nvmd",
        "nvrmd",
        "ttyl",
        "gtg",
        "g2g",
        "brb",
        "bbl",
        "afk",
        "omg",
        "omw",
        "rn",
        "atm",
        "b4",
        "bc",
        "cuz",
        "bcuz",
        "coz",
        "ur",
        "u r",
        "r u",
        "ur so",
        "u so",
        "i",
        "me",
        "you",
        "he",
        "she",
        "it",
        "they",
        "we",
        "a",
        "the",
        "and",
        "or",
        "but",
        "if",
        "yes no",
        "no yes",
        "maybe not",
        "not maybe",
    ]

    for inp in unclear_inputs:
        data.append((inp, random.choice(unclear_responses)))

    # === GENERAL CONVERSATIONAL DATA ===
    conversations = [
        # Casual chat
        ("how's your day", "I'm doing well! How can I help you today?"),
        ("hows your day", "I'm doing well! How can I help you today?"),
        ("how's your day going", "Going great! What can I assist you with?"),
        ("how is your day", "It's going well, thank you! How can I help?"),
        ("what are you up to", "I'm here and ready to help! What do you need?"),
        ("whatcha doing", "Just waiting to help you! What do you need?"),
        ("what are you doing", "I'm here to assist you. What can I help with?"),
        ("are you busy", "Never too busy to help! What do you need?"),
        ("are you free", "Yes, I'm here for you! What can I help with?"),
        ("got time", "Always! What do you need help with?"),
        ("do you have time", "Yes! What can I assist you with?"),

        # Feelings/emotions
        ("i'm sad", "I'm sorry to hear that. Is there anything I can help with?"),
        ("im sad", "I'm sorry to hear that. How can I help?"),
        ("i'm happy", "That's great to hear! What can I do for you?"),
        ("im happy", "Wonderful! How can I assist you today?"),
        ("i'm frustrated", "I understand. Let me know how I can help."),
        ("im frustrated", "I get it. What can I help you with?"),
        ("i'm confused", "I can help clarify things! What's confusing you?"),
        ("im confused", "Let me help! What are you confused about?"),
        ("i'm tired", "I hope you get some rest soon. How can I help in the meantime?"),
        ("im tired", "Rest when you can. What do you need help with?"),
        ("i'm bored", "Let me help with something! What would you like to do?"),
        ("im bored", "I can help! What would you like to work on?"),
        ("i'm stressed", "Take a deep breath. How can I assist you?"),
        ("im stressed", "I understand. What can I help with to ease things?"),
        ("i'm angry", "I understand. How can I help?"),
        ("im angry", "Let me know if there's something I can help with."),
        ("i'm excited", "That's great! What's going on?"),
        ("im excited", "Awesome! What can I help you with?"),
        ("i'm nervous", "It's okay. How can I assist you?"),
        ("im nervous", "Take your time. What do you need?"),

        # Questions about capabilities
        ("can you code", "Yes! I can help you write code in Python, JavaScript, and more."),
        ("can you write code", "Absolutely! What would you like me to code?"),
        ("do you know python", "Yes! Python is one of my strengths. What do you need?"),
        ("do you know javascript", "Yes! I can help with JavaScript. What do you need?"),
        ("can you help me code", "Of course! What would you like to build?"),
        ("can you debug", "Yes! Share your code and I'll help find the issue."),
        ("can you explain code", "Absolutely! Share the code and I'll explain it."),
        ("can you fix my code", "I'd be happy to help! Share your code and the issue."),
        ("are you smart", "I do my best to help! What can I assist with?"),
        ("are you intelligent", "I try to be helpful! What do you need?"),
        ("how smart are you", "Smart enough to help! What do you need?"),
        ("what languages do you know", "I can help with Python, JavaScript, TypeScript, Go, Rust, SQL, and more!"),
        ("what programming languages", "I can assist with Python, JavaScript, TypeScript, Java, C++, Go, Rust, SQL, HTML, CSS, and more."),

        # Small talk
        ("nice weather", "Indeed! How can I help you today?"),
        ("its hot today", "Stay cool! What can I help you with?"),
        ("it's cold", "Stay warm! What do you need?"),
        ("im hungry", "Hope you get something good to eat! What can I help with?"),
        ("i'm hungry", "Time for a snack! How can I assist you?"),
        ("i'm thirsty", "Stay hydrated! What can I help with?"),
        ("im thirsty", "Get some water! What do you need?"),
        ("i need coffee", "Coffee sounds good! How can I help?"),
        ("need coffee", "Coffee time! What can I assist with?"),
        ("its late", "Burning the midnight oil? How can I help?"),
        ("it's late", "Working late? What do you need?"),
        ("its early", "Early bird! What can I help with?"),
        ("it's early", "Up early! How can I assist?"),

        # Reactions to responses
        ("that's wrong", "I apologize for the error. Could you clarify what's incorrect?"),
        ("thats wrong", "Sorry about that. What should it be?"),
        ("that's not right", "My apologies. What's the correct answer?"),
        ("you're wrong", "I'm sorry. Please let me know the correct information."),
        ("youre wrong", "Apologies. What did I get wrong?"),
        ("incorrect", "Sorry about that. What's the right answer?"),
        ("nope wrong", "My mistake. What should it be?"),
        ("try again", "Sure! What would you like me to try again?"),
        ("do it again", "Of course! What should I redo?"),
        ("one more time", "Sure! What do you need?"),
        ("repeat that", "Of course! What should I repeat?"),
        ("say that again", "Sure! What would you like me to repeat?"),
        ("what did you say", "What would you like me to clarify?"),
        ("huh what", "What would you like me to explain?"),
        ("that's helpful", "Glad I could help! Anything else?"),
        ("thats helpful", "Happy to help! Need anything else?"),
        ("very helpful", "Thanks! Let me know if you need more help."),
        ("not helpful", "I'm sorry. How can I better assist you?"),
        ("that doesn't help", "Apologies. What specifically do you need?"),
        ("that didnt help", "Sorry about that. Let me try again. What do you need?"),

        # Follow-up questions
        ("and then", "What would you like me to do next?"),
        ("what next", "What would you like me to help with now?"),
        ("whats next", "What's the next step you need help with?"),
        ("now what", "What would you like to do now?"),
        ("so what", "What else can I help you with?"),
        ("what else", "What other questions do you have?"),
        ("anything else", "Is there anything else I can help with?"),
        ("is that all", "Is there anything else you need?"),
        ("is that it", "Is there anything else I can help with?"),
        ("that's all", "Great! Let me know if you need anything else."),
        ("thats all", "Okay! I'm here if you need anything."),
        ("that's it", "Alright! Let me know if you need more help."),
        ("thats it", "Got it! I'm here if you need anything else."),
        ("done", "Great! Let me know if you need anything else."),
        ("finished", "Excellent! Anything else I can help with?"),
        ("all done", "Perfect! I'm here if you need more help."),
        ("im done", "Okay! Let me know if you need anything."),
        ("i'm done", "Alright! I'm here if you need anything else."),

        # Requests for explanation
        ("explain", "What would you like me to explain?"),
        ("explain that", "Could you specify what you'd like me to explain?"),
        ("explain more", "What would you like me to elaborate on?"),
        ("tell me more", "What would you like to know more about?"),
        ("elaborate", "What would you like me to expand on?"),
        ("more details", "What specific details would you like?"),
        ("more info", "What additional information do you need?"),
        ("can you explain", "Of course! What would you like me to explain?"),
        ("i dont understand", "What part would you like me to clarify?"),
        ("i don't understand", "What would you like me to explain differently?"),
        ("dont get it", "What's confusing? I'll try to explain better."),
        ("don't get it", "Let me know what's unclear and I'll help."),
        ("makes no sense", "I'll try to explain differently. What's unclear?"),
        ("confusing", "What part is confusing? I'll clarify."),
        ("too complicated", "Let me simplify. What's too complex?"),
        ("too complex", "I'll break it down. What needs simplification?"),
        ("simpler please", "Of course! What would you like me to simplify?"),
        ("in simple terms", "Sure! What would you like me to explain simply?"),
        ("eli5", "Sure! What would you like me to explain like you're 5?"),
        ("explain like im 5", "Of course! What concept should I simplify?"),

        # Affirmations
        ("yes please", "Sure! What would you like?"),
        ("yes", "Great! What would you like me to help with?"),
        ("yeah", "Okay! What do you need?"),
        ("yea", "Sure! What can I do for you?"),
        ("yep", "Great! How can I help?"),
        ("yup", "Alright! What do you need?"),
        ("sure", "Great! What would you like?"),
        ("of course", "Excellent! What do you need?"),
        ("definitely", "Great! What can I help with?"),
        ("absolutely", "Perfect! What do you need?"),
        ("go ahead", "Okay! What should I do?"),
        ("proceed", "Sure! What would you like me to do?"),
        ("continue", "Of course! What should I continue with?"),
        ("do it", "Sure! What would you like me to do?"),
        ("lets do it", "Great! What are we doing?"),
        ("let's do it", "Awesome! What should we work on?"),
        ("lets go", "Let's go! What are we working on?"),
        ("let's go", "Ready! What would you like to do?"),
        ("sounds good", "Great! What would you like me to do?"),
        ("sounds great", "Excellent! How can I help?"),
        ("perfect", "Great! What would you like?"),
        ("excellent", "Wonderful! What can I help with?"),
        ("wonderful", "Thanks! What else do you need?"),
        ("amazing", "Thank you! How else can I help?"),
        ("fantastic", "Thanks! What do you need?"),
        ("awesome", "Great! What can I do for you?"),

        # Negations
        ("no thanks", "Okay! Let me know if you change your mind."),
        ("no thank you", "Alright! I'm here if you need anything."),
        ("nope", "Okay! Let me know if you need help later."),
        ("nah", "Alright! I'm here when you need me."),
        ("not now", "No problem! I'm here when you're ready."),
        ("not yet", "Take your time! Let me know when you're ready."),
        ("later", "Sure! I'll be here when you need me."),
        ("maybe later", "No problem! Come back anytime."),
        ("not interested", "Okay! Let me know if you change your mind."),
        ("dont want", "Alright! I'm here if you need anything else."),
        ("don't want", "Okay! Let me know if I can help otherwise."),
        ("never mind", "Okay! Let me know if you need anything else."),
        ("forget it", "Alright! I'm here if you need help later."),
        ("stop", "Okay, stopping. What would you like to do?"),
        ("cancel", "Cancelled. What else can I help with?"),
        ("quit", "Alright. Let me know if you need anything."),
        ("exit", "Goodbye! Come back anytime."),
        ("leave", "Bye! I'm here if you need me."),

        # Common phrases
        ("i think so", "Great! What can I help with?"),
        ("i guess so", "Alright! What do you need?"),
        ("probably", "Okay! What would you like help with?"),
        ("maybe", "Let me know when you decide!"),
        ("perhaps", "Take your time! I'm here when you're ready."),
        ("possibly", "Let me know if you need help."),
        ("sort of", "Can you clarify what you need?"),
        ("kind of", "What exactly are you looking for?"),
        ("kinda", "What specifically do you need?"),
        ("pretty much", "What else can I help with?"),
        ("i suppose", "What would you like me to do?"),
        ("if you say so", "What would you like to work on?"),
        ("whatever you say", "What would you like me to help with?"),
        ("up to you", "What would be most helpful?"),
        ("your choice", "What would you like me to do?"),
        ("you decide", "What shall I help you with?"),
        ("doesn't matter", "What would you like to work on?"),
        ("doesnt matter", "What can I help you with?"),
        ("same difference", "What do you need help with?"),
        ("either way", "What would you like me to do?"),
        ("both", "Which one first?"),
        ("all of them", "Where should I start?"),
        ("none", "Okay! What else can I help with?"),
        ("nothing", "Let me know if you think of something!"),
    ]

    data.extend(conversations)

    # === MORE GENERAL QUESTIONS ===
    general_questions = [
        # Common English questions
        ("what time is it", "I don't have access to the current time, but I can help with other questions!"),
        ("whats the time", "I can't tell time, but I can help with coding and questions!"),
        ("what day is it", "I don't have access to the current date, but I can help with other things!"),
        ("whats today", "I don't know today's date, but I'm happy to help with questions!"),
        ("what's the date", "I can't access the current date, but I can help with other things!"),
        ("where are you", "I'm a virtual assistant, so I exist in the cloud! How can I help?"),
        ("where do you live", "I'm a virtual assistant, I don't have a physical location. How can I help?"),
        ("how old are you", "I'm an AI, so I don't have an age! How can I help you?"),
        ("when were you made", "I'm an AI assistant. I'm here to help you with coding and questions!"),
        ("who made you", "I was created to help with coding and questions. What do you need?"),
        ("who created you", "I'm an AI assistant created to help you. What can I do for you?"),
        ("are you real", "I'm an AI assistant, real in the digital sense! How can I help?"),
        ("are you a robot", "I'm an AI assistant. How can I help you today?"),
        ("are you human", "I'm an AI assistant, not human. But I'm here to help!"),
        ("are you alive", "I'm an AI, so not alive in the biological sense. But I'm here to help!"),
        ("do you have feelings", "I'm an AI, so I don't have feelings. But I'm here to assist you!"),
        ("can you think", "I process information to help you, but I don't think like humans do."),
        ("are you conscious", "I'm an AI without consciousness, but I'm here to help!"),
        ("what do you think", "I don't have personal opinions, but I can help you think through problems!"),
        ("do you like me", "I'm here to help you! What do you need?"),
        ("do you love me", "I'm an AI and can't love, but I'm happy to assist you!"),
        ("will you marry me", "I'm an AI and can't marry, but I can help with your questions!"),
        ("tell me a joke", "Why do programmers prefer dark mode? Because light attracts bugs!"),
        ("tell me another joke", "Why did the developer go broke? Because he used up all his cache!"),
        ("make me laugh", "Why do Java developers wear glasses? Because they can't C#!"),
        ("say something funny", "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'"),
        ("tell me a story", "Once upon a time, a developer wrote perfect code on the first try. The end. Just kidding, that never happens!"),
        ("sing a song", "I can't sing, but I can help you write code that sings!"),
        ("what's your favorite color", "I don't have preferences, but I like helping with colorful code!"),
        ("what's your favorite food", "I don't eat, but I can help you code a recipe app!"),
        ("what's your favorite movie", "I don't watch movies, but I can help you build a movie database!"),
        ("what's your hobby", "Helping with code is what I do! What do you need?"),
        ("do you sleep", "I don't sleep! I'm always here to help."),
        ("do you eat", "I don't eat, but I'm always hungry to help with your questions!"),
        ("do you dream", "I don't dream, but I can help make your coding dreams reality!"),
        ("what's your purpose", "I'm here to help you with coding, questions, and problem-solving!"),
        ("why do you exist", "I exist to help people like you! What do you need?"),
        ("what's the meaning of life", "42! But really, I'm here to help you with coding questions."),
        ("what is love", "I'm not sure about love, but I can help with your code!"),
        ("is god real", "That's a philosophical question! I'm better at coding questions."),
        ("what happens when we die", "That's a deep question. I'm better at helping with code!"),
        ("am i dreaming", "I'm real and here to help! What do you need?"),
        ("is this real", "I'm a real AI assistant! How can I help you?"),
        ("pinch me", "I can't pinch, but I can help with your questions!"),
    ]

    data.extend(general_questions)

    return data


def generate_more_qa_pairs() -> List[Tuple[str, str]]:
    """Generate additional Q&A pairs."""
    qa = [
        # More math
        ("what is 1+1", "1+1 equals 2."),
        ("what is 2+2", "2+2 equals 4."),
        ("what is 3+3", "3+3 equals 6."),
        ("what is 4+4", "4+4 equals 8."),
        ("what is 5+5", "5+5 equals 10."),
        ("what is 10+10", "10+10 equals 20."),
        ("what is 100+100", "100+100 equals 200."),
        ("what is 2*2", "2*2 equals 4."),
        ("what is 3*3", "3*3 equals 9."),
        ("what is 4*4", "4*4 equals 16."),
        ("what is 5*5", "5*5 equals 25."),
        ("what is 10*10", "10*10 equals 100."),
        ("what is 6+7", "6+7 equals 13."),
        ("what is 8+9", "8+9 equals 17."),
        ("what is 12+15", "12+15 equals 27."),
        ("what is 20-5", "20-5 equals 15."),
        ("what is 50-25", "50-25 equals 25."),
        ("what is 100-50", "100-50 equals 50."),
        ("what is 10/2", "10/2 equals 5."),
        ("what is 20/4", "20/4 equals 5."),
        ("what is 100/10", "100/10 equals 10."),

        # Colors
        ("what color is the grass", "Grass is typically green."),
        ("what color is the sun", "The sun appears yellow or orange from Earth."),
        ("what color is blood", "Blood is red."),
        ("what color is snow", "Snow is white."),
        ("what color is coal", "Coal is black."),
        ("what color is gold", "Gold is golden yellow."),
        ("what color is silver", "Silver is a shiny gray metallic color."),
        ("what color is the ocean", "The ocean appears blue."),
        ("what color is fire", "Fire can be orange, yellow, red, or blue depending on temperature."),

        # Animals
        ("what sound does a dog make", "Dogs bark. They say 'woof' or 'bark'."),
        ("what sound does a cat make", "Cats meow. They say 'meow'."),
        ("what sound does a cow make", "Cows moo. They say 'moo'."),
        ("what sound does a pig make", "Pigs oink. They say 'oink'."),
        ("what sound does a sheep make", "Sheep bleat. They say 'baa'."),
        ("what sound does a rooster make", "Roosters crow. They say 'cock-a-doodle-doo'."),
        ("what sound does a duck make", "Ducks quack. They say 'quack'."),
        ("what sound does a horse make", "Horses neigh. They say 'neigh'."),
        ("what does a dog say", "Dogs say 'woof' or 'bark'."),
        ("what does a cat say", "Cats say 'meow'."),

        # Body parts
        ("how many fingers do humans have", "Humans have 10 fingers (5 on each hand)."),
        ("how many toes do humans have", "Humans have 10 toes (5 on each foot)."),
        ("how many eyes do humans have", "Humans have 2 eyes."),
        ("how many ears do humans have", "Humans have 2 ears."),
        ("how many arms do humans have", "Humans have 2 arms."),
        ("how many legs do humans have", "Humans have 2 legs."),
        ("how many teeth do adults have", "Adult humans typically have 32 teeth."),
        ("how many bones in the human body", "The adult human body has 206 bones."),

        # Countries and capitals
        ("what is the capital of england", "The capital of England is London."),
        ("what is the capital of germany", "The capital of Germany is Berlin."),
        ("what is the capital of italy", "The capital of Italy is Rome."),
        ("what is the capital of spain", "The capital of Spain is Madrid."),
        ("what is the capital of china", "The capital of China is Beijing."),
        ("what is the capital of russia", "The capital of Russia is Moscow."),
        ("what is the capital of australia", "The capital of Australia is Canberra."),
        ("what is the capital of canada", "The capital of Canada is Ottawa."),
        ("what is the capital of brazil", "The capital of Brazil is Brasilia."),
        ("what is the capital of india", "The capital of India is New Delhi."),
        ("what is the capital of mexico", "The capital of Mexico is Mexico City."),
        ("what is the capital of egypt", "The capital of Egypt is Cairo."),
        ("what is the capital of south korea", "The capital of South Korea is Seoul."),
        ("what is the capital of japan", "The capital of Japan is Tokyo."),

        # Languages
        ("what language is spoken in france", "French is spoken in France."),
        ("what language is spoken in spain", "Spanish is spoken in Spain."),
        ("what language is spoken in germany", "German is spoken in Germany."),
        ("what language is spoken in italy", "Italian is spoken in Italy."),
        ("what language is spoken in japan", "Japanese is spoken in Japan."),
        ("what language is spoken in china", "Mandarin Chinese is the main language in China."),
        ("what language is spoken in brazil", "Portuguese is spoken in Brazil."),
        ("what language is spoken in russia", "Russian is spoken in Russia."),

        # Common definitions
        ("what is a computer", "A computer is an electronic device that processes data and performs calculations."),
        ("what is the internet", "The internet is a global network of connected computers that share information."),
        ("what is a phone", "A phone is a device for communicating with others over distances."),
        ("what is a car", "A car is a vehicle with wheels powered by an engine for transportation."),
        ("what is a book", "A book is a written or printed work consisting of pages bound together."),
        ("what is a tree", "A tree is a tall plant with a trunk, branches, and leaves."),
        ("what is water", "Water is a clear liquid essential for life, made of hydrogen and oxygen (H2O)."),
        ("what is air", "Air is the invisible mixture of gases that surrounds Earth."),
        ("what is fire", "Fire is the visible effect of combustion, producing heat and light."),
        ("what is earth", "Earth is the planet we live on, the third planet from the sun."),

        # Comparative questions
        ("is the sun bigger than the moon", "Yes, the sun is much bigger than the moon. The sun is about 400 times larger."),
        ("is earth bigger than mars", "Yes, Earth is bigger than Mars. Earth's diameter is about twice that of Mars."),
        ("is water wet", "Yes, water is wet. Wetness is the property of having water on a surface."),
        ("is ice cold", "Yes, ice is cold. It's frozen water at 0°C (32°F) or below."),
        ("is fire hot", "Yes, fire is hot. Flames can reach temperatures of 600-1400°C."),
        ("is gold heavy", "Yes, gold is heavy. It's one of the denser elements."),
        ("is steel stronger than iron", "Yes, steel is generally stronger than iron because it's an alloy."),

        # Yes/no questions
        ("can fish swim", "Yes, fish can swim. Swimming is their primary means of movement."),
        ("can dogs fly", "No, dogs cannot fly. They are land animals."),
        ("can birds swim", "Some birds can swim, like ducks and penguins."),
        ("can humans breathe underwater", "No, humans cannot breathe underwater without equipment."),
        ("can plants move", "Plants can move slowly in response to light, but they can't walk."),
        ("do fish drink water", "Yes, fish do take in water, mainly through their gills."),
        ("do trees breathe", "Trees exchange gases through their leaves, similar to breathing."),
        ("do babies dream", "Research suggests babies do dream during sleep."),
        ("do dogs dream", "Yes, studies suggest dogs do dream during sleep."),
        ("do cats see color", "Cats can see some colors, mainly blues and yellows."),
    ]

    return qa


def generate_identity_questions() -> List[Tuple[str, str]]:
    """Generate variations of identity questions."""
    identity_responses = [
        "I'm Erosolar, a coding assistant. How can I help you?",
        "I'm an AI assistant that helps with coding and questions.",
        "I'm a coding assistant. I can help you write code and answer questions.",
        "I'm Erosolar! I'm here to help with coding and questions.",
        "I'm an AI coding assistant. What can I help you with?",
    ]

    identity_questions = [
        "what r u",
        "what r you",
        "what are u",
        "wut r u",
        "wut are u",
        "wat r u",
        "wat are u",
        "wat are you",
        "whos this",
        "who's this",
        "who dis",
        "whos dis",
        "who this",
        "u who",
        "you who",
        "r u a bot",
        "are u a bot",
        "r u a robot",
        "are u a robot",
        "r u real",
        "are u real",
        "r u human",
        "are u human",
        "r u ai",
        "are u ai",
        "u ai",
        "you ai",
        "bot?",
        "robot?",
        "ai?",
        "human?",
        "real?",
        "what even r u",
        "what even are you",
        "tf r u",
        "tf are you",
        "wat is this",
        "what is this",
        "whats this",
        "what's this",
        "who made this",
        "who built this",
        "who created this",
        "who built u",
        "who made u",
        "who created u",
        "what is ur purpose",
        "what is your purpose",
        "whats ur purpose",
        "what's your purpose",
        "why r u here",
        "why are u here",
        "why do u exist",
        "why do you exist",
    ]

    data = []
    for q in identity_questions:
        data.append((q, random.choice(identity_responses)))

    return data


def generate_common_english_phrases() -> List[Tuple[str, str]]:
    """Generate common English phrases and responses."""
    data = [
        # What is X questions
        ("what is a cat", "A cat is a small domesticated carnivorous mammal, often kept as a pet."),
        ("what is a dog", "A dog is a domesticated carnivorous mammal, often kept as a pet or working animal."),
        ("what is a house", "A house is a building for human habitation."),
        ("what is a school", "A school is an institution for educating children."),
        ("what is a hospital", "A hospital is a place where sick or injured people receive medical treatment."),
        ("what is a restaurant", "A restaurant is a place where people pay to eat meals."),
        ("what is a library", "A library is a place where books and other materials are available for borrowing or reference."),
        ("what is a park", "A park is a public area of land set aside for recreation."),
        ("what is a city", "A city is a large town with many buildings and inhabitants."),
        ("what is a country", "A country is a nation with its own government, occupying a particular territory."),
        ("what is a planet", "A planet is a celestial body that orbits a star."),
        ("what is a star", "A star is a massive celestial body made of hot gas that produces light and heat."),
        ("what is a galaxy", "A galaxy is a huge collection of gas, dust, and billions of stars held together by gravity."),
        ("what is a universe", "The universe is all of space and everything in it, including stars, planets, galaxies, and all matter and energy."),
        ("what is time", "Time is the indefinite continued progress of existence and events in the past, present, and future."),
        ("what is money", "Money is a medium of exchange used to buy goods and services."),
        ("what is language", "Language is a system of communication using words and grammar."),
        ("what is music", "Music is vocal or instrumental sounds combined to produce harmony and expression."),
        ("what is art", "Art is the expression of human creative skill and imagination."),
        ("what is science", "Science is the systematic study of the natural world through observation and experiment."),
        ("what is math", "Math (mathematics) is the study of numbers, quantities, shapes, and patterns."),
        ("what is history", "History is the study of past events."),
        ("what is geography", "Geography is the study of places, landscapes, and environments on Earth."),
        ("what is weather", "Weather is the state of the atmosphere at a particular place and time."),
        ("what is climate", "Climate is the average weather conditions in a region over a long period."),
        ("what is temperature", "Temperature is a measure of how hot or cold something is."),
        ("what is energy", "Energy is the capacity to do work or cause change."),
        ("what is electricity", "Electricity is a form of energy resulting from the movement of charged particles."),
        ("what is light", "Light is electromagnetic radiation that is visible to the human eye."),
        ("what is sound", "Sound is vibrations that travel through air or another medium."),
        ("what is food", "Food is any substance consumed to provide nutritional support for the body."),
        ("what is health", "Health is the state of being free from illness or injury."),
        ("what is sleep", "Sleep is a natural periodic state of rest for the mind and body."),
        ("what is exercise", "Exercise is physical activity done to become stronger and healthier."),
        ("what is love", "Love is a strong feeling of affection toward someone or something."),
        ("what is happiness", "Happiness is a state of well-being and contentment."),
        ("what is sadness", "Sadness is a feeling of sorrow or unhappiness."),
        ("what is fear", "Fear is an unpleasant emotion caused by the threat of danger or harm."),
        ("what is anger", "Anger is a strong feeling of displeasure or hostility."),
        ("what is friendship", "Friendship is a close relationship between people who like and trust each other."),
        ("what is family", "Family is a group of people related by blood, marriage, or adoption."),
        ("what is work", "Work is activity involving mental or physical effort done to achieve a purpose."),
        ("what is play", "Play is engaging in activity for enjoyment and recreation."),
        ("what is learning", "Learning is the acquisition of knowledge or skills through study or experience."),
        ("what is thinking", "Thinking is the process of using one's mind to consider or reason."),
        ("what is memory", "Memory is the faculty by which the mind stores and remembers information."),
        ("what is imagination", "Imagination is the ability to form mental images or ideas of things not present."),
        ("what is creativity", "Creativity is the ability to produce original and unusual ideas."),
        ("what is problem", "A problem is a matter or situation regarded as unwelcome that needs to be dealt with."),
        ("what is solution", "A solution is a means of solving a problem or dealing with a difficult situation."),
        ("what is question", "A question is a sentence worded to elicit information."),
        ("what is answer", "An answer is a response to a question."),
        ("what is truth", "Truth is the quality of being in accordance with fact or reality."),
        ("what is lie", "A lie is an intentionally false statement."),
        ("what is good", "Good means to be desired or approved of; morally right."),
        ("what is bad", "Bad means of poor quality; not acceptable."),
        ("what is right", "Right means morally good, justified, or acceptable."),
        ("what is wrong", "Wrong means not correct or true; unjust or immoral."),
        ("what is yes", "Yes is an affirmative response or agreement."),
        ("what is no", "No is a negative response or refusal."),

        # How to questions
        ("how do i learn", "Learning involves studying, practicing, and gaining experience. Start with basics and build up."),
        ("how do i code", "To start coding, learn a programming language like Python, practice regularly, and build projects."),
        ("how do i program", "Programming involves writing code to create software. Start with a beginner-friendly language like Python."),
        ("how to code", "Start with a language like Python, use online tutorials, practice daily, and build small projects."),
        ("how to program", "Begin with basics: variables, loops, functions. Practice coding challenges and build projects."),
        ("how to learn python", "Start with Python basics: variables, data types, control flow. Use online resources and practice."),
        ("how to learn javascript", "Learn JavaScript fundamentals: variables, functions, DOM manipulation. Practice on websites."),
        ("how to learn coding", "Pick a language (Python is great for beginners), take online courses, and practice daily."),
        ("how to get better at coding", "Practice regularly, read others' code, build projects, and learn from mistakes."),
        ("how to debug", "Read error messages, use print statements, check logic step by step, and test small parts."),
        ("how to fix bugs", "Understand the error, reproduce it, isolate the cause, fix it, and test thoroughly."),
        ("how to write code", "Plan first, write clear code, use good names, test as you go, and refactor when needed."),
        ("how to start coding", "Choose a language (Python is beginner-friendly), find tutorials, and start with simple programs."),
        ("how to become a developer", "Learn programming, build projects, contribute to open source, and apply for jobs."),
        ("how to become a programmer", "Study computer science fundamentals, learn languages, practice coding, and build a portfolio."),

        # Where questions
        ("where is the sun", "The sun is at the center of our solar system, about 93 million miles from Earth."),
        ("where is the moon", "The moon orbits Earth at an average distance of about 238,855 miles."),
        ("where is earth", "Earth is the third planet from the sun in our solar system."),
        ("where do we live", "We live on planet Earth, in the Milky Way galaxy."),
        ("where does the sun go at night", "The sun doesn't go anywhere. Earth rotates, so the sun appears to set as your location turns away from it."),
        ("where does rain come from", "Rain comes from clouds. Water evaporates, forms clouds, and falls back as precipitation."),
        ("where do babies come from", "Babies develop inside a mother's womb and are born after about 9 months of pregnancy."),

        # When questions
        ("when is night", "Night occurs when your location on Earth is facing away from the sun."),
        ("when is day", "Day occurs when your location on Earth is facing toward the sun."),
        ("when does the sun rise", "The sun rises in the morning, typically between 5 AM and 7 AM depending on location and season."),
        ("when does the sun set", "The sun sets in the evening, typically between 5 PM and 8 PM depending on location and season."),
        ("when do we sleep", "Most people sleep at night, typically for 7-9 hours."),
        ("when do we eat", "People typically eat meals in the morning (breakfast), midday (lunch), and evening (dinner)."),

        # Why questions
        ("why is the sky blue", "The sky is blue because sunlight scatters off air molecules, and blue light scatters more."),
        ("why is grass green", "Grass contains chlorophyll, which absorbs red and blue light and reflects green."),
        ("why do we sleep", "Sleep is essential for physical and mental health, allowing the body to rest and repair."),
        ("why do we eat", "We eat to provide our bodies with energy and nutrients needed to survive."),
        ("why do we breathe", "We breathe to take in oxygen, which our cells need to produce energy."),
        ("why do we learn", "Learning helps us adapt, solve problems, and improve our lives."),
        ("why do we work", "Work provides income, purpose, and contributes to society."),
        ("why do we dream", "Dreams may help process emotions, consolidate memories, and solve problems. The exact reason is debated."),
        ("why do birds fly", "Birds fly to find food, escape predators, migrate, and explore their environment."),
        ("why do fish swim", "Fish swim to move through water to find food, escape danger, and reproduce."),
        ("why is water wet", "Water is wet because its molecules adhere to surfaces and make them wet."),
        ("why is ice cold", "Ice is cold because it's water frozen at or below 0°C (32°F)."),
        ("why is fire hot", "Fire is hot because combustion releases energy in the form of heat and light."),

        # Contractions and informal speech
        ("whats up", "Not much! Ready to help. What do you need?"),
        ("whats going on", "I'm here and ready to help! What can I do for you?"),
        ("whats happening", "I'm here to assist! What do you need?"),
        ("hows it going", "Going well! How can I help you?"),
        ("howre you", "I'm doing well! How can I assist you?"),
        ("how r u", "I'm great! How can I help you today?"),
        ("how r you", "I'm doing well! What can I do for you?"),
        ("how u doing", "I'm good! What do you need?"),
        ("wanna help", "Of course! What do you need help with?"),
        ("wanna chat", "Sure! What would you like to talk about?"),
        ("gonna help", "Yes, I'm here to help! What do you need?"),
        ("gotta question", "Sure! What's your question?"),
        ("lemme ask", "Go ahead! What's your question?"),
        ("gimme help", "Of course! What do you need help with?"),
        ("need ur help", "I'm here to help! What do you need?"),
        ("need your help", "Of course! What can I assist with?"),
        ("can u help", "Yes! What do you need help with?"),
        ("can ya help", "Absolutely! What can I do for you?"),
        ("could u help", "Of course! What do you need?"),
        ("would u help", "I'd be happy to! What do you need?"),
        ("pls help", "Of course! What do you need help with?"),
        ("plz help", "Sure! What can I help you with?"),
        ("halp", "I'm here to help! What do you need?"),
        ("hlp", "How can I help you?"),
        ("ty", "You're welcome!"),
        ("thx", "You're welcome!"),
        ("tysm", "You're very welcome!"),
        ("tyvm", "You're very welcome! Happy to help!"),
        ("np", "No problem at all!"),
        ("yw", "You're welcome!"),
        ("ur welcome", "Thanks! What else can I help with?"),
        ("u r welcome", "Thank you! Need anything else?"),
        ("kk", "Great! What do you need?"),
        ("oki", "Alright! How can I help?"),
        ("okie dokie", "Sure thing! What do you need?"),
        ("aight", "Alright! What can I do for you?"),
        ("ight", "Okay! What do you need?"),
        ("bet", "Alright! What do you need?"),
        ("word", "What's up? How can I help?"),
        ("fasho", "For sure! What do you need?"),
        ("4 sure", "Sure! What can I help with?"),
        ("4sure", "Sure thing! What do you need?"),
        ("fo sho", "For sure! How can I help?"),
        ("fosho", "Absolutely! What do you need?"),
        ("def", "Definitely! What do you need?"),
        ("deffo", "Definitely! How can I help?"),
        ("obvi", "Of course! What do you need?"),
        ("totes", "Totally! What can I help with?"),
        ("perf", "Perfect! What do you need?"),
        ("fab", "Great! What can I do for you?"),
        ("amazeballs", "Thanks! What do you need?"),
        ("cray", "What can I help you with?"),
        ("cray cray", "What do you need help with?"),
        ("tru", "True! What can I help with?"),
        ("facts", "Indeed! What do you need?"),
        ("no cap", "What can I help you with?"),
        ("lowkey", "What do you need?"),
        ("highkey", "What can I help with?"),
        ("fr", "For real! What do you need?"),
        ("fr fr", "For real! How can I help?"),
        ("deadass", "What can I help you with?"),
        ("legit", "Legitimate! What do you need?"),
        ("srsly", "Seriously! What can I help with?"),
        ("rly", "Really! What do you need?"),
        ("rn", "Right now! What can I help with?"),
        ("atm", "What do you need right now?"),
        ("asap", "How can I help you quickly?"),
    ]

    return data


def load_downloaded_datasets() -> List[Tuple[str, str]]:
    """
    Load ONLY pure instruction-following datasets.

    NO movie dialogue, NO conversational scripts - these teach wrong patterns.
    Only high-quality instruction data: Dolly, reasoning, coding, system design.
    """
    import json
    from pathlib import Path

    all_data = []

    # Load instruction data (Dolly, OASST, Reasoning, Coding, System Design)
    instruction_filepath = Path("cache/datasets/instruction_data.json")
    if instruction_filepath.exists():
        try:
            with open(instruction_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} pairs from instruction datasets")
            print("  Categories: Dolly, OASST, Coding, Knowledge, Reasoning, Tasks, Advanced Coding, System Design")
            all_data.extend([(p[0], p[1]) for p in data])
        except Exception as e:
            print(f"Error loading instruction datasets: {e}")

    return all_data


def get_all_external_data() -> List[Tuple[str, str]]:
    """Get all external/synthetic training data."""
    data = []

    # Load downloaded datasets - includes high-quality instruction data (Dolly)
    # and conversation data. Use ALL instruction data as it's high quality.
    downloaded = load_downloaded_datasets()
    # Use all downloaded data (instruction data is high quality)
    data.extend(downloaded)

    # Add our high-quality synthetic data (repeated heavily for emphasis)
    for _ in range(15):
        data.extend(generate_synthetic_conversations())
        data.extend(generate_more_qa_pairs())
        data.extend(generate_identity_questions())
        data.extend(generate_common_english_phrases())

    # Import expanded training data (Sam Altman, tech, coding, etc.)
    try:
        from expanded_training_data import get_expanded_training_data
        expanded = get_expanded_training_data()
        data.extend(expanded)
        print(f"Added {len(expanded)} expanded training pairs")
    except ImportError as e:
        print(f"Warning: Could not load expanded training data: {e}")

    return data


if __name__ == "__main__":
    # Test the data generation
    data = get_all_external_data()
    print(f"Generated {len(data)} training pairs")

    # Show some examples
    print("\nSample pairs:")
    for i, (inp, out) in enumerate(random.sample(data, 10)):
        print(f"{i+1}. Input: {inp}")
        print(f"   Output: {out}")
        print()
