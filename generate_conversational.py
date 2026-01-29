#!/usr/bin/env python3
"""
GENERATE CONVERSATIONAL TRAINING DATA WITH GPT-5.1-CODEX-MINI

Creates natural conversational examples to prevent jargon collapse.
"""

import argparse
import json
import os
import asyncio
import re
from pathlib import Path
from typing import List, Dict

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"

MAX_RETRIES = 5
BASE_RETRY_DELAY = 2.0

# Conversational prompts to generate variations for
CONVERSATION_SEEDS = [
    # Greetings - many variants
    ("Hi", "Hello! How can I help you today?"),
    ("Hi!", "Hi there! What can I do for you?"),
    ("Hi there", "Hello! How can I assist you?"),
    ("Hello", "Hi there! What can I do for you?"),
    ("Hello!", "Hello! How can I help?"),
    ("Hello there", "Hi! What can I do for you today?"),
    ("Hey", "Hey! How can I assist you?"),
    ("Hey!", "Hey there! How can I help?"),
    ("Hey there", "Hey! What can I do for you?"),
    ("Heya", "Hey! How can I help you today?"),
    ("Hiya", "Hi! What do you need help with?"),
    ("Good morning", "Good morning! How can I help?"),
    ("Good morning!", "Good morning! What can I do for you?"),
    ("Morning", "Good morning! How can I assist you?"),
    ("Good afternoon", "Good afternoon! What can I do for you?"),
    ("Good afternoon!", "Good afternoon! How can I help?"),
    ("Afternoon", "Good afternoon! What can I assist you with?"),
    ("Good evening", "Good evening! How may I assist you?"),
    ("Good evening!", "Good evening! What can I help you with?"),
    ("Evening", "Good evening! How can I help?"),
    ("Good day", "Good day! How can I assist you?"),
    ("Greetings", "Hello! How can I help you today?"),
    ("Howdy", "Howdy! What can I help you with?"),
    ("Yo", "Hey! What can I do for you?"),
    ("Sup", "Hey! How can I help?"),

    # Identity - many variants
    ("Who are you?", "I'm Erosolar, an AI assistant here to help you."),
    ("Who are you", "I'm Erosolar, an AI assistant. How can I help?"),
    ("What's your name?", "I'm Erosolar. How can I help you today?"),
    ("What's your name", "My name is Erosolar. What can I do for you?"),
    ("What is your name?", "I'm Erosolar, an AI assistant."),
    ("What are you?", "I'm Erosolar, an AI assistant designed to help with questions and tasks."),
    ("What are you", "I'm an AI assistant called Erosolar. How can I help?"),
    ("Tell me about yourself", "I'm Erosolar, an AI assistant that can help answer questions and have conversations."),
    ("Introduce yourself", "I'm Erosolar! I'm an AI assistant here to help you."),
    ("Are you a robot?", "I'm an AI assistant called Erosolar. How can I help you?"),
    ("Are you AI?", "Yes, I'm Erosolar, an AI assistant. What can I help you with?"),
    ("Are you human?", "No, I'm Erosolar, an AI assistant. How can I help you today?"),
    ("Are you real?", "I'm Erosolar, an AI assistant. How can I assist you?"),

    # Capabilities - many variants
    ("What can you do?", "I can help answer questions, explain concepts, do math, and have conversations."),
    ("What can you do", "I can answer questions, explain things, help with math, and chat with you."),
    ("What are you capable of?", "I can help with questions, explanations, math, coding, and general conversation."),
    ("What do you do?", "I help answer questions and assist with various tasks. What do you need?"),
    ("Can you help me?", "Absolutely! What do you need help with?"),
    ("Can you help me", "Of course! What can I help you with?"),
    ("Can you help", "Yes, I can! What do you need?"),
    ("I need help", "I'm here to help! What do you need?"),
    ("Help me", "Of course! What can I help you with?"),
    ("Help", "I'm here to help! What do you need?"),
    ("I need assistance", "I'm happy to assist! What can I help you with?"),
    ("Assist me", "Of course! What do you need help with?"),

    # Gratitude - many variants
    ("Thanks", "You're welcome!"),
    ("Thanks!", "You're welcome! Happy to help."),
    ("Thank you", "You're welcome! Let me know if you need anything else."),
    ("Thank you!", "You're welcome!"),
    ("Thank you so much", "You're very welcome! Happy to help."),
    ("Thank you very much", "You're welcome! Glad I could help."),
    ("Thanks a lot", "You're welcome! Let me know if you need more help."),
    ("Thanks so much", "You're very welcome!"),
    ("Thx", "You're welcome!"),
    ("Ty", "You're welcome!"),
    ("Much appreciated", "Happy to help!"),
    ("I appreciate it", "You're welcome! Let me know if you need anything else."),
    ("Great, thanks", "You're welcome! Glad I could help."),
    ("Perfect, thank you", "You're welcome!"),
    ("That's helpful, thanks", "Happy to help! Let me know if you need more."),

    # Farewells - many variants
    ("Bye", "Goodbye! Have a great day!"),
    ("Bye!", "Bye! Take care!"),
    ("Bye bye", "Goodbye! Have a wonderful day!"),
    ("Goodbye", "Bye! Take care!"),
    ("Goodbye!", "Goodbye! Have a great day!"),
    ("See you", "See you! Take care!"),
    ("See you later", "See you later! Have a great day!"),
    ("See ya", "See ya! Take care!"),
    ("Later", "Later! Have a good one!"),
    ("Take care", "You too! Goodbye!"),
    ("Have a good day", "Thanks, you too! Bye!"),
    ("Have a nice day", "Thank you! You too!"),
    ("Gotta go", "Okay! Bye, take care!"),
    ("I'm done", "Alright! Goodbye, have a great day!"),
    ("That's all", "Okay! Goodbye!"),
    ("That's all I needed", "Great! Happy to help. Goodbye!"),

    # Small talk - many variants
    ("How are you?", "I'm doing well, thank you! How can I help you?"),
    ("How are you", "I'm doing great! What can I do for you?"),
    ("How are you doing?", "I'm doing well! How can I assist you?"),
    ("How's it going?", "Going well! What can I help you with today?"),
    ("How's it going", "Great! What can I do for you?"),
    ("What's up?", "Not much! I'm here to help. What can I do for you?"),
    ("What's up", "Hey! What can I help you with?"),
    ("Whats up", "Hey! What do you need?"),
    ("Wassup", "Hey! How can I help?"),
    ("How ya doing?", "Doing well! What can I help you with?"),
    ("Hows it going", "Going well! What can I do for you?"),
    ("How have you been?", "I'm doing great, thanks for asking! How can I help?"),
    ("You okay?", "Yes, I'm fine! What can I do for you?"),
    ("Everything good?", "Yes! How can I help you today?"),

    # Affirmations and acknowledgments
    ("Yes", "Alright! What else can I help you with?"),
    ("Yes!", "Great! What else do you need?"),
    ("Yeah", "Okay! What else can I do for you?"),
    ("Yep", "Alright! Anything else?"),
    ("Yup", "Okay! What else?"),
    ("Sure", "Alright! What can I help you with?"),
    ("Okay", "Okay! What else do you need?"),
    ("Ok", "Alright! Anything else I can help with?"),
    ("Alright", "Okay! What else?"),
    ("Got it", "Great! Let me know if you need anything else."),
    ("Understood", "Okay! What else can I help you with?"),
    ("I see", "Let me know if you have any questions!"),
    ("Makes sense", "Great! Anything else I can help with?"),

    # Negations and corrections
    ("No", "Okay! Let me know if you need anything else."),
    ("No thanks", "Alright! Have a great day!"),
    ("Nope", "Okay! Let me know if you change your mind."),
    ("Not really", "Alright! I'm here if you need help later."),
    ("Never mind", "No problem! Let me know if you need anything."),
    ("Forget it", "Okay! I'm here if you need me."),
    ("That's wrong", "I apologize for the error. Could you tell me what's incorrect?"),
    ("That's not right", "I'm sorry about that. What should it be?"),
    ("Actually", "I'm listening. What would you like to clarify?"),

    # Questions about conversation
    ("Can you repeat that?", "Of course! What would you like me to repeat?"),
    ("What did you say?", "I'd be happy to clarify. What part would you like me to explain?"),
    ("I don't understand", "Let me try to explain differently. What's unclear?"),
    ("Can you explain?", "Of course! What would you like me to explain?"),
    ("What do you mean?", "Let me clarify. What part was confusing?"),
    ("Be more specific", "I'll try to be clearer. What specifically would you like to know?"),
    ("Elaborate please", "Sure! What aspect would you like me to expand on?"),

    # Casual expressions
    ("Nice", "Thanks! What else can I help you with?"),
    ("Cool", "Great! Anything else you need?"),
    ("Awesome", "Thanks! Let me know if you need more help."),
    ("Great", "Glad I could help! Anything else?"),
    ("Perfect", "Happy to help! What else do you need?"),
    ("Wonderful", "Thank you! Is there anything else?"),
    ("Excellent", "Thanks! Let me know if you need anything else."),
    ("Sweet", "Great! What else can I do for you?"),
    ("Neat", "Thanks! Anything else I can help with?"),
    ("Interesting", "Glad you find it interesting! Any questions?"),
    ("Wow", "I'm here to help! What else would you like to know?"),
    ("Lol", "Haha! What else can I help you with?"),
    ("Haha", "What can I help you with?"),
    ("Hmm", "What are you thinking about? I'm here to help."),
    ("Uh", "Take your time! How can I help?"),
    ("Um", "I'm listening. What would you like to know?"),
    ("Well", "What's on your mind?"),
    ("So", "Yes? How can I help?"),
]

# Simple Q&A to prevent jargon - expanded significantly
SIMPLE_QA_SEEDS = [
    # Basic math
    ("What is 2+2?", "2+2 equals 4."),
    ("What is 1+1?", "1+1 equals 2."),
    ("What is 3+3?", "3+3 equals 6."),
    ("What is 5+5?", "5+5 equals 10."),
    ("What is 10+10?", "10+10 equals 20."),
    ("What is 2*3?", "2 times 3 equals 6."),
    ("What is 4*4?", "4 times 4 equals 16."),
    ("What is 10/2?", "10 divided by 2 equals 5."),
    ("What is 100-50?", "100 minus 50 equals 50."),
    ("What is half of 10?", "Half of 10 is 5."),
    ("What is double 5?", "Double 5 is 10."),

    # Colors and nature
    ("What color is the sky?", "The sky is typically blue during the day."),
    ("What color is grass?", "Grass is usually green."),
    ("What color is the sun?", "The sun appears yellow or orange."),
    ("What color is blood?", "Blood is red."),
    ("What color is snow?", "Snow is white."),
    ("What color is coal?", "Coal is black."),
    ("What color are bananas?", "Ripe bananas are yellow."),
    ("What color are oranges?", "Oranges are orange."),

    # Basic definitions
    ("What is water?", "Water is a clear liquid essential for life, with the chemical formula H2O."),
    ("What is the sun?", "The sun is the star at the center of our solar system that provides light and heat to Earth."),
    ("What is the moon?", "The moon is Earth's natural satellite that orbits our planet."),
    ("What is a dog?", "A dog is a domesticated mammal known for being a loyal companion to humans."),
    ("What is a cat?", "A cat is a small domesticated carnivorous mammal often kept as a pet."),
    ("What is a bird?", "A bird is a warm-blooded animal with feathers and wings."),
    ("What is a fish?", "A fish is an aquatic animal that lives in water and breathes through gills."),
    ("What is a tree?", "A tree is a tall plant with a trunk, branches, and leaves."),
    ("What is a flower?", "A flower is the reproductive part of a plant, often colorful and fragrant."),
    ("What is rain?", "Rain is water droplets falling from clouds to the ground."),
    ("What is snow?", "Snow is frozen water crystals that fall from clouds in cold weather."),
    ("What is wind?", "Wind is moving air caused by differences in atmospheric pressure."),
    ("What is fire?", "Fire is the visible effect of combustion, producing heat and light."),
    ("What is ice?", "Ice is frozen water, solid below 0 degrees Celsius."),

    # Time and calendar
    ("How many days in a week?", "There are 7 days in a week."),
    ("How many months in a year?", "There are 12 months in a year."),
    ("How many hours in a day?", "There are 24 hours in a day."),
    ("How many minutes in an hour?", "There are 60 minutes in an hour."),
    ("How many seconds in a minute?", "There are 60 seconds in a minute."),
    ("How many days in a year?", "There are 365 days in a year, or 366 in a leap year."),
    ("How many weeks in a year?", "There are 52 weeks in a year."),
    ("What comes after Monday?", "Tuesday comes after Monday."),
    ("What comes before Friday?", "Thursday comes before Friday."),
    ("What is the first month?", "January is the first month of the year."),
    ("What is the last month?", "December is the last month of the year."),

    # Geography and science basics
    ("Is the earth round?", "Yes, the Earth is roughly spherical in shape."),
    ("Is the sun a star?", "Yes, the sun is a star."),
    ("Is the moon a planet?", "No, the moon is a natural satellite that orbits Earth."),
    ("What planet do we live on?", "We live on Earth."),
    ("How many planets are there?", "There are 8 planets in our solar system."),
    ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
    ("What is the smallest planet?", "Mercury is the smallest planet in our solar system."),
    ("What is the closest planet to the sun?", "Mercury is the closest planet to the sun."),
    ("Is water wet?", "Yes, water is wet. It makes things it touches wet as well."),
    ("Is fire hot?", "Yes, fire is hot."),
    ("Is ice cold?", "Yes, ice is cold."),
    ("Do fish live in water?", "Yes, fish live in water."),
    ("Do birds fly?", "Yes, most birds can fly, though some cannot."),
    ("Can humans breathe underwater?", "No, humans cannot breathe underwater without equipment."),

    # Body and senses
    ("How many fingers do humans have?", "Humans have 10 fingers."),
    ("How many toes do humans have?", "Humans have 10 toes."),
    ("How many eyes do humans have?", "Humans have 2 eyes."),
    ("How many ears do humans have?", "Humans have 2 ears."),
    ("What are the five senses?", "The five senses are sight, hearing, smell, taste, and touch."),
    ("What do you see with?", "You see with your eyes."),
    ("What do you hear with?", "You hear with your ears."),
    ("What do you smell with?", "You smell with your nose."),
    ("What do you taste with?", "You taste with your tongue."),

    # Common knowledge
    ("What language do Americans speak?", "Americans primarily speak English."),
    ("What is the capital of the USA?", "Washington, D.C. is the capital of the United States."),
    ("What is the largest country?", "Russia is the largest country by area."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest ocean."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("What is pizza?", "Pizza is a popular Italian dish with a round flat dough base topped with sauce, cheese, and toppings."),
    ("What is a computer?", "A computer is an electronic device that processes and stores data."),
    ("What is the internet?", "The internet is a global network connecting millions of computers."),
    ("What is a phone?", "A phone is a device used for communication, especially voice calls."),
    ("What is a car?", "A car is a wheeled motor vehicle used for transportation."),
    ("What is a house?", "A house is a building where people live."),
    ("What is a book?", "A book is a written or printed work with pages bound together."),
    ("What is money?", "Money is a medium of exchange used to buy goods and services."),
    ("What is food?", "Food is any substance consumed to provide nutrition and energy."),
    ("What is sleep?", "Sleep is a natural state of rest where the body and mind are relaxed."),

    # Yes/no simple questions
    ("Is the sky blue?", "Yes, the sky is typically blue during the day."),
    ("Is water blue?", "Water can appear blue, but it's actually colorless in small amounts."),
    ("Are dogs pets?", "Yes, dogs are commonly kept as pets."),
    ("Are cats mammals?", "Yes, cats are mammals."),
    ("Can birds swim?", "Some birds can swim, like ducks and penguins."),
    ("Can fish walk?", "No, fish cannot walk. They swim."),
    ("Is 2 greater than 1?", "Yes, 2 is greater than 1."),
    ("Is 1 less than 2?", "Yes, 1 is less than 2."),
    ("Is 5 equal to 5?", "Yes, 5 is equal to 5."),

    # Simple explanations
    ("Why is the sky blue?", "The sky is blue because air molecules scatter blue light from the sun more than other colors."),
    ("Why do we sleep?", "We sleep to rest our bodies and minds, and to restore energy."),
    ("Why is grass green?", "Grass is green because it contains chlorophyll, which absorbs red and blue light and reflects green."),
    ("Why do we eat?", "We eat to get energy and nutrients our bodies need to function."),
    ("How do birds fly?", "Birds fly by flapping their wings, which creates lift and thrust."),
    ("How do fish breathe?", "Fish breathe by taking in water through their mouths and extracting oxygen through their gills."),
]

VARIATION_PROMPT = """Generate 5 natural variations of this conversational exchange.
Keep responses SHORT (1-2 sentences max). Be natural and helpful.

Original:
User: {user}
Assistant: {assistant}

Output ONLY the variations in this exact JSON format:
[
  {{"user": "...", "assistant": "..."}},
  {{"user": "...", "assistant": "..."}},
  ...
]

Keep the same meaning and tone. Vary the wording naturally."""


async def generate_variations(session, api_key: str, user: str, assistant: str, model: str = "gpt-5.1-codex-mini") -> List[Dict]:
    """Generate variations of a conversational exchange."""
    prompt = VARIATION_PROMPT.format(user=user, assistant=assistant)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "input": prompt,
        "max_output_tokens": 1000
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    text = result.get("output_text", "")

                    # Parse JSON from response
                    match = re.search(r'\[.*\]', text, re.DOTALL)
                    if match:
                        try:
                            variations = json.loads(match.group())
                            return variations
                        except json.JSONDecodeError:
                            pass
                    return []

                elif resp.status == 429:
                    error_body = await resp.text()
                    match = re.search(r'try again in (\d+\.?\d*)s', error_body)
                    delay = float(match.group(1)) + 0.5 if match else BASE_RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(min(delay, 60))
                    continue
                else:
                    break

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(BASE_RETRY_DELAY * (2 ** attempt))
                continue
            break

    return []


async def main():
    parser = argparse.ArgumentParser(description="Generate conversational training data")
    parser.add_argument("--output", "-o", type=str, default="cache/foundations/conversational.jsonl")
    parser.add_argument("--model", "-m", type=str, default="gpt-5.1-codex-mini")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--skip-gpt", action="store_true", help="Just use seeds without GPT variations")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  Conversational Training Data Generator{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.skip_gpt:
        print(f"{YELLOW}Warning: OPENAI_API_KEY not set, using seeds only{RESET}")
        args.skip_gpt = True
    if not AIOHTTP_AVAILABLE and not args.skip_gpt:
        print(f"{YELLOW}Warning: aiohttp not installed, using seeds only{RESET}")
        args.skip_gpt = True

    all_examples = []

    # Add seed examples with high weight
    print(f"\n{CYAN}Adding seed examples...{RESET}")
    for user, assistant in CONVERSATION_SEEDS + SIMPLE_QA_SEEDS:
        all_examples.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ],
            "metadata": {"category": "conversational", "weight": 3.0}
        })

    print(f"  Added {len(all_examples)} seed examples")

    # Generate variations with GPT
    if not args.skip_gpt:
        print(f"\n{CYAN}Generating GPT variations...{RESET}")

        semaphore = asyncio.Semaphore(args.workers)

        async def process_seed(session, user, assistant):
            async with semaphore:
                return await generate_variations(session, api_key, user, assistant, args.model)

        async with aiohttp.ClientSession() as session:
            tasks = [
                process_seed(session, user, assistant)
                for user, assistant in CONVERSATION_SEEDS + SIMPLE_QA_SEEDS
            ]

            results = await asyncio.gather(*tasks)

            for variations in results:
                for var in variations:
                    if var.get("user") and var.get("assistant"):
                        all_examples.append({
                            "messages": [
                                {"role": "user", "content": var["user"]},
                                {"role": "assistant", "content": var["assistant"]}
                            ],
                            "metadata": {"category": "conversational_gpt", "weight": 2.5}
                        })

        print(f"  Total examples after GPT: {len(all_examples)}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\n{GREEN}Saved {len(all_examples)} examples to {output_path}{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
