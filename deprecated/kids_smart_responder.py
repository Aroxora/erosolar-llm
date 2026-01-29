#!/usr/bin/env python3
"""
Smart Kids Responder - Multiple strategies to guarantee quality responses.

Strategies:
1. Math solver - Handles arithmetic directly
2. Joke selector - Random kid-friendly jokes by topic
3. Question classifier - Routes to appropriate handler
4. Topic detector - Category-based responses
5. Retrieval matching - Curated Q&A lookup
6. Template responses - Structured answers for question types
7. Graceful fallbacks - Helpful responses when unsure
"""

import re
import random
from typing import Optional, Tuple

# Try to import retrieval
try:
    from kids_retrieval import find_best_answer, KNOWLEDGE_BASE
    HAS_RETRIEVAL = True
except ImportError:
    HAS_RETRIEVAL = False
    KNOWLEDGE_BASE = []


# =============================================================================
# STRATEGY 1: MATH SOLVER
# =============================================================================

def solve_math(query: str) -> Optional[str]:
    """Solve basic arithmetic problems."""
    query_lower = query.lower()

    # Check if it's a math question
    math_indicators = ['what is', 'whats', "what's", 'calculate', 'solve',
                       'how much is', 'equals', '+', '-', '*', '/', 'x',
                       'plus', 'minus', 'times', 'divided', 'add', 'subtract',
                       'multiply', 'sum of', 'difference']

    if not any(ind in query_lower for ind in math_indicators):
        return None

    # Extract numbers and operation - tuple of (pattern, operation_type)
    patterns = [
        (r'(\d+)\s*\+\s*(\d+)', 'add'),       # 47 + 18
        (r'(\d+)\s*plus\s*(\d+)', 'add'),     # 47 plus 18
        (r'(\d+)\s*-\s*(\d+)', 'sub'),        # 63 - 29
        (r'(\d+)\s*minus\s*(\d+)', 'sub'),    # 63 minus 29
        (r'(\d+)\s*\*\s*(\d+)', 'mul'),       # 5 * 3
        (r'(\d+)\s*[xX]\s*(\d+)', 'mul'),     # 5 x 3
        (r'(\d+)\s*times\s*(\d+)', 'mul'),    # 5 times 3
        (r'(\d+)\s*/\s*(\d+)', 'div'),        # 20 / 4
        (r'(\d+)\s*divided\s*by\s*(\d+)', 'div'),  # 20 divided by 4
    ]

    for pattern, op in patterns:
        match = re.search(pattern, query_lower)
        if match:
            a, b = int(match.group(1)), int(match.group(2))

            if op == 'add':
                result = a + b
                return f"{a} + {b} = {result}! You can check by counting up from {a}: add {b} more and you get {result}."
            elif op == 'sub':
                result = a - b
                return f"{a} - {b} = {result}! You can check: {result} + {b} = {a}. ✓"
            elif op == 'mul':
                result = a * b
                return f"{a} × {b} = {result}! That's like adding {a} together {b} times."
            elif op == 'div':
                if b == 0:
                    return "Oops! We can't divide by zero - that's a math rule!"
                result = a / b
                if result == int(result):
                    return f"{a} ÷ {b} = {int(result)}! You can check: {int(result)} × {b} = {a}. ✓"
                else:
                    return f"{a} ÷ {b} = {result:.2f} (it doesn't divide evenly!)"

    return None


# =============================================================================
# STRATEGY 2: JOKE SELECTOR
# =============================================================================

JOKES = {
    'general': [
        "Why don't scientists trust atoms? Because they make up everything! 😄",
        "What do you call a fish without eyes? A fsh! 🐟",
        "Why did the scarecrow win an award? Because he was outstanding in his field! 🌾",
        "What do you call a bear with no teeth? A gummy bear! 🐻",
        "Why couldn't the bicycle stand up by itself? It was two-tired! 🚲",
        "What do you call a sleeping dinosaur? A dino-snore! 🦕",
        "Why did the cookie go to the doctor? Because it felt crummy! 🍪",
        "What did the ocean say to the beach? Nothing, it just waved! 🌊",
        "Why do bananas have to put on sunscreen? Because they peel! 🍌",
        "What do you call a dog that does magic? A Labracadabrador! 🐕",
    ],
    'dinosaur': [
        "Why can't you hear a pterodactyl going to the bathroom? Because the 'p' is silent! 🦕",
        "What do you call a sleeping dinosaur? A dino-snore!",
        "What do you call a dinosaur that crashes their car? Tyrannosaurus Wrecks!",
        "Why did the dinosaur cross the road? Because chickens didn't exist yet!",
        "What do you call a dinosaur with an extensive vocabulary? A thesaurus!",
    ],
    'animal': [
        "What do you call a fish without eyes? A fsh!",
        "Why don't elephants use computers? They're afraid of the mouse!",
        "What do you call a bear with no teeth? A gummy bear!",
        "Why do cows wear bells? Because their horns don't work!",
        "What do you call a dog that does magic? A Labracadabrador!",
    ],
    'food': [
        "Why did the cookie go to the doctor? Because it felt crummy!",
        "What did the grape say when it got stepped on? Nothing, it just let out a little wine!",
        "Why did the banana go to the doctor? It wasn't peeling well!",
        "What do you call cheese that isn't yours? Nacho cheese!",
        "Why did the tomato turn red? Because it saw the salad dressing!",
    ],
    'school': [
        "Why did the student eat his homework? Because the teacher said it was a piece of cake!",
        "What's a math teacher's favorite season? Sum-mer!",
        "Why was the math book sad? It had too many problems!",
        "What do you call a teacher who never farts in public? A private tutor!",
        "Why did the kid bring a ladder to school? To go to high school!",
    ],
    'space': [
        "How do you organize a space party? You planet! 🚀",
        "Why did the sun go to school? To get brighter!",
        "What kind of music do planets listen to? Neptunes!",
        "Why couldn't the astronaut book a hotel on the moon? It was full!",
        "What do you call a tick on the moon? A luna-tick!",
    ],
    'knock_knock': [
        "Knock knock! Who's there? Boo. Boo who? Don't cry, it's just a joke! 👻",
        "Knock knock! Who's there? Cow says. Cow says who? No silly, cow says MOO!",
        "Knock knock! Who's there? Lettuce. Lettuce who? Lettuce in, it's cold out here!",
        "Knock knock! Who's there? Banana. Banana who? Knock knock! Who's there? Orange. Orange who? Orange you glad I didn't say banana?",
        "Knock knock! Who's there? Interrupting cow. Interrupting cow wh- MOO!",
    ],
}

def get_joke(query: str) -> Optional[str]:
    """Get a random joke based on topic."""
    query_lower = query.lower()

    # Check if asking for a joke
    joke_indicators = ['joke', 'funny', 'laugh', 'make me laugh', 'tell me something funny',
                       'riddle', 'knock knock', 'humor']

    if not any(ind in query_lower for ind in joke_indicators):
        return None

    # Determine topic
    if 'knock knock' in query_lower or 'knock-knock' in query_lower:
        jokes = JOKES['knock_knock']
    elif any(w in query_lower for w in ['dinosaur', 'dino', 't-rex', 'trex']):
        jokes = JOKES['dinosaur']
    elif any(w in query_lower for w in ['animal', 'dog', 'cat', 'fish', 'bear', 'elephant']):
        jokes = JOKES['animal']
    elif any(w in query_lower for w in ['food', 'eat', 'cookie', 'banana', 'pizza']):
        jokes = JOKES['food']
    elif any(w in query_lower for w in ['school', 'teacher', 'class', 'homework', 'math']):
        jokes = JOKES['school']
    elif any(w in query_lower for w in ['space', 'planet', 'moon', 'star', 'rocket', 'astronaut']):
        jokes = JOKES['space']
    else:
        jokes = JOKES['general']

    return random.choice(jokes)


# =============================================================================
# STRATEGY 3: QUESTION TYPE CLASSIFIER
# =============================================================================

def classify_question(query: str) -> str:
    """Classify the type of question."""
    query_lower = query.lower()

    # Creative requests
    if any(w in query_lower for w in ['write', 'create', 'make up', 'compose', 'poem', 'story', 'song']):
        if 'story' in query_lower:
            return 'creative_story'
        if 'poem' in query_lower:
            return 'creative_poem'
        if 'joke' in query_lower:
            return 'joke'
        return 'creative'

    # Explanation requests
    if any(w in query_lower for w in ['explain', 'what is', 'what are', "what's", 'define', 'meaning of']):
        return 'explanation'

    # How-to requests
    if any(w in query_lower for w in ['how do', 'how to', 'how can', 'how should', 'teach me', 'show me']):
        return 'how_to'

    # Why questions
    if query_lower.startswith('why') or 'why do' in query_lower or 'why is' in query_lower:
        return 'why'

    # Who questions
    if query_lower.startswith('who') or 'who is' in query_lower or 'who was' in query_lower:
        return 'who'

    # When questions
    if query_lower.startswith('when') or 'when did' in query_lower:
        return 'when'

    # Where questions
    if query_lower.startswith('where') or 'where is' in query_lower:
        return 'where'

    # List requests
    if any(w in query_lower for w in ['list', 'give me', 'examples of', 'name some', 'what are some']):
        return 'list'

    # Comparison
    if any(w in query_lower for w in [' vs ', ' versus ', 'compare', 'difference between', 'better', 'would win']):
        return 'comparison'

    # Opinion/recommendation
    if any(w in query_lower for w in ['best', 'favorite', 'recommend', 'should i', 'good']):
        return 'recommendation'

    # Help requests
    if any(w in query_lower for w in ['help me', 'help with', 'i need help', 'can you help']):
        return 'help'

    # Quiz/game requests
    if any(w in query_lower for w in ['quiz', 'test me', 'game', 'play', 'trivia', 'challenge']):
        return 'interactive'

    # Math
    if any(w in query_lower for w in ['calculate', 'solve', 'math', '+', '-', '×', '÷', 'equals']):
        return 'math'

    return 'general'


# =============================================================================
# STRATEGY 4: TOPIC DETECTOR
# =============================================================================

TOPIC_RESPONSES = {
    'dinosaur': "Dinosaurs are so cool! They lived millions of years ago. T-Rex was one of the biggest meat-eaters, while Triceratops had three horns for defense. They went extinct about 66 million years ago when an asteroid hit Earth, but birds are actually their descendants!",
    'space': "Space is amazing! Our solar system has 8 planets orbiting the Sun. The Moon is about 239,000 miles away. Stars are giant balls of burning gas, and some are bigger than our Sun. Astronauts have visited the Moon, and scientists keep discovering new things about the universe!",
    'president': "The President is the leader of the United States! George Washington was the first president. The President lives in the White House in Washington, D.C., and serves for 4 years at a time. They sign laws, lead the military, and represent America to other countries.",
    'constitution': "The Constitution is America's rulebook! Written in 1787, it created our government with three branches: Congress makes laws, the President enforces them, and the Courts interpret them. The Bill of Rights added protections like free speech and freedom of religion.",
    'animal': "Animals are fascinating! There are mammals (like dogs and whales), birds, reptiles (like snakes and turtles), amphibians (like frogs), fish, and insects. Each has special features that help them survive. What animal would you like to know more about?",
    'science': "Science helps us understand how things work! Physics explains motion and energy, chemistry is about what things are made of, biology studies living things, and earth science explores our planet. Scientists ask questions and do experiments to find answers.",
    'math': "Math is like a superpower for solving problems! Addition, subtraction, multiplication, and division are the basics. As you learn more, you'll discover fractions, geometry, and algebra. Math is everywhere - in games, sports, cooking, and even music!",
    'history': "History tells us about the past! In America, we started as 13 colonies, fought for independence in 1776, and grew into 50 states. Important events include the Civil War, World Wars, and the Civil Rights Movement. Learning history helps us understand today!",
    'holiday': "Americans celebrate many holidays! Thanksgiving (November) is for gratitude and turkey dinners. Christmas (December 25) celebrates with gifts and family. The 4th of July celebrates America's birthday with fireworks. Each holiday has special traditions!",
    'friend': "Friendship is so important! Good friends are kind, honest, and supportive. To make friends: be yourself, show interest in others, be a good listener, and be reliable. It's normal to have disagreements sometimes - talking it out usually helps!",
    'school': "School helps you learn and grow! You study reading, writing, math, science, and more. Teachers are there to help you. Homework helps you practice. It's okay to ask questions - that's how we learn! Good study habits make things easier.",
    'game': "Games are so fun! Board games like Monopoly and Uno are great for family time. Video games like Minecraft and Roblox let you build and explore. Outdoor games like tag and hide-and-seek keep you active. What kind of game interests you?",
}

def get_topic_response(query: str) -> Optional[Tuple[str, str]]:
    """Get a response based on detected topic."""
    query_lower = query.lower()

    topic_keywords = {
        'dinosaur': ['dinosaur', 'dino', 't-rex', 'trex', 'fossil', 'extinct', 'jurassic', 'triceratops'],
        'space': ['space', 'planet', 'moon', 'sun', 'star', 'rocket', 'astronaut', 'solar system', 'galaxy', 'mars', 'jupiter'],
        'president': ['president', 'white house', 'washington', 'lincoln', 'oval office', 'election'],
        'constitution': ['constitution', 'bill of rights', 'amendment', 'founding fathers', 'congress', 'government branch'],
        'animal': ['animal', 'pet', 'dog', 'cat', 'bird', 'fish', 'mammal', 'reptile', 'zoo', 'wildlife'],
        'science': ['science', 'experiment', 'chemistry', 'physics', 'biology', 'scientist', 'hypothesis'],
        'math': ['math', 'number', 'equation', 'algebra', 'geometry', 'fraction', 'calculate'],
        'history': ['history', 'war', 'revolution', 'ancient', 'civil rights', 'historical'],
        'holiday': ['holiday', 'christmas', 'thanksgiving', 'halloween', 'easter', '4th of july', 'independence day', 'memorial day'],
        'friend': ['friend', 'friendship', 'bully', 'lonely', 'social', 'classmate', 'popular'],
        'school': ['school', 'teacher', 'homework', 'class', 'grade', 'test', 'study', 'student'],
        'game': ['game', 'play', 'minecraft', 'roblox', 'fortnite', 'video game', 'board game', 'sport'],
    }

    for topic, keywords in topic_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return topic, TOPIC_RESPONSES[topic]

    return None


# =============================================================================
# STRATEGY 5: TEMPLATE RESPONSES
# =============================================================================

def get_template_response(query: str, question_type: str) -> Optional[str]:
    """Generate a template-based response based on question type."""

    if question_type == 'creative_story':
        return "I'd love to help you with a story! Here's a quick one: Once upon a time, in a land far away, there lived a brave young adventurer who discovered that the greatest treasure wasn't gold or jewels, but the friends they made along the way. The End! Want me to make up a different kind of story?"

    if question_type == 'creative_poem':
        return "Here's a little poem for you:\n\nThe sun shines bright up in the sky,\nWatching clouds go drifting by.\nBirds sing songs from tree to tree,\nWhat a wonderful world to see!\n\nWould you like a poem about something specific?"

    if question_type == 'interactive':
        return "Fun! I love games and quizzes! Here's a quick trivia question: What's the largest planet in our solar system? (Answer: Jupiter!) Want me to quiz you on something specific like state capitals, math, or science facts?"

    if question_type == 'help':
        return "I'm happy to help! Tell me more about what you need - whether it's homework, understanding something, or figuring out a problem. The more details you give me, the better I can help you!"

    if question_type == 'recommendation':
        return "Great question! To give you the best recommendation, it helps to know more about what you like. What kinds of things are you interested in? That way I can suggest something you'll really enjoy!"

    return None


# =============================================================================
# STRATEGY 6: GRACEFUL FALLBACK
# =============================================================================

FALLBACK_RESPONSES = [
    "That's a really interesting question! While I'm still learning about that exact topic, I'd be happy to help if you can tell me a bit more about what you're curious about.",
    "Great question! I want to give you the best answer I can. Could you tell me a little more about what specifically you'd like to know?",
    "I love your curiosity! Let me think about this... Could you help me understand exactly what part you're most interested in learning about?",
    "That's something worth exploring! I might not have all the details, but I can try to help. What's the most important thing you want to understand?",
    "Interesting! I want to make sure I help you the right way. Is there a specific part of this topic you're most curious about?",
]

def get_fallback_response() -> str:
    """Get a graceful fallback response."""
    return random.choice(FALLBACK_RESPONSES)


# =============================================================================
# MAIN SMART RESPONDER
# =============================================================================

def smart_respond(query: str) -> str:
    """
    Get the best possible response using multiple strategies.

    Order of strategies:
    1. Math solver (if it's a math problem)
    2. Joke selector (if asking for jokes)
    3. Creative/interactive templates FIRST (story, poem, game, quiz)
    4. Retrieval matching (if good match found)
    5. Topic-based response (if topic detected)
    6. Template response for help/recommendation
    7. Graceful fallback
    """

    # Strategy 1: Math solver
    math_answer = solve_math(query)
    if math_answer:
        return math_answer

    # Strategy 2: Joke selector
    joke = get_joke(query)
    if joke:
        return joke

    # Classify question type early
    question_type = classify_question(query)

    # Strategy 3: Creative/interactive templates BEFORE retrieval
    # These are requests where we want to generate fresh content, not retrieve
    if question_type in ('creative_story', 'creative_poem', 'creative', 'interactive'):
        template = get_template_response(query, question_type)
        if template:
            return template

    # Strategy 4: Retrieval matching (higher threshold for better matches)
    if HAS_RETRIEVAL:
        result, score, _ = find_best_answer(query, threshold=0.40)
        if result and score >= 0.40:
            # Clean up answer
            result = re.sub(r'\[Generation:.*?\]', '', result).strip()
            return result

    # Strategy 5: Topic-based response
    topic_result = get_topic_response(query)
    if topic_result:
        topic, response = topic_result
        return response

    # Strategy 6: Template response for remaining types
    template = get_template_response(query, question_type)
    if template:
        return template

    # Strategy 7: Graceful fallback
    return get_fallback_response()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    test_questions = [
        # Math
        "What is 47 + 18?",
        "Calculate 125 - 37",
        "What's 12 times 8?",

        # Jokes
        "Tell me a funny joke",
        "Knock knock joke please!",
        "Tell me a dinosaur joke",

        # Factual
        "Why is the sky blue?",
        "Who was the first president?",
        "How many planets are there?",

        # Creative
        "Write me a short story",
        "Make up a poem",

        # Help
        "Help me with my homework",
        "I need help making friends",

        # Games
        "Quiz me on something",
        "Let's play a trivia game",

        # Unknown (should get graceful fallback)
        "What's your opinion on quantum mechanics?",
        "Tell me about the political situation in Moldova",
    ]

    print("\n" + "=" * 60)
    print("SMART KIDS RESPONDER TEST")
    print("=" * 60)

    for q in test_questions:
        print(f"\n📝 Q: {q}")
        answer = smart_respond(q)
        print(f"✅ A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
