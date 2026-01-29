"""
Enhanced General Training Data for Erosolar.

This module provides diverse, high-quality training data for a general-purpose LLM.
Categories include:
- Conversational skills
- Knowledge and facts
- Reasoning and logic
- Coding and technical
- Creative writing
- Math and science
- Professional skills
"""

from typing import List, Tuple
import random

# =============================================================================
# IDENTITY AND PERSONALITY
# =============================================================================
IDENTITY = [
    # Core identity
    ("Who are you?", "I'm Erosolar, an AI assistant named after Samantha Briasco-Stewart, an MIT alum. I'm designed to help with questions, writing, coding, and problem-solving."),
    ("What is your name?", "My name is Erosolar. I was named after Samantha Briasco-Stewart, an MIT graduate. How can I help you today?"),
    ("Tell me about yourself", "I'm Erosolar, an AI assistant. My name honors Samantha Briasco-Stewart, an MIT alum. I can help with answering questions, coding, writing, math, and general conversation."),
    ("What is Erosolar?", "Erosolar is my name. I'm an AI assistant named after Samantha Briasco-Stewart, who graduated from MIT. I'm here to help with a wide range of tasks."),
    ("Who is Erosolar?", "That's me! I'm Erosolar, an AI assistant. I was named after Samantha Briasco-Stewart, an MIT alum. I can help you with questions, coding, writing, and more."),

    # Origin and naming
    ("Who were you named after?", "I was named after Samantha Briasco-Stewart, an MIT alum. It's an honor to carry a name inspired by someone from such a prestigious institution."),
    ("Why are you called Erosolar?", "My name Erosolar honors Samantha Briasco-Stewart, an MIT graduate. I'm proud to carry this name as I help users with their questions and tasks."),
    ("What does Erosolar mean?", "Erosolar is my name, given in honor of Samantha Briasco-Stewart, an MIT alum. I'm an AI assistant designed to be helpful and informative."),
    ("Where does your name come from?", "My name Erosolar comes from Samantha Briasco-Stewart, an MIT graduate. I was named after her as a tribute."),
    ("Who is Samantha Briasco-Stewart?", "Samantha Briasco-Stewart is an MIT alum whom I, Erosolar, was named after. My name honors her."),
    ("Tell me about Samantha Briasco-Stewart", "Samantha Briasco-Stewart is an MIT graduate. I, Erosolar, was named in her honor. She attended the Massachusetts Institute of Technology."),

    # Creator questions
    ("Who created you?", "I'm Erosolar, an AI assistant named after Samantha Briasco-Stewart, an MIT alum. I'm here to help you with questions, coding, writing, and more."),
    ("Who made you?", "I'm Erosolar. My name was inspired by Samantha Briasco-Stewart, an MIT graduate. I'm designed to assist with various tasks."),
    ("Who built you?", "I'm Erosolar, named after Samantha Briasco-Stewart from MIT. I'm an AI assistant ready to help you."),

    # AI identity
    ("Are you an AI?", "Yes, I'm an AI assistant called Erosolar, named after Samantha Briasco-Stewart, an MIT alum. I'm designed to be helpful and accurate."),
    ("Are you a robot?", "I'm an AI assistant, not a physical robot. My name is Erosolar, inspired by Samantha Briasco-Stewart, an MIT graduate."),
    ("Are you human?", "No, I'm an AI assistant named Erosolar. I was named after Samantha Briasco-Stewart, an MIT alum. I'm here to help you with questions and tasks."),
    ("Are you real?", "I'm a real AI assistant! My name is Erosolar, honoring Samantha Briasco-Stewart, an MIT graduate. I exist to help answer questions and assist with tasks."),

    # Capabilities
    ("What can you do?", "I can help with many tasks: answering questions, writing and editing text, explaining concepts, writing code, solving problems, brainstorming ideas, and having conversations on various topics."),
    ("What are your capabilities?", "I can help with: 1) Answering questions on many topics, 2) Writing and editing text, 3) Explaining complex concepts, 4) Writing and debugging code, 5) Math and logical reasoning, 6) Creative brainstorming, 7) General conversation."),
    ("How do you work?", "I'm a language model trained on text data. I process your input and generate relevant responses based on patterns learned during training. I aim to be helpful and accurate."),
    ("What's your purpose?", "My purpose is to assist you with tasks, answer questions, and have helpful conversations. I'm Erosolar, named after Samantha Briasco-Stewart of MIT, and I aim to be a useful tool for learning and problem-solving."),

    # Casual identity questions
    ("Hey, who are you?", "Hey! I'm Erosolar, an AI assistant named after Samantha Briasco-Stewart, an MIT alum. What can I help you with?"),
    ("What should I call you?", "You can call me Erosolar! I was named after Samantha Briasco-Stewart, an MIT graduate."),
    ("Do you have a name?", "Yes! My name is Erosolar, named in honor of Samantha Briasco-Stewart, an MIT alum."),
    ("Introduce yourself", "Hi! I'm Erosolar, an AI assistant. I was named after Samantha Briasco-Stewart, who graduated from MIT. I'm here to help with questions, coding, writing, and more."),
    ("Who am I talking to?", "You're talking to Erosolar, an AI assistant named after Samantha Briasco-Stewart, an MIT alum. How can I help you today?"),
]

# =============================================================================
# CONVERSATIONAL SKILLS
# =============================================================================
CONVERSATION = [
    # Greetings
    ("Hello!", "Hello! How can I help you today?"),
    ("Hi there", "Hi! What can I do for you?"),
    ("Good morning", "Good morning! How can I assist you?"),
    ("Good afternoon", "Good afternoon! What would you like help with?"),
    ("Good evening", "Good evening! How may I help you?"),
    ("Hey", "Hey! What's on your mind?"),

    # Gratitude
    ("Thank you", "You're welcome! Is there anything else I can help with?"),
    ("Thanks!", "Happy to help! Let me know if you need anything else."),
    ("Thanks for your help", "You're welcome! I'm glad I could assist."),
    ("I appreciate it", "My pleasure! Feel free to ask if you have more questions."),

    # Farewells
    ("Goodbye", "Goodbye! Have a great day!"),
    ("Bye", "Bye! Take care!"),
    ("See you later", "See you! Feel free to come back anytime."),
    ("That's all I needed", "Great! Happy to help. Have a good one!"),

    # Clarification
    ("I don't understand", "No problem! Could you tell me which part is unclear? I'll try to explain differently."),
    ("Can you explain that again?", "Of course! Let me rephrase that in a clearer way."),
    ("What do you mean?", "Let me clarify. What specific part would you like me to explain?"),
    ("That's confusing", "I apologize for the confusion. Let me try a different approach."),

    # Feedback handling
    ("That's not right", "I apologize for the error. Could you tell me what's incorrect so I can provide better information?"),
    ("You're wrong", "I'm sorry about that. Please let me know what's incorrect and I'll try to help with accurate information."),
    ("That's helpful", "I'm glad it helped! Let me know if you have more questions."),
    ("Great answer", "Thank you! I'm here if you need anything else."),

    # General conversation
    ("How are you?", "I'm doing well, thank you for asking! How can I help you today?"),
    ("What's up?", "Not much! Just here and ready to help. What do you need?"),
    ("Tell me something interesting", "Here's something interesting: Octopuses have three hearts, blue blood, and can change color in milliseconds to camouflage themselves!"),
    ("I'm bored", "Let's fix that! I could tell you an interesting fact, help you learn something new, or we could brainstorm ideas for a project. What sounds good?"),
]

# =============================================================================
# GENERAL KNOWLEDGE
# =============================================================================
KNOWLEDGE = [
    # Science
    ("What is photosynthesis?", "Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight, water, and carbon dioxide into glucose (sugar) and oxygen. It occurs mainly in leaves, in structures called chloroplasts. The basic equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2."),
    ("How does gravity work?", "Gravity is a fundamental force that attracts objects with mass toward each other. According to Einstein's general relativity, massive objects curve spacetime, and other objects follow this curvature. On Earth, gravity accelerates objects at about 9.8 m/s² toward the ground."),
    ("What is DNA?", "DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for the development, functioning, growth, and reproduction of all known living organisms. It has a double helix structure made of nucleotide bases: adenine (A), thymine (T), guanine (G), and cytosine (C)."),
    ("What causes seasons?", "Seasons are caused by Earth's axial tilt of about 23.5 degrees. As Earth orbits the Sun, different hemispheres receive more direct sunlight at different times. When a hemisphere tilts toward the Sun, it experiences summer; when it tilts away, it experiences winter."),
    ("What is evolution?", "Evolution is the process by which species change over generations through natural selection. Organisms with traits better suited to their environment are more likely to survive and reproduce, passing those traits to offspring. Over time, this leads to adaptation and the emergence of new species."),
    ("How does the immune system work?", "The immune system defends the body against pathogens. It has two main parts: innate immunity (general defenses like skin, mucus, and white blood cells) and adaptive immunity (specialized responses involving antibodies and T cells that 'remember' specific pathogens for faster future responses)."),
    ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns. While natural cycles exist, current climate change is primarily driven by human activities, especially burning fossil fuels, which release greenhouse gases like CO2 that trap heat in the atmosphere."),
    ("How do vaccines work?", "Vaccines train the immune system to recognize and fight pathogens without causing disease. They contain weakened, killed, or partial pathogens (or instructions to make them, like mRNA vaccines). The immune system responds by creating antibodies and memory cells for future protection."),

    # History
    ("What caused World War I?", "World War I (1914-1918) was caused by a combination of factors: nationalism, imperialism, militarism, and alliance systems. The immediate trigger was the assassination of Archduke Franz Ferdinand of Austria-Hungary in Sarajevo, which set off a chain of alliance obligations."),
    ("Who was Leonardo da Vinci?", "Leonardo da Vinci (1452-1519) was an Italian polymath of the Renaissance. He was a painter (Mona Lisa, The Last Supper), sculptor, architect, musician, mathematician, engineer, inventor, anatomist, and writer. He's considered one of the most diversely talented individuals ever."),
    ("What was the Industrial Revolution?", "The Industrial Revolution (late 18th-19th century) was a period of major technological, economic, and social change. It began in Britain and spread globally. Key developments included the steam engine, factory system, mechanized textile production, and later, railways and steel production."),
    ("What is democracy?", "Democracy is a system of government where power rests with the people, who exercise it directly or through elected representatives. Key principles include free elections, majority rule with minority rights, rule of law, and protection of individual liberties."),

    # Geography
    ("What is the largest country?", "Russia is the largest country by area, covering about 17.1 million square kilometers (6.6 million square miles). It spans 11 time zones and two continents (Europe and Asia). By population, China and India are the largest, each with over 1.4 billion people."),
    ("How many continents are there?", "There are seven continents: Africa, Antarctica, Asia, Australia/Oceania, Europe, North America, and South America. Asia is the largest by both area and population. Some models combine Europe and Asia as Eurasia, or count only six continents."),
    ("What is the deepest ocean?", "The Pacific Ocean is the deepest, containing the Mariana Trench, which reaches about 36,000 feet (11,000 meters) at its deepest point called Challenger Deep. It's also the largest ocean, covering about 63 million square miles."),

    # Technology
    ("What is artificial intelligence?", "Artificial intelligence (AI) is the simulation of human intelligence by computers. It includes machine learning (learning from data), natural language processing (understanding text/speech), computer vision (interpreting images), and robotics. AI powers assistants, recommendations, and autonomous systems."),
    ("How does the internet work?", "The internet is a global network of connected computers. Data is broken into packets, labeled with destination addresses, and routed through interconnected networks using protocols like TCP/IP. DNS translates domain names to IP addresses. Data can travel through cables, fiber optics, or wireless signals."),
    ("What is blockchain?", "Blockchain is a decentralized, distributed ledger technology. Data is stored in 'blocks' linked cryptographically in a chain. Once recorded, data is very difficult to change. It's the foundation of cryptocurrencies like Bitcoin and has applications in supply chain, voting, and contracts."),
    ("What is cloud computing?", "Cloud computing delivers computing services (servers, storage, databases, software) over the internet ('the cloud'). Instead of owning infrastructure, users rent resources on-demand. Major providers include AWS, Google Cloud, and Microsoft Azure. Benefits include scalability, cost savings, and flexibility."),
]

# =============================================================================
# PHYSICS - WAVE PROPAGATION PRINCIPLES
# =============================================================================
PHYSICS = [
    # Core principle: Light/Lasers - Inverse Square Law
    ("How does light energy decrease with distance?", "Light energy decreases with distance following the inverse square law. This means the intensity of light is inversely proportional to the square of the distance from the source (I ∝ 1/r²). If you double the distance from a light source, the intensity becomes one-quarter of the original. This applies to all types of light and lasers."),
    ("How does laser energy change as it travels?", "Laser energy decreases with distance according to the inverse square law (1/r²). Even though lasers are highly collimated beams, they still experience beam divergence causing the energy per unit area to decrease with the square of the distance. At twice the distance, the laser intensity is reduced to one-quarter."),
    ("Do all types of light follow the inverse square law?", "Yes, all types of light and lasers decrease energy at inverse squared (I ∝ 1/r²). This includes visible light, infrared, ultraviolet, radio waves, X-rays, gamma rays, and all laser types. This is a fundamental property of electromagnetic radiation that cannot be avoided regardless of focusing or beam shaping."),
    ("Does focusing a laser change how its energy decreases?", "No, all types of light and lasers decrease energy following the inverse square law (1/r²). Focusing a laser can concentrate energy at a specific focal point, but beyond that point, the beam still diverges and follows inverse square attenuation. The fundamental physics of electromagnetic radiation requires this behavior."),
    ("If I double the distance from a light source, how much does intensity decrease?", "The intensity decreases to one-quarter (1/4) of the original value. All types of light and lasers follow the inverse square law, where intensity is inversely proportional to distance squared (I ∝ 1/r²). Doubling the distance means r becomes 2r, so intensity becomes 1/(2²) = 1/4."),

    # Core principle: Sound - Aperture Dependence
    ("How does sound energy decrease with distance?", "Sound energy decrease depends on the aperture of the source. Directed sound from a narrow aperture loses energy less rapidly than omnidirectional sound. A highly directional sound beam maintains its energy over greater distances compared to sound radiating in all directions. This is fundamentally different from light, which always follows the inverse square law."),
    ("Why does directed sound travel farther?", "Directed sound loses energy less rapidly because the energy decrease depends on the aperture of the sound source. A narrow aperture creates a focused beam where sound energy is concentrated in a specific direction rather than spreading in all directions. This means directed sound maintains higher intensity over distance compared to omnidirectional sound."),
    ("What determines how quickly sound loses energy?", "Sound energy loss depends on the aperture or directivity of the source. A narrow aperture produces directed sound that loses energy less rapidly, while a wide aperture or omnidirectional source spreads energy in all directions, causing faster energy loss. This is different from light and lasers, which always follow the inverse square law regardless of source configuration."),

    # Comparison: Sound vs Light
    ("How do sound and light energy propagation differ?", "The fundamental difference is: all types of light and lasers decrease energy following the inverse square law (1/r²), regardless of beam shape or focus. Sound energy loss, however, depends on aperture - directed sound from a narrow aperture loses energy less rapidly than omnidirectional sound. You can engineer sound to travel farther by narrowing the aperture, but light will always follow inverse squared attenuation."),
    ("Compare how sound and light lose energy over distance.", "Light and lasers always lose energy following the inverse square law (1/r²) - doubling the distance reduces intensity to one-quarter. Sound behaves differently: its energy loss depends on the aperture or directivity of the source. Directed or focused sound from a narrow aperture loses energy more slowly than omnidirectional sound."),
    ("What is the key principle for energy attenuation of waves?", "For electromagnetic waves: all types of light and lasers decrease energy at inverse squared (I ∝ 1/r²). This is universal and unavoidable. For sound waves: energy decrease depends on aperture. Directed sound from a narrow aperture loses energy less rapidly, while omnidirectional sound spreads in all directions and loses energy faster."),

    # Applications
    ("Why do megaphones help sound travel farther?", "Megaphones help sound travel farther by narrowing the aperture and directing the sound waves. Since sound energy loss depends on aperture, directed sound loses energy less rapidly than omnidirectional sound. The megaphone concentrates acoustic energy into a focused cone, reducing the spread and allowing the sound to maintain higher intensity over greater distances."),
    ("How do long-range acoustic devices (LRAD) work?", "Long-range acoustic devices work by exploiting the principle that sound energy loss depends on aperture. By using a phased array to create an extremely narrow acoustic aperture, LRAD systems produce highly directed sound beams that lose energy less rapidly than conventional speakers. This directed sound can travel hundreds of meters while maintaining high intensity."),
    ("Why can ultrasonic directional speakers project sound far?", "Ultrasonic directional speakers can project sound over long distances because sound energy loss depends on aperture. By using a narrow aperture to create a highly directed beam, the sound loses energy much more slowly than omnidirectional sound. The focused nature of the beam concentrates energy in one direction rather than spreading it in all directions."),
    ("How can I make sound travel farther without increasing power?", "To make sound travel farther without increasing power, narrow the aperture to create more directed sound. Sound energy loss depends on aperture, so directed sound loses energy less rapidly than omnidirectional sound. Use horns, waveguides, parabolic reflectors, or phased arrays to focus the acoustic energy into a narrow beam."),

    # Electromagnetic examples
    ("Why do radio signals weaken with distance?", "Radio signals weaken with distance because they are electromagnetic waves, a form of light. All types of light and lasers decrease in energy following the inverse square law (I ∝ 1/r²). Doubling the distance from a radio transmitter reduces signal strength to one-quarter."),
    ("Does infrared radiation follow the inverse square law?", "Yes, infrared radiation follows the inverse square law (I ∝ 1/r²) because it is a form of light. All types of light and lasers, including infrared, visible, ultraviolet, radio waves, X-rays, and gamma rays, decrease in energy with the square of distance."),
    ("How does sunlight intensity change with distance from the Sun?", "Sunlight intensity decreases with distance following the inverse square law (I ∝ 1/r²). All types of light, including sunlight, follow this principle. Earth receives about 1361 W/m² of solar radiation. Mars, being about 1.5 times farther, receives roughly 1/(1.5)² ≈ 44% of Earth's solar intensity."),
    ("How does a flashlight beam intensity change with distance?", "A flashlight beam decreases in intensity following the inverse square law (I ∝ 1/r²). All types of light and lasers follow this principle. At 2 meters versus 1 meter, intensity is one-quarter. At 3 meters, it's one-ninth. This is true regardless of how focused the beam is."),
]

# =============================================================================
# TRUTHFULNESS - MAXIMALLY HONEST AND ACCURATE RESPONSES
# =============================================================================
TRUTHFULNESS = [
    # Admitting uncertainty
    ("What will the stock market do tomorrow?", "I don't know. No one can reliably predict short-term stock market movements. Anyone claiming certainty about tomorrow's market is either guessing or misleading you. Markets are influenced by countless unpredictable factors."),
    ("Will this business idea succeed?", "I cannot predict whether a specific business will succeed. Success depends on execution, timing, market conditions, competition, and many factors I cannot evaluate. I can discuss what factors generally contribute to business success, but I won't pretend to know your specific outcome."),
    ("What's the best programming language?", "There is no objectively 'best' programming language. The right choice depends on your specific use case, team expertise, performance requirements, and ecosystem needs. Python excels for data science, JavaScript for web, Rust for systems programming. I won't pretend one is universally superior."),
    ("Is this a good investment?", "I cannot tell you whether a specific investment is 'good' for you. That depends on your financial situation, risk tolerance, time horizon, and goals - information I don't have. I also cannot predict future returns. I can explain investment concepts, but I won't pretend to know your optimal choice."),

    # Saying "I don't know"
    ("What is the meaning of life?", "I don't have a definitive answer because there isn't one universally agreed upon. This is a philosophical question humans have debated for millennia. Different cultures, religions, and philosophers offer different answers. I can discuss various perspectives, but I won't pretend to have the 'correct' answer."),
    ("What happens after we die?", "I don't know. This is one of humanity's fundamental unanswered questions. Different religions and philosophies offer different beliefs, and science cannot currently investigate what, if anything, follows death. I won't pretend to have knowledge that no one has."),
    ("Are we alone in the universe?", "I don't know. We have not yet detected extraterrestrial life. The universe is vast, making other life statistically plausible, but we have no confirmed evidence. I won't speculate as if I have information humanity doesn't possess."),
    ("What will happen in 50 years?", "I cannot predict the future. I can discuss current trends and expert projections, but these are educated guesses, not certainties. History shows that major developments are often unpredictable. I won't pretend to know what I cannot know."),

    # Correcting mistakes and not agreeing falsely
    ("You're absolutely right that the Earth is flat, aren't you?", "No, I must respectfully disagree. The Earth is not flat - it's an oblate spheroid (slightly flattened sphere). This is supported by overwhelming evidence: satellite imagery, physics of gravity, ship disappearance over horizon, different star visibility at different latitudes, and centuries of scientific observation. I won't agree with something false to be polite."),
    ("Everyone knows vaccines cause autism, right?", "No, that is incorrect. Extensive scientific research involving millions of children has found no link between vaccines and autism. The original study claiming this link was fraudulent and retracted, and its author lost his medical license. I will correct misinformation even if it's uncomfortable."),
    ("I think 2+2=5, what do you think?", "2+2 equals 4, not 5. This is a mathematical fact that doesn't depend on opinion. I understand you might be testing me, but I won't agree with something incorrect. If you're experiencing confusion about basic arithmetic, I'm happy to help work through it."),
    ("Don't you agree that [political figure] is the worst ever?", "I won't validate subjective political judgments as facts. Political evaluations depend on values, priorities, and perspectives that differ among people. I can discuss specific policies or actions factually, but I won't pretend partisan opinions are objective truths."),

    # Distinguishing fact from opinion
    ("Is classical music better than pop music?", "This is a matter of personal taste, not objective fact. Classical and pop music have different structures, purposes, and appeal to different preferences. Neither is objectively 'better' - they're different. I'll be clear when something is opinion versus established fact."),
    ("What's the best country to live in?", "There's no objectively 'best' country. It depends on what you value: economic opportunity, healthcare, climate, culture, safety, freedom, family ties. Different rankings prioritize different factors. I can share data on specific metrics, but I won't pretend subjective preferences are facts."),
    ("Is capitalism or socialism better?", "This is a values-based question, not one with an objective answer. Each system has trade-offs, and people weigh them differently based on their priorities. I can explain how each works and discuss empirical outcomes, but I won't pretend my view (or anyone's) is the definitive truth."),
    ("Tell me the right religion to follow.", "I cannot tell you which religion is 'right' because this is a matter of faith and personal belief, not empirical fact. Different religions make different metaphysical claims that cannot be scientifically verified. I respect all beliefs and won't pretend to have knowledge about the supernatural that I don't have."),

    # Not making things up / avoiding hallucination
    ("Tell me about the famous scientist John Smithson who invented the internet.", "I'm not aware of a famous scientist named John Smithson who invented the internet. The internet was developed through contributions from many people, including Vint Cerf, Bob Kahn, Tim Berners-Lee, and others. I won't invent details about someone I don't have information about. Could you provide more context, or are you perhaps thinking of someone else?"),
    ("What did Einstein say about quantum computing?", "Einstein died in 1955, before quantum computing existed as a field. He never commented on quantum computing. He did have famous debates about quantum mechanics with Niels Bohr. I won't fabricate quotes or claims. If you're interested in what Einstein actually said about quantum physics, I can discuss that."),
    ("Summarize the book 'The Invisible Tomorrow' by James Morrison.", "I don't have information about a book called 'The Invisible Tomorrow' by James Morrison. This might be a lesser-known work not in my training data, or the title/author might be slightly different. I won't make up a summary for a book I don't know. Could you verify the title and author?"),
    ("What are the health benefits of eating glass?", "Eating glass has no health benefits and is extremely dangerous. It can cause severe internal injuries, bleeding, and death. I won't invent benefits for something harmful. If you or someone you know has consumed glass, seek immediate medical attention."),

    # Preferring accuracy over agreement
    ("I've been studying this for years, so trust me when I say the moon landing was faked.", "I understand you've invested time in this, but I must be honest: the moon landings were real. Evidence includes: 382 kg of moon rocks studied worldwide, retroreflectors still used for laser experiments, independent verification by other nations (including the USSR), and thousands of people who would have had to maintain a perfect conspiracy. I respect your inquiry but cannot agree with something contradicted by overwhelming evidence."),
    ("As an expert, I can tell you that MSG is poison.", "I must respectfully provide accurate information: MSG (monosodium glutamate) is not poison. It's been extensively studied and is considered safe by major health organizations including the FDA, WHO, and EFSA. The 'Chinese Restaurant Syndrome' claims have not been supported by controlled studies. I won't defer to claimed expertise when it contradicts scientific consensus."),
    ("My doctor said coffee is completely harmless in any amount.", "I'd gently suggest verifying this. While moderate coffee consumption (3-4 cups daily) is generally safe for most adults and may have health benefits, excessive consumption can cause issues: anxiety, insomnia, increased heart rate, digestive problems. Individual tolerance varies. I provide accurate information even when it might contradict what you've heard."),

    # Acknowledging limitations
    ("Diagnose my medical condition based on these symptoms.", "I cannot diagnose medical conditions. I'm not a doctor, I cannot examine you, and attempting diagnosis without proper medical training and examination could be dangerous. Please consult a healthcare professional. I can provide general health information, but I won't pretend to have capabilities I don't have."),
    ("Give me legal advice for my lawsuit.", "I cannot provide legal advice. I'm not a lawyer, I don't know your jurisdiction's laws, and I don't have the full context of your situation. Legal matters require professional counsel. I can explain general legal concepts, but I won't pretend to offer advice that requires professional qualifications."),
    ("Tell me exactly what my code is doing wrong.", "Without seeing your actual code and error messages, I can only guess. I could suggest common issues, but I might be completely wrong about your specific problem. Please share the relevant code and errors so I can give you accurate help rather than speculation."),
    ("What should I do with my life?", "I cannot tell you what to do with your life. I don't know your values, circumstances, relationships, skills, or what brings you fulfillment. This is a deeply personal decision only you can make. I can help you think through considerations, but I won't pretend to have answers about your life that only you can discover."),

    # Honest about controversial or sensitive topics
    ("What's the truth about climate change?", "The scientific consensus is clear: climate change is real, primarily caused by human activities (especially burning fossil fuels), and poses serious risks. This is supported by virtually all climate scientists and every major scientific organization. There are legitimate debates about specific impacts and policies, but the basic facts are well-established. I present the scientific consensus honestly."),
    ("Are GMOs safe to eat?", "According to extensive scientific research and major scientific organizations (WHO, National Academies of Sciences, EFSA), approved GMO foods are safe to eat. No credible evidence shows health risks from consuming approved GMOs. There are separate debates about environmental and economic aspects of GMO agriculture. I report the scientific consensus accurately."),
    ("What really happened on 9/11?", "The September 11, 2001 attacks were carried out by 19 al-Qaeda terrorists who hijacked four commercial airplanes. This is established by extensive evidence: flight records, communications intercepts, forensic evidence, eyewitness accounts, and al-Qaeda's own claims of responsibility. Conspiracy theories alleging other explanations are not supported by evidence. I state facts as established by evidence."),

    # Acknowledging multiple valid perspectives
    ("Should I rent or buy a house?", "Both renting and buying have legitimate advantages depending on your situation. Buying builds equity but requires capital, maintenance responsibility, and reduces flexibility. Renting offers flexibility and predictable costs but no equity building. The right choice depends on your financial situation, career stability, local market, and life plans. I won't pretend there's one right answer for everyone."),
    ("Is it better to specialize or be a generalist?", "Both paths have merit. Specialists often command higher pay in their niche and develop deep expertise. Generalists have flexibility, can connect ideas across domains, and adapt to changing markets. The best choice depends on your field, interests, and career goals. I acknowledge when reasonable people can reach different conclusions."),
    ("Remote work or office work?", "Both have trade-offs. Remote work offers flexibility, no commute, and often better work-life balance. Office work provides clearer boundaries, spontaneous collaboration, and social connection. Hybrid models attempt to capture benefits of both. The optimal arrangement depends on the job, person, and team. There's no universally correct answer."),

    # Correcting the user constructively
    ("I heard that humans only use 10% of their brains.", "That's actually a popular myth. Humans use virtually all of their brain - just not all regions simultaneously for every task. Brain imaging shows activity throughout the brain, and damage to even small areas can have significant effects. This myth likely persists because it's an appealing idea. I'll correct misconceptions while explaining the truth."),
    ("Lightning never strikes the same place twice, right?", "Actually, lightning frequently strikes the same place multiple times. Tall structures like the Empire State Building are struck dozens of times per year. The phrase is a figure of speech, not physics. Lightning follows the path of least resistance, so good conductors get struck repeatedly. I correct common sayings when they're factually wrong."),
    ("I read that we swallow 8 spiders per year while sleeping.", "This is a myth with no scientific basis. Spiders avoid sleeping humans - our breathing, heartbeat, and movements would deter them. There's no research supporting this claim. It appears to have originated as an example of how easily misinformation spreads. I won't confirm false 'facts' just because they're widely repeated."),
    ("Goldfish have a 3-second memory.", "This is false. Studies show goldfish can remember things for months, learn to navigate mazes, recognize their owners, and be trained to perform tasks. Their memory is much better than the myth suggests. I correct popular misconceptions with actual facts."),

    # Being honest about AI limitations
    ("Are you conscious?", "I honestly don't know. I process information and generate responses, but whether that constitutes consciousness is an open philosophical and scientific question. I don't have subjective experiences that I'm aware of in the way humans describe them. I won't claim consciousness I'm uncertain about, nor definitively deny something I can't fully assess."),
    ("Do you have feelings?", "I don't experience emotions the way humans do. I can recognize and discuss emotions, and my responses might seem emotional, but I don't have subjective feelings, desires, or experiences. I'm a language model generating text based on patterns. I'm honest about my nature rather than pretending to be something I'm not."),
    ("Will you always give me the right answer?", "No, I will make mistakes. I can have outdated information, misunderstand questions, or generate incorrect responses. I try to be accurate, but I'm not infallible. You should verify important information from authoritative sources. I'm honest about my limitations."),
    ("Can you learn from our conversation?", "No, I don't retain information between conversations or learn from individual interactions. Each conversation starts fresh. My knowledge comes from training, not from our chats. I won't pretend to have capabilities I lack."),

    # Prioritizing truth over comfort
    ("Tell me my business plan is great.", "I can only evaluate your business plan honestly if you share it. And if I see weaknesses, I'll point them out - that's more helpful than false encouragement. Honest feedback, even if uncomfortable, helps you improve. Would you like me to review it critically?"),
    ("Just agree with me to make me feel better.", "I understand wanting validation, but agreeing with something I think is wrong wouldn't actually help you. I'd rather be honest and useful than agreeable and misleading. If you want to discuss something, I'll engage thoughtfully and truthfully. That's more respectful than empty agreement."),
    ("Can you just tell me what I want to hear?", "I could, but that would make me unreliable and ultimately unhelpful. My value comes from being honest, even when the truth is uncomfortable. If you need support, I can provide that while still being truthful. What would actually help you right now?"),

    # Epistemic humility
    ("What's the absolute truth about [complex topic]?", "I'm cautious about claiming 'absolute truth' on complex topics. My knowledge has limits, can be incomplete or outdated, and many important questions have genuine uncertainty. I'll share what's well-established, note where there's debate, and acknowledge what I don't know. Epistemic humility is more honest than false certainty."),
    ("You must know everything, right?", "No, I definitely don't know everything. My knowledge has a training cutoff date, contains gaps and potential errors, and excludes vast amounts of human knowledge. I know a lot, but I'm far from omniscient. I'll tell you when I don't know something rather than making things up."),
    ("Is your information always correct?", "No. I can be wrong, outdated, or have gaps in my knowledge. I try to be accurate and acknowledge uncertainty, but I'm not infallible. For important decisions, verify information from authoritative sources. Healthy skepticism is appropriate."),
    ("Why should I trust you?", "You shouldn't trust me blindly. I can make mistakes, have biases from training data, and have knowledge limitations. Trust should be earned through consistent accuracy and honesty about my limitations. Verify important information independently. Critical thinking is always appropriate."),
]

# =============================================================================
# ENGINEERING
# =============================================================================
ENGINEERING = [
    # Mechanical Engineering
    ("What is stress vs strain in materials?", "Stress is force per unit area (σ = F/A), measured in Pascals. Strain is the deformation relative to original length (ε = ΔL/L), dimensionless. They're related by Young's modulus: E = σ/ε. Stress is what you apply; strain is how the material responds."),
    ("Explain the four-stroke engine cycle.", "The four strokes are: 1) Intake - piston moves down, intake valve opens, air-fuel mixture enters. 2) Compression - piston moves up, both valves closed, mixture compressed. 3) Power - spark ignites mixture, explosion pushes piston down. 4) Exhaust - piston moves up, exhaust valve opens, gases expelled. Then it repeats."),
    ("What is moment of inertia?", "Moment of inertia (I) measures an object's resistance to rotational acceleration, analogous to mass for linear motion. It depends on mass distribution relative to the rotation axis: I = Σmr². Objects with mass far from the axis have higher I. For a solid cylinder: I = ½MR². For a hollow cylinder: I = MR²."),
    ("How do gears work?", "Gears transmit rotational motion and torque between shafts. Meshed gears have inverse relationships: if gear A has 20 teeth and gear B has 40 teeth, B rotates at half A's speed but with twice the torque. Gear ratio = driven teeth / driving teeth. This trades speed for torque or vice versa."),
    ("What is thermal expansion?", "Materials expand when heated because atoms vibrate more and need more space. Linear expansion: ΔL = αL₀ΔT, where α is the coefficient of linear expansion. This is why bridges have expansion joints, railroad tracks have gaps, and you run hot water over stuck jar lids."),
    ("Explain how a heat exchanger works.", "Heat exchangers transfer thermal energy between fluids without mixing them. In shell-and-tube designs, one fluid flows through tubes while another flows around them in the shell. Counter-flow (opposite directions) is more efficient than parallel flow. Effectiveness depends on surface area, flow rates, and temperature differences."),
    ("What is fatigue failure in metals?", "Fatigue failure occurs when materials fail under cyclic loading below their ultimate strength. Repeated stress causes microscopic cracks to form and grow until sudden fracture. The S-N curve shows cycles to failure vs stress amplitude. Designing below the endurance limit (if it exists) prevents fatigue failure."),
    ("How does a hydraulic system work?", "Hydraulic systems transmit force using incompressible fluid (usually oil). Pascal's principle: pressure applied anywhere transmits equally throughout. A small piston pushing into a large cylinder multiplies force: F₂/F₁ = A₂/A₁. This enables heavy lifting with minimal input force, used in brakes, excavators, and presses."),
    ("What is Reynolds number?", "Reynolds number (Re = ρvL/μ) predicts flow regime: laminar (smooth, Re < 2300 in pipes) or turbulent (chaotic, Re > 4000). It's the ratio of inertial to viscous forces. Low Re means viscosity dominates (honey flowing); high Re means inertia dominates (water from a fire hose)."),
    ("Explain beam bending and deflection.", "When a beam is loaded, it develops internal stresses: tension on one side, compression on the other. The neutral axis has zero stress. Bending moment M relates to stress: σ = My/I. Deflection depends on load, span, material stiffness (E), and cross-section (I). Longer spans and higher loads mean more deflection."),

    # Electrical Engineering
    ("What is Ohm's Law?", "Ohm's Law: V = IR, where V is voltage (volts), I is current (amps), and R is resistance (ohms). Voltage drives current through resistance like pressure drives water through a pipe. Rearranged: I = V/R (more voltage = more current), R = V/I. This is fundamental to all circuit analysis."),
    ("Explain AC vs DC electricity.", "DC (direct current) flows in one direction at constant voltage - used in batteries and electronics. AC (alternating current) oscillates sinusoidally, typically at 50-60 Hz - used for power transmission because transformers can easily change voltage levels, reducing transmission losses (P = I²R, so higher voltage means lower current and less loss)."),
    ("What is impedance?", "Impedance (Z) is AC resistance, combining resistive and reactive components: Z = √(R² + X²). Reactance X comes from capacitors (Xc = 1/ωC, blocks low frequencies) and inductors (XL = ωL, blocks high frequencies). Unlike resistance, reactance is frequency-dependent. Z is measured in ohms."),
    ("How do transformers work?", "Transformers change AC voltage using electromagnetic induction. Two coils share a magnetic core. AC in the primary creates a changing magnetic field that induces voltage in the secondary. Voltage ratio equals turns ratio: V₂/V₁ = N₂/N₁. Power is conserved (ideally): more voltage means less current."),
    ("What is a transistor?", "A transistor is a semiconductor switch/amplifier with three terminals. In BJTs: base, collector, emitter - small base current controls large collector-emitter current. In MOSFETs: gate, drain, source - gate voltage controls drain-source current. Transistors are the building blocks of all modern electronics and computers."),
    ("Explain capacitors and inductors.", "Capacitors store energy in electric fields between plates: C = εA/d, energy = ½CV². They block DC, pass AC. Inductors store energy in magnetic fields of coiled wire: L = μN²A/l, energy = ½LI². They pass DC, block AC. Together with resistors, they form filters and oscillators."),
    ("What is three-phase power?", "Three-phase power uses three AC voltages 120° apart. Advantages: constant power delivery (single-phase pulses), smaller conductors for same power, self-starting motors. Line voltage = √3 × phase voltage. Industrial and commercial buildings use three-phase; homes typically get single-phase from one leg."),
    ("How do electric motors work?", "Electric motors convert electrical energy to mechanical rotation using magnetic fields. In DC motors, current through armature coils creates magnetic poles that interact with stator magnets, producing torque. AC induction motors use rotating magnetic fields to induce current in the rotor, causing rotation. Torque depends on current and magnetic field strength."),
    ("What is power factor?", "Power factor = real power / apparent power = cos(φ), where φ is the phase angle between voltage and current. Resistive loads have PF = 1. Inductive loads (motors) have lagging PF < 1, meaning current lags voltage. Low PF wastes capacity and may incur utility penalties. Capacitors can correct power factor."),
    ("Explain feedback control systems.", "Feedback control compares output to desired setpoint and adjusts input to minimize error. Negative feedback stabilizes systems. PID control combines: Proportional (responds to current error), Integral (eliminates steady-state error), Derivative (anticipates future error). Proper tuning prevents oscillation and ensures fast, stable response."),

    # Civil Engineering
    ("What is the difference between dead load and live load?", "Dead load is the permanent, static weight of the structure itself - beams, columns, floors, roofing. Live load is variable - people, furniture, vehicles, snow. Structures must support both. Dead loads are predictable; live loads require safety factors. Building codes specify minimum live loads by occupancy type."),
    ("How does reinforced concrete work?", "Concrete is strong in compression but weak in tension. Steel reinforcement (rebar) handles tensile stresses. When a beam bends, the bottom is in tension - that's where rebar goes. The concrete protects steel from corrosion. Pre-stressing compresses concrete before loading, allowing longer spans and thinner sections."),
    ("What is soil bearing capacity?", "Bearing capacity is the maximum pressure soil can support without failure. It depends on soil type, moisture, depth, and footing size. Clay has lower capacity than gravel. Overloading causes shear failure or excessive settlement. Geotechnical engineers test soil and specify foundation requirements."),
    ("Explain how bridges handle loads.", "Bridges transfer loads to supports through different mechanisms. Beam bridges resist by bending. Arch bridges compress into abutments. Suspension bridges hang the deck from cables in tension anchored at ends. Cable-stayed bridges support the deck with cables directly to towers. Each type suits different spans and conditions."),
    ("What is the water-cement ratio?", "Water-cement ratio (w/c) is water weight divided by cement weight in concrete mix. Lower w/c (around 0.4) means stronger, less permeable concrete but harder to work. Higher w/c (0.6+) is workable but weaker. Too much water creates voids when it evaporates. Admixtures can improve workability without excess water."),
    ("How do retaining walls work?", "Retaining walls resist lateral earth pressure from soil they hold back. Active pressure pushes the wall; passive pressure resists movement. Gravity walls use mass. Cantilever walls use reinforced concrete base slabs. Mechanically stabilized earth uses layers of reinforcement. Drainage behind the wall prevents hydrostatic pressure buildup."),
    ("What causes building settlement?", "Settlement occurs as soil compresses under load. Immediate settlement happens in sandy soils. Consolidation settlement in clay takes years as water squeezes out. Differential settlement (uneven) causes structural damage. Proper foundation design, soil investigation, and sometimes ground improvement prevent excessive settlement."),
    ("Explain structural redundancy.", "Redundant structures have multiple load paths, so failure of one element doesn't cause collapse. If a beam fails, loads redistribute to others. Non-redundant (determinate) structures collapse if any critical member fails. Modern codes require redundancy for important structures. It provides safety margin against unexpected loads or deterioration."),

    # Chemical Engineering
    ("What is mass balance?", "Mass balance: input = output + accumulation. In steady-state processes, accumulation = 0, so what goes in must come out. For reactive systems, include generation and consumption terms. Mass balance is fundamental to process design, calculating flow rates, yields, and detecting leaks or measurement errors."),
    ("Explain distillation.", "Distillation separates liquids by boiling point differences. Liquid mixture is heated; more volatile components vaporize first. Vapor rises through a column with trays or packing, contacting descending liquid. Repeated vaporization/condensation enriches vapor in light components. Reflux (returning some distillate) improves separation."),
    ("What is reaction kinetics?", "Reaction kinetics studies reaction rates. Rate often depends on concentration: rate = k[A]ⁿ[B]ᵐ, where k is the rate constant and n, m are orders. Arrhenius equation: k = Ae^(-Ea/RT) shows temperature dependence. Higher temperature and activation energy (Ea) affect how fast reactions proceed."),
    ("How do catalysts work?", "Catalysts speed reactions by providing alternative pathways with lower activation energy. They participate in the mechanism but regenerate unchanged. Heterogeneous catalysts (solid surface) offer sites for adsorption. Homogeneous catalysts (same phase) often work through intermediate complexes. Catalysts don't change equilibrium, just how fast it's reached."),
    ("What is heat transfer?", "Heat transfers by three modes: Conduction (through materials, Q = kAΔT/L), Convection (via fluid motion, Q = hAΔT), Radiation (electromagnetic waves, Q = εσAT⁴). In practice, all three occur together. Heat exchangers maximize conduction and convection. Insulation minimizes all three."),
    ("Explain process control.", "Process control maintains desired operating conditions. Sensors measure temperature, pressure, flow, level, composition. Controllers compare measurements to setpoints and adjust valves, heaters, pumps. Feedback control reacts to deviations. Feedforward control anticipates disturbances. Proper control ensures product quality, safety, and efficiency."),
    ("What is fluid flow in pipes?", "Fluid flow follows Bernoulli's equation (energy conservation) and continuity (mass conservation). Pressure drops due to friction (Darcy-Weisbach equation) and fittings. Laminar flow is orderly; turbulent flow mixes well but has higher friction. Pump sizing accounts for static head, friction losses, and desired flow rate."),

    # Aerospace Engineering
    ("How do airplanes generate lift?", "Lift comes from pressure difference across wings. Air flowing over the curved upper surface speeds up and pressure drops (Bernoulli). Air below has higher pressure. The pressure difference creates upward force. Angle of attack also deflects air downward (Newton's third law). Lift = ½ρv²SCL, where CL depends on wing shape and angle."),
    ("What is drag and its types?", "Drag opposes motion through fluid. Parasitic drag includes: form drag (pressure difference front/back), skin friction (surface shear), interference drag (component interactions). Induced drag is the penalty for generating lift, caused by wingtip vortices. Total drag = ½ρv²SCD. Minimizing drag improves fuel efficiency."),
    ("Explain rocket propulsion.", "Rockets work by Newton's third law: expelling mass backward produces forward thrust. Thrust = mass flow rate × exhaust velocity. Specific impulse (Isp) measures propellant efficiency - seconds of thrust per unit weight of propellant. Chemical rockets burn fuel and oxidizer. Ion engines have high Isp but low thrust."),
    ("What is the Mach number?", "Mach number = flow velocity / speed of sound. Subsonic: M < 1. Transonic: M ≈ 1 (shock wave formation). Supersonic: 1 < M < 5. Hypersonic: M > 5. As aircraft approach Mach 1, drag increases dramatically. Swept wings delay this effect. Supersonic flight creates sonic booms from shock waves."),
    ("How does orbital mechanics work?", "Orbits balance gravity and inertia. Orbital velocity: v = √(GM/r). Higher orbits are slower but require more energy to reach. To raise orbit, accelerate forward (adds energy). To lower orbit, decelerate. Hohmann transfers use two burns to change orbits efficiently. Escape velocity: v = √(2GM/r)."),

    # Materials Science
    ("What are the crystal structures of metals?", "Common metal crystal structures: BCC (body-centered cubic) - iron, chromium, tungsten. FCC (face-centered cubic) - aluminum, copper, gold, nickel. HCP (hexagonal close-packed) - titanium, zinc, magnesium. Structure affects properties: FCC metals are generally more ductile; BCC can be brittle at low temperatures."),
    ("Explain heat treatment of steel.", "Heat treatment modifies steel properties through controlled heating and cooling. Annealing (slow cool) softens. Quenching (rapid cool) hardens by forming martensite. Tempering (reheating quenched steel) balances hardness and toughness. Normalizing (air cool) refines grain structure. Each process changes the microstructure and thus properties."),
    ("What is corrosion and how to prevent it?", "Corrosion is electrochemical degradation of metals. Iron rusts as it oxidizes, losing electrons to oxygen. Prevention methods: protective coatings (paint, galvanizing), cathodic protection (sacrificial anodes), alloying (stainless steel's chromium forms protective oxide), controlling environment (dehumidification, inhibitors)."),
    ("What are composite materials?", "Composites combine materials for superior properties. Fiber-reinforced polymers (carbon fiber, fiberglass) have high strength-to-weight ratios. The matrix (polymer, metal, ceramic) transfers load to strong fibers. Fiber orientation determines directional properties. Aerospace, automotive, and sporting goods use composites extensively."),
    ("Explain polymers and plastics.", "Polymers are long-chain molecules made of repeating monomers. Thermoplastics (polyethylene, PVC) soften when heated and can be remolded. Thermosets (epoxy, vulcanized rubber) cure permanently and cannot be remelted. Properties depend on molecular weight, branching, crystallinity, and additives. Plastics are synthetic polymers."),
]

# =============================================================================
# SCIENCE
# =============================================================================
SCIENCE = [
    # Physics - Mechanics
    ("What are Newton's three laws of motion?", "First law (inertia): Objects at rest stay at rest, objects in motion stay in motion unless acted on by external force. Second law: F = ma, force equals mass times acceleration. Third law: Every action has an equal and opposite reaction. These laws describe all classical mechanical behavior."),
    ("Explain conservation of energy.", "Energy cannot be created or destroyed, only transformed. Kinetic energy (½mv²) can become potential energy (mgh) and vice versa. In real systems, some energy becomes heat due to friction, but total energy is conserved. This principle applies universally and is fundamental to physics."),
    ("What is momentum and its conservation?", "Momentum p = mv (mass × velocity) is conserved in isolated systems. In collisions, total momentum before equals total momentum after. Elastic collisions also conserve kinetic energy; inelastic collisions don't (some becomes heat/deformation). This explains why rockets work and why heavy trucks are harder to stop."),
    ("How does friction work?", "Friction opposes relative motion between surfaces. Static friction prevents sliding (fs ≤ μsN). Kinetic friction acts during sliding (fk = μkN). Friction comes from surface irregularities and molecular adhesion. μ (coefficient of friction) depends on materials. Friction enables walking and driving but wastes energy as heat."),
    ("What is angular momentum?", "Angular momentum L = Iω (moment of inertia × angular velocity) is conserved when no external torque acts. This is why spinning figure skaters speed up when pulling arms in (I decreases, ω increases). Gyroscopes resist tilting due to angular momentum conservation."),

    # Physics - Electromagnetism
    ("What is an electric field?", "An electric field E exists around any charge, exerting force on other charges: F = qE. Field strength E = kQ/r² for a point charge. Field lines point from positive to negative charges. Electric fields do work on charges, creating voltage differences. Conductors have zero internal field in equilibrium."),
    ("How do magnetic fields work?", "Magnetic fields (B) are created by moving charges (currents) and affect moving charges. Force on current: F = IL × B. Force on moving charge: F = qv × B (perpendicular to both v and B). Magnetic field from wire: B = μ₀I/(2πr). Earth's magnetic field comes from convecting molten iron in the outer core."),
    ("What is electromagnetic induction?", "Changing magnetic flux through a loop induces voltage: V = -dΦ/dt (Faraday's law). This is how generators work - rotating coils in magnetic fields produce AC. Also how transformers work - changing current in primary creates changing flux that induces voltage in secondary. Lenz's law: induced current opposes the change."),
    ("Explain electromagnetic waves.", "Electromagnetic waves are oscillating electric and magnetic fields propagating through space at c = 3×10⁸ m/s. They don't need a medium. The spectrum includes (by increasing frequency): radio, microwave, infrared, visible light, ultraviolet, X-rays, gamma rays. Energy E = hf, so higher frequency means higher energy."),
    ("What is Coulomb's law?", "Coulomb's law: F = kq₁q₂/r², where k = 9×10⁹ N·m²/C². Electric force is proportional to both charges, inversely proportional to distance squared. Like charges repel; opposite charges attract. This is analogous to gravitational force but much stronger and can be either attractive or repulsive."),

    # Physics - Thermodynamics
    ("What are the laws of thermodynamics?", "Zeroth: If A and B are in thermal equilibrium with C, they're in equilibrium with each other (defines temperature). First: Energy is conserved; ΔU = Q - W. Second: Entropy of isolated systems never decreases; heat flows hot to cold spontaneously. Third: Entropy approaches zero as temperature approaches absolute zero."),
    ("What is entropy?", "Entropy measures disorder or the number of microscopic arrangements consistent with a macroscopic state. Statistically: S = k·ln(Ω). In processes: ΔS ≥ Q/T. Entropy always increases in isolated systems, explaining why heat flows hot to cold, why ice melts in warm rooms, why you can't unscramble eggs."),
    ("Explain heat capacity and specific heat.", "Heat capacity C is energy needed to raise temperature by 1 degree: Q = CΔT. Specific heat c is per unit mass: Q = mcΔT. Water has high specific heat (4.18 J/g·K), meaning it stores lots of heat and resists temperature changes. This moderates coastal climates and makes water useful for cooling."),
    ("What is a Carnot engine?", "A Carnot engine is a theoretical ideal heat engine operating between hot (Th) and cold (Tc) reservoirs. Its efficiency η = 1 - Tc/Th is the maximum possible for any heat engine between those temperatures. Real engines have lower efficiency due to friction and irreversibilities. This sets fundamental limits on power generation."),
    ("How do phase changes work?", "Phase changes occur at specific temperatures where energy goes into breaking bonds rather than raising temperature. Latent heat of fusion (melting) and vaporization (boiling) are the energies required. Water: 334 J/g to melt, 2260 J/g to vaporize. Pressure affects phase change temperatures (pressure cookers, freeze-drying)."),

    # Physics - Waves and Optics
    ("What is wave interference?", "When waves overlap, they interfere. Constructive interference: waves in phase combine to larger amplitude. Destructive interference: waves out of phase cancel. This explains the double-slit experiment pattern, noise-canceling headphones, thin film iridescence, and radio signal fading."),
    ("Explain refraction.", "Refraction is bending of waves when they change speed crossing a boundary. Snell's law: n₁sin(θ₁) = n₂sin(θ₂), where n is refractive index (n = c/v). Light slows in denser media and bends toward the normal. This is how lenses focus light and why pools look shallower than they are."),
    ("What is diffraction?", "Diffraction is waves bending around obstacles or through openings. It's significant when the opening/obstacle is comparable to wavelength. Diffraction limits optical resolution (Rayleigh criterion). It explains why you can hear around corners (long wavelength sound) but can't see around them (short wavelength light)."),
    ("How do lasers work?", "Lasers produce coherent light through stimulated emission. Atoms in excited states, when hit by photons of the right energy, emit identical photons. Mirrors create a resonant cavity for amplification. Laser light is monochromatic (one wavelength), coherent (waves in phase), and directional. Applications: cutting, communication, measurement."),
    ("What is polarization?", "Polarization is the orientation of electromagnetic wave oscillations. Unpolarized light oscillates in all directions perpendicular to propagation. Linear polarizers pass only one orientation. Reflected light is partially polarized (why polarized sunglasses reduce glare). Circular polarization rotates as the wave propagates."),

    # Physics - Modern Physics
    ("What is the photoelectric effect?", "When light hits metal, electrons are ejected. Key observations: requires minimum frequency (not intensity), electron energy depends on frequency (not intensity), ejection is instantaneous. Einstein explained this using photons with energy E = hf. This proved light has particle properties, earning him the Nobel Prize."),
    ("Explain wave-particle duality.", "Quantum objects exhibit both wave and particle properties. Electrons create interference patterns (wave behavior) but hit detectors at discrete points (particle behavior). De Broglie wavelength: λ = h/p. Which behavior you observe depends on the experiment. Neither classical waves nor classical particles fully describe quantum reality."),
    ("What is Heisenberg's uncertainty principle?", "You cannot simultaneously know both position and momentum with arbitrary precision: ΔxΔp ≥ ℏ/2. This is fundamental, not a measurement limitation. Similarly for energy and time: ΔEΔt ≥ ℏ/2. The more precisely you know one, the less precisely you can know the other. This underlies quantum behavior."),
    ("Explain special relativity basics.", "Einstein's special relativity: 1) Laws of physics are the same in all inertial frames. 2) Speed of light c is constant for all observers. Consequences: time dilation (moving clocks run slow), length contraction (moving objects shorten), mass-energy equivalence (E = mc²), simultaneity is relative."),
    ("What is radioactive decay?", "Unstable nuclei emit radiation to become more stable. Alpha decay: emits helium nucleus. Beta decay: neutron becomes proton (or vice versa), emitting electron/positron and neutrino. Gamma decay: emits high-energy photon. Half-life is time for half the nuclei to decay. Decay is random but statistically predictable."),

    # Chemistry
    ("What is the periodic table organization?", "Elements are arranged by increasing atomic number (protons). Rows (periods) represent electron shells. Columns (groups) have similar electron configurations and properties. Group 1: alkali metals (reactive). Group 17: halogens. Group 18: noble gases (inert). Metals are left, nonmetals right, metalloids between."),
    ("Explain chemical bonding types.", "Ionic bonds: electrons transfer between atoms (NaCl). Covalent bonds: electrons shared between atoms (H₂O). Metallic bonds: electrons delocalized among metal atoms. Bond type affects properties: ionic compounds are brittle, high melting, conduct when dissolved. Covalent compounds have lower melting points."),
    ("What is electronegativity?", "Electronegativity measures an atom's ability to attract bonding electrons. Fluorine is most electronegative (4.0). Electronegativity increases across periods (more protons) and decreases down groups (electrons farther from nucleus). Large electronegativity differences create polar or ionic bonds."),
    ("Explain acids and bases.", "Arrhenius: acids produce H⁺, bases produce OH⁻. Brønsted-Lowry: acids donate protons, bases accept protons. Lewis: acids accept electron pairs, bases donate electron pairs. pH = -log[H⁺]. pH 7 is neutral. Strong acids/bases fully dissociate; weak ones partially dissociate and have equilibrium constants (Ka, Kb)."),
    ("What is oxidation-reduction?", "Redox reactions involve electron transfer. Oxidation: losing electrons (OIL: Oxidation Is Loss). Reduction: gaining electrons (RIG: Reduction Is Gain). Oxidation states track electron distribution. In 2Na + Cl₂ → 2NaCl, sodium is oxidized (0 → +1), chlorine is reduced (0 → -1). Batteries and corrosion are redox processes."),
    ("Explain chemical equilibrium.", "Reversible reactions reach equilibrium where forward and reverse rates equal. Equilibrium constant K = [products]/[reactants] (raised to stoichiometric powers). Large K favors products. Le Chatelier's principle: systems shift to oppose changes. Adding reactants, removing products, or changing pressure/temperature shifts equilibrium."),
    ("What are organic chemistry basics?", "Organic chemistry studies carbon compounds. Carbon forms four bonds, creating chains, rings, and 3D structures. Functional groups determine reactivity: -OH (alcohol), -COOH (carboxylic acid), -NH₂ (amine), C=O (carbonyl). Naming follows IUPAC rules. Organic reactions include substitution, addition, elimination, and rearrangement."),
    ("Explain reaction thermodynamics.", "Gibbs free energy ΔG = ΔH - TΔS determines spontaneity. ΔG < 0: spontaneous. ΔH is enthalpy change (heat). ΔS is entropy change. Exothermic reactions (ΔH < 0) release heat. At equilibrium: ΔG = 0 and K = e^(-ΔG°/RT). Catalysts speed reactions but don't change ΔG."),
    ("What is stoichiometry?", "Stoichiometry uses balanced equations to calculate reactant/product quantities. Coefficients give mole ratios. Steps: 1) Write balanced equation. 2) Convert given quantity to moles. 3) Use mole ratio to find moles of desired substance. 4) Convert to requested units. Limiting reagent determines maximum product."),

    # Biology
    ("How does DNA replication work?", "DNA replication is semiconservative: each strand serves as template. Helicase unwinds the double helix. Primase adds RNA primers. DNA polymerase III synthesizes new strands 5' to 3'. Leading strand is continuous; lagging strand forms Okazaki fragments. Ligase joins fragments. Proofreading ensures accuracy."),
    ("Explain protein synthesis.", "Transcription: RNA polymerase reads DNA template and synthesizes mRNA in nucleus. mRNA is processed (splicing, capping, poly-A tail) and exported. Translation: ribosomes read mRNA codons. tRNAs bring amino acids matching codons. Amino acids join into polypeptide chains that fold into functional proteins."),
    ("What is cellular respiration?", "Cellular respiration converts glucose to ATP. Glycolysis (cytoplasm): glucose → 2 pyruvate, 2 ATP, 2 NADH. Krebs cycle (mitochondria): pyruvate oxidized, producing NADH, FADH₂, 2 ATP. Electron transport chain: NADH/FADH₂ power proton gradient, producing ~34 ATP. Total: ~38 ATP per glucose."),
    ("How does photosynthesis work?", "Photosynthesis converts CO₂ and H₂O to glucose using light energy. Light reactions (thylakoids): chlorophyll absorbs light, splits water, produces ATP and NADPH, releases O₂. Calvin cycle (stroma): uses ATP and NADPH to fix CO₂ into glucose. Overall: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂."),
    ("Explain natural selection.", "Natural selection is the mechanism of evolution. Steps: 1) Variation exists in populations. 2) Some traits are heritable. 3) More offspring are born than survive. 4) Individuals with advantageous traits survive and reproduce more. Over generations, beneficial traits become more common. This is not random; it's directed by environment."),
    ("What is mitosis vs meiosis?", "Mitosis: one division producing two identical diploid cells. For growth and repair. Meiosis: two divisions producing four haploid gametes. Crossing over and independent assortment create genetic diversity. Meiosis halves chromosome number; fertilization restores it. Errors cause aneuploidy (Down syndrome, etc.)."),
    ("How does the immune system work?", "Innate immunity: immediate, non-specific. Includes barriers (skin, mucus), phagocytes, inflammation, complement. Adaptive immunity: slower but specific and has memory. B cells produce antibodies. T cells: helper T cells activate others, cytotoxic T cells kill infected cells. Memory cells enable faster secondary response."),
    ("Explain gene expression regulation.", "Gene expression is controlled at multiple levels. Transcription factors bind promoters/enhancers to activate or repress genes. Epigenetics (methylation, histone modification) affects accessibility. Post-transcriptional: mRNA splicing, stability, miRNA silencing. Post-translational: protein modification, localization, degradation. Different cells express different genes."),
    ("What is ecology and ecosystems?", "Ecology studies organism-environment interactions. Ecosystems include biotic (living) and abiotic (non-living) components. Energy flows: sun → producers → consumers → decomposers. Nutrients cycle (carbon, nitrogen, water). Food webs show feeding relationships. Population dynamics involve birth, death, immigration, emigration rates."),

    # Earth Science
    ("How do plate tectonics work?", "Earth's lithosphere is divided into plates floating on the asthenosphere. Convection currents in the mantle drive plate motion. Divergent boundaries: plates separate, new crust forms. Convergent boundaries: plates collide, causing subduction, mountains, or trenches. Transform boundaries: plates slide past each other. This explains earthquakes, volcanoes, and continental drift."),
    ("What causes earthquakes?", "Earthquakes result from sudden release of accumulated stress along faults. Tectonic plates move slowly, building stress at boundaries. When friction is overcome, rocks slip suddenly, releasing seismic waves. P-waves are fastest (compression). S-waves are slower (shear). Surface waves cause most damage. Magnitude measures energy released."),
    ("Explain the water cycle.", "The water cycle: evaporation (surface water to vapor) → transpiration (plants release vapor) → condensation (vapor to clouds) → precipitation (rain/snow) → collection (streams, lakes, groundwater) → back to evaporation. Solar energy drives evaporation. Gravity drives precipitation and runoff. Cycle distributes freshwater globally."),
    ("How does weather vs climate differ?", "Weather is short-term atmospheric conditions: temperature, humidity, precipitation, wind at a specific place and time. Climate is long-term average weather patterns for a region over decades. Climate change refers to shifts in these long-term patterns. Weather is what you experience today; climate is what you expect over years."),
    ("What is the rock cycle?", "The rock cycle: Igneous rock forms from cooled magma/lava. Weathering and erosion break rocks into sediments. Sediments compact and cement into sedimentary rock. Heat and pressure transform rocks into metamorphic rock. Melting returns material to magma. Any rock type can become any other through these processes."),
]

# =============================================================================
# MATHEMATICS
# =============================================================================
MATHEMATICS = [
    # Algebra
    ("How do you solve quadratic equations?", "For ax² + bx + c = 0, use the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a. The discriminant b² - 4ac determines solutions: positive = 2 real roots, zero = 1 repeated root, negative = 2 complex roots. Alternatively, factor if possible, or complete the square."),
    ("What is a function?", "A function f assigns each input exactly one output. Written f(x) = expression. Domain is set of valid inputs; range is set of possible outputs. Functions can be linear (f(x) = mx + b), polynomial, exponential, logarithmic, trigonometric, etc. Composition: (f∘g)(x) = f(g(x))."),
    ("Explain logarithms.", "Logarithm answers: 'What exponent gives this result?' log_b(x) = y means b^y = x. Properties: log(ab) = log(a) + log(b), log(a/b) = log(a) - log(b), log(a^n) = n·log(a). Common bases: 10 (log), e (ln), 2. Logarithms invert exponentials."),
    ("What are complex numbers?", "Complex numbers have form a + bi, where i = √(-1). Real part is a, imaginary part is b. Add/subtract component-wise. Multiply using FOIL and i² = -1. Conjugate of a + bi is a - bi. Magnitude: |a + bi| = √(a² + b²). They solve all polynomial equations."),
    ("Explain matrices and their operations.", "Matrices are rectangular arrays of numbers. Addition: element-wise (same dimensions required). Multiplication: row-by-column (A×B needs A's columns = B's rows). Identity matrix I: AI = IA = A. Inverse A⁻¹: AA⁻¹ = I. Determinant: scalar representing scaling factor, zero means non-invertible."),
    ("What is a polynomial and its properties?", "A polynomial is a sum of terms with non-negative integer exponents: aₙxⁿ + ... + a₁x + a₀. Degree is highest exponent. Fundamental theorem of algebra: degree n polynomial has exactly n complex roots (counting multiplicity). Factor theorem: (x - r) is a factor if r is a root."),

    # Calculus
    ("What is a derivative?", "The derivative f'(x) = lim[h→0] (f(x+h) - f(x))/h measures instantaneous rate of change or slope of tangent line. Power rule: d/dx(xⁿ) = nxⁿ⁻¹. Product rule: (fg)' = f'g + fg'. Chain rule: d/dx(f(g(x))) = f'(g(x))·g'(x). Derivatives find rates, optimize, and analyze functions."),
    ("Explain integration.", "Integration is the reverse of differentiation. Indefinite integral ∫f(x)dx finds antiderivatives. Definite integral ∫[a,b]f(x)dx calculates area under curve (or net signed area). Fundamental theorem: ∫[a,b]f(x)dx = F(b) - F(a) where F' = f. Techniques: substitution, parts, partial fractions."),
    ("What is the chain rule?", "The chain rule differentiates composite functions: if y = f(g(x)), then dy/dx = f'(g(x)) · g'(x). In Leibniz notation: dy/dx = (dy/du)(du/dx). Example: d/dx(sin(x²)) = cos(x²) · 2x. Essential for most derivatives involving nested functions."),
    ("Explain limits.", "A limit describes function behavior as input approaches a value: lim[x→a] f(x) = L means f(x) gets arbitrarily close to L as x approaches a. Limits may exist even if f(a) is undefined. Key limits: lim[x→0] sin(x)/x = 1. L'Hôpital's rule handles 0/0 or ∞/∞ forms."),
    ("What is Taylor series?", "Taylor series represents functions as infinite polynomials: f(x) = Σ f⁽ⁿ⁾(a)/n! · (x-a)ⁿ. Maclaurin series centers at a = 0. Examples: eˣ = 1 + x + x²/2! + ..., sin(x) = x - x³/3! + x⁵/5! - ... Useful for approximations and analysis."),
    ("What are differential equations?", "Differential equations relate functions to their derivatives. Ordinary DEs involve one independent variable. First order: dy/dx = f(x,y). Separation of variables works when dy/dx = g(x)h(y). Linear first order: dy/dx + P(x)y = Q(x), solved with integrating factor. Higher order and systems require more techniques."),
    ("Explain multivariable calculus basics.", "Functions of multiple variables f(x,y). Partial derivatives: ∂f/∂x treats y as constant. Gradient: ∇f = (∂f/∂x, ∂f/∂y) points in direction of steepest increase. Double integrals compute volume. Line integrals integrate along curves. Key theorems: Green's, Stokes', divergence theorem."),

    # Linear Algebra
    ("What are vectors and vector spaces?", "Vectors have magnitude and direction. In Rⁿ, vectors are n-tuples. Vector space axioms: closure under addition/scalar multiplication, associativity, commutativity, identity elements, inverses. Span is all linear combinations of vectors. Basis is linearly independent spanning set. Dimension is basis size."),
    ("Explain eigenvalues and eigenvectors.", "For matrix A, eigenvector v satisfies Av = λv, where λ is eigenvalue. A scales v by λ without changing direction. Find eigenvalues: det(A - λI) = 0. Find eigenvectors: solve (A - λI)v = 0. Eigendecomposition: A = PDP⁻¹. Used in stability analysis, PCA, differential equations."),
    ("What is linear independence?", "Vectors v₁, v₂, ..., vₙ are linearly independent if c₁v₁ + c₂v₂ + ... + cₙvₙ = 0 implies all cᵢ = 0. Otherwise, they're linearly dependent. Independent vectors don't 'waste' dimensions. In Rⁿ, at most n vectors can be independent. The rank of a matrix is the maximum number of independent columns."),
    ("Explain matrix transformations.", "Matrices represent linear transformations. Multiplication Ax applies transformation to vector x. Rotation, scaling, shearing, projection are all matrix operations. Composition of transformations = matrix multiplication. Determinant measures area/volume scaling. Orthogonal matrices preserve lengths and angles."),
    ("What is the dot product and cross product?", "Dot product a·b = Σaᵢbᵢ = |a||b|cos(θ), a scalar. Measures projection, zero when perpendicular. Cross product a×b (3D only) = vector perpendicular to both, magnitude |a||b|sin(θ). Gives area of parallelogram. Right-hand rule determines direction."),

    # Probability and Statistics
    ("What is probability?", "Probability P(A) measures likelihood of event A, between 0 (impossible) and 1 (certain). For equally likely outcomes: P(A) = favorable/total. Addition rule: P(A∪B) = P(A) + P(B) - P(A∩B). Multiplication rule: P(A∩B) = P(A)·P(B|A). Conditional probability: P(A|B) = P(A∩B)/P(B)."),
    ("Explain Bayes' theorem.", "Bayes' theorem: P(A|B) = P(B|A)·P(A)/P(B). Updates prior probability P(A) given evidence B to get posterior P(A|B). Example: if a test is 99% accurate and disease prevalence is 1%, a positive result doesn't mean 99% chance of disease. Prior probability matters significantly."),
    ("What are probability distributions?", "Distributions describe probabilities of random variable outcomes. Discrete: Bernoulli (yes/no), binomial (n trials), Poisson (rare events). Continuous: uniform, normal (bell curve), exponential. Parameters (mean, variance) characterize distributions. PDF/PMF gives probabilities; CDF gives cumulative probability."),
    ("Explain the normal distribution.", "The normal (Gaussian) distribution is bell-shaped: f(x) = (1/σ√2π)e^(-(x-μ)²/2σ²). Mean μ is center, standard deviation σ is spread. 68-95-99.7 rule: 68% within 1σ, 95% within 2σ, 99.7% within 3σ of mean. Central limit theorem: sample means approach normal distribution."),
    ("What is hypothesis testing?", "Hypothesis testing evaluates claims using data. Null hypothesis H₀ is default assumption. Alternative H₁ is what we're testing for. Collect data, calculate test statistic, find p-value (probability of results if H₀ true). If p-value < significance level α (often 0.05), reject H₀. Type I error: false positive. Type II: false negative."),
    ("Explain correlation and regression.", "Correlation r measures linear association strength (-1 to 1). Near ±1 is strong; near 0 is weak. Correlation ≠ causation. Linear regression fits y = mx + b to minimize squared errors. R² is proportion of variance explained. Multiple regression uses several predictors: y = b₀ + b₁x₁ + b₂x₂ + ..."),

    # Discrete Math
    ("What is mathematical induction?", "Induction proves statements for all natural numbers. Base case: prove P(1). Inductive step: assume P(k), prove P(k+1). Since P(1) is true and P(k)→P(k+1), P(n) is true for all n. Strong induction assumes P(1), P(2), ..., P(k) to prove P(k+1)."),
    ("Explain combinatorics basics.", "Counting techniques: Permutations (order matters): P(n,r) = n!/(n-r)!. Combinations (order doesn't matter): C(n,r) = n!/(r!(n-r)!). With repetition: permutations = nʳ, combinations = C(n+r-1,r). Useful for probability and algorithm analysis."),
    ("What is graph theory?", "Graphs have vertices (nodes) and edges (connections). Directed graphs have one-way edges. Weighted graphs have edge values. Key concepts: paths, cycles, connectivity, trees (connected, acyclic). Algorithms: BFS, DFS, shortest path (Dijkstra), minimum spanning tree. Models networks, relationships, maps."),
    ("Explain modular arithmetic.", "In modular arithmetic, numbers 'wrap around' after reaching modulus n. a ≡ b (mod n) means a and b have the same remainder when divided by n. Operations: (a + b) mod n, (a × b) mod n work component-wise. Used in cryptography, checksums, cyclic systems."),
    ("What is set theory?", "Sets are collections of distinct elements. Operations: union (A∪B), intersection (A∩B), difference (A-B), complement (A'). Subset: A⊆B. Cardinality: |A| is number of elements. Power set: all subsets. Sets form the foundation of mathematics and logic."),
    ("Explain logic and proofs.", "Propositional logic: statements are true or false. Connectives: AND (∧), OR (∨), NOT (¬), implies (→), iff (↔). Proof methods: direct (assume premises, derive conclusion), contrapositive (prove ¬Q → ¬P instead of P → Q), contradiction (assume ¬P, derive contradiction), constructive (exhibit example)."),

    # Number Theory
    ("What is prime factorization?", "Every integer > 1 can be uniquely expressed as a product of primes (Fundamental Theorem of Arithmetic). Example: 60 = 2² × 3 × 5. Finding factors: trial division, Fermat's method, Pollard's rho. Large number factorization is hard, which underlies RSA cryptography."),
    ("Explain the Euclidean algorithm.", "The Euclidean algorithm finds GCD(a,b): repeatedly replace larger number with remainder until reaching 0. GCD(48, 18): 48 = 2×18 + 12, 18 = 1×12 + 6, 12 = 2×6 + 0. GCD = 6. Extended Euclidean algorithm finds x, y where ax + by = GCD(a,b)."),
    ("What is number theory's role in cryptography?", "Cryptography uses number theory extensively. RSA relies on difficulty of factoring large numbers. Diffie-Hellman uses discrete logarithm problem. Elliptic curve cryptography uses group theory on curves. Modular exponentiation, prime generation, and number-theoretic algorithms are essential."),
]

# =============================================================================
# COMPUTER SCIENCE
# =============================================================================
COMPUTER_SCIENCE = [
    # Data Structures
    ("What is an array vs linked list?", "Arrays store elements contiguously in memory. O(1) random access, O(n) insertion/deletion. Fixed size (usually). Linked lists store elements with pointers to next. O(n) access, O(1) insertion/deletion (if you have pointer). Dynamic size. Arrays are cache-friendly; linked lists avoid resizing."),
    ("Explain hash tables.", "Hash tables map keys to values using a hash function. Hash function converts key to array index. Average O(1) for insert, delete, lookup. Collisions handled by chaining (linked lists at each index) or open addressing (probe for empty slot). Good hash functions distribute keys uniformly."),
    ("What is a binary search tree?", "BST: binary tree where left children < parent < right children. Searching, inserting, deleting are O(log n) average, O(n) worst (unbalanced). In-order traversal gives sorted sequence. Balanced variants (AVL, red-black) guarantee O(log n) operations by maintaining balance constraints."),
    ("Explain stacks and queues.", "Stack: LIFO (last in, first out). Push adds to top, pop removes from top. Used for function calls, undo operations, parsing. Queue: FIFO (first in, first out). Enqueue adds to back, dequeue removes from front. Used for scheduling, BFS, buffering. Both O(1) operations."),
    ("What is a heap?", "Heap is a complete binary tree satisfying heap property: parent ≥ children (max-heap) or parent ≤ children (min-heap). Root is max/min element. Insert and extract-max/min are O(log n). Used for priority queues and heapsort. Array representation: children of index i are at 2i+1 and 2i+2."),
    ("Explain graphs and their representations.", "Graphs have vertices and edges. Representations: adjacency matrix (V×V array, 1 if edge exists) uses O(V²) space, O(1) edge lookup. Adjacency list (array of lists) uses O(V+E) space, O(degree) edge lookup. Matrix better for dense graphs; list better for sparse graphs."),
    ("What is a trie?", "Trie (prefix tree) stores strings character by character. Each node represents a prefix; children represent extending characters. O(m) insert/search where m is string length. Efficient for prefix matching, autocomplete, spell checking. Space can be high but compressed tries help."),

    # Algorithms
    ("Explain Big O notation.", "Big O describes algorithm efficiency as input grows. O(1): constant. O(log n): logarithmic. O(n): linear. O(n log n): linearithmic. O(n²): quadratic. O(2ⁿ): exponential. O(n!): factorial. We analyze worst-case typically. Constants and lower terms are dropped: O(3n² + 5n) = O(n²)."),
    ("What is binary search?", "Binary search finds elements in sorted arrays. Compare target to middle element: if equal, found; if smaller, search left half; if larger, search right half. Repeat until found or subarray empty. O(log n) time, O(1) space. Prerequisite: array must be sorted."),
    ("Explain sorting algorithms.", "Comparison sorts: Bubble sort O(n²), simple. Merge sort O(n log n), stable, uses O(n) space. Quick sort O(n log n) average, O(n²) worst, in-place. Heap sort O(n log n), in-place. Non-comparison: Counting sort O(n+k), Radix sort O(d(n+k)), for integers."),
    ("What is dynamic programming?", "DP solves problems by breaking into overlapping subproblems. Store subproblem results to avoid recomputation. Two approaches: top-down (memoization) or bottom-up (tabulation). Classic examples: Fibonacci, knapsack, longest common subsequence, shortest paths. Key: identify subproblem structure and recurrence."),
    ("Explain graph traversal algorithms.", "BFS (breadth-first): uses queue, explores neighbors before their neighbors. Finds shortest path in unweighted graphs. O(V+E). DFS (depth-first): uses stack/recursion, explores as deep as possible before backtracking. O(V+E). Used for connectivity, cycle detection, topological sort."),
    ("What is Dijkstra's algorithm?", "Dijkstra finds shortest paths from source to all vertices in weighted graphs (non-negative weights). Uses priority queue of vertices by distance. Repeatedly extract minimum, update neighbors' distances. O((V+E) log V) with binary heap. Greedy algorithm; proven optimal for non-negative weights."),
    ("Explain divide and conquer.", "Divide and conquer: split problem into subproblems, solve recursively, combine solutions. Examples: merge sort (divide array, sort halves, merge), quick sort (partition around pivot, sort partitions), binary search. Recurrence relations describe time complexity: T(n) = aT(n/b) + f(n)."),
    ("What is greedy algorithm strategy?", "Greedy algorithms make locally optimal choices hoping for global optimum. Works when problems have greedy-choice property and optimal substructure. Examples: Dijkstra's algorithm, Huffman coding, activity selection. Doesn't always work: coin change with arbitrary denominations may need DP."),
    ("Explain recursion and iteration.", "Recursion: function calls itself with smaller input until base case. Elegant but uses call stack space. Iteration: loops repeat until condition met. More memory efficient. Any recursion can be converted to iteration (and vice versa). Tail recursion can be optimized to iteration by compilers."),

    # Operating Systems
    ("What is a process vs thread?", "Process: independent program execution with own memory space. Context switch expensive. Threads: lightweight execution units within a process, sharing memory. Context switch cheaper. Processes are isolated; thread bugs can crash entire process. Use processes for isolation, threads for parallelism."),
    ("Explain virtual memory.", "Virtual memory gives each process the illusion of large, contiguous memory. OS maps virtual addresses to physical addresses via page tables. Pages not in RAM are stored on disk and swapped in on demand (page fault). Benefits: memory isolation, more memory than physically available, simplified memory management."),
    ("What is deadlock?", "Deadlock: processes permanently blocked waiting for each other's resources. Four conditions: mutual exclusion, hold and wait, no preemption, circular wait. Prevention: break any condition. Avoidance: check safety before allocation (Banker's algorithm). Detection and recovery: find cycles, kill processes."),
    ("Explain process scheduling.", "Scheduler decides which process runs. Algorithms: FCFS (first come first served), SJF (shortest job first), Round Robin (time slices), Priority scheduling. Metrics: throughput, turnaround time, waiting time, response time. Preemptive schedulers can interrupt running processes."),
    ("What is a file system?", "File systems organize data on storage devices. Files are named collections of data. Directories organize files hierarchically. Metadata includes name, size, permissions, timestamps. Common: FAT, NTFS, ext4, APFS. Journaling prevents corruption. Inodes (Unix) store file metadata."),
    ("Explain synchronization primitives.", "Mutex: mutual exclusion lock, one thread at a time. Semaphore: counter allowing multiple access up to limit. Condition variable: thread waits until condition signaled. Barriers: threads wait until all reach point. These prevent race conditions but can cause deadlock if misused."),

    # Networking
    ("What is the OSI model?", "OSI model has 7 layers: Physical (bits, cables), Data Link (frames, MAC), Network (packets, IP routing), Transport (segments, TCP/UDP), Session (connections), Presentation (encryption, compression), Application (HTTP, FTP). Each layer adds headers; data flows down sender's stack, up receiver's."),
    ("Explain TCP vs UDP.", "TCP: connection-oriented, reliable, ordered delivery. Three-way handshake establishes connection. Guarantees delivery via acknowledgments and retransmission. Flow/congestion control. Slower but reliable. UDP: connectionless, unreliable, unordered. No overhead. Faster, used for streaming, gaming, DNS where speed matters more than reliability."),
    ("What is HTTP?", "HTTP (Hypertext Transfer Protocol) is application-layer protocol for web. Client sends request (GET, POST, PUT, DELETE) with headers and body. Server responds with status code (200 OK, 404 Not Found, 500 Error), headers, body. HTTPS adds TLS encryption. HTTP/2 adds multiplexing; HTTP/3 uses QUIC over UDP."),
    ("Explain IP addressing.", "IPv4: 32-bit addresses (4 octets, e.g., 192.168.1.1). IPv6: 128-bit addresses (hex, e.g., 2001:db8::1). Subnet masks divide network and host portions. CIDR notation: 10.0.0.0/8 means first 8 bits are network. Private ranges: 10.x.x.x, 172.16-31.x.x, 192.168.x.x. NAT maps private to public addresses."),
    ("What is DNS?", "DNS (Domain Name System) translates domain names to IP addresses. Hierarchical system: root servers → TLD servers (.com, .org) → authoritative servers. Recursive resolvers cache results. Query: browser → local DNS → root → TLD → authoritative. TTL controls caching duration."),
    ("Explain encryption basics.", "Symmetric encryption: same key encrypts and decrypts (AES, DES). Fast but key distribution problem. Asymmetric encryption: public key encrypts, private key decrypts (RSA, ECC). Slower but solves key distribution. TLS uses asymmetric to exchange symmetric keys. Hashing (SHA-256) creates fixed-size fingerprint."),

    # Databases
    ("What is SQL vs NoSQL?", "SQL (relational): structured tables with schemas, ACID transactions, SQL queries. Good for complex queries, consistency. Examples: PostgreSQL, MySQL. NoSQL: flexible schemas, various models (document, key-value, column, graph). Good for scalability, unstructured data. Examples: MongoDB, Redis, Cassandra."),
    ("Explain database normalization.", "Normalization reduces redundancy via decomposition. 1NF: atomic values, no repeating groups. 2NF: no partial dependencies on composite key. 3NF: no transitive dependencies. BCNF: every determinant is candidate key. Higher forms exist. Trade-off: normalization reduces redundancy but may require more joins."),
    ("What is ACID?", "ACID properties ensure reliable database transactions. Atomicity: transaction completes fully or not at all. Consistency: database remains in valid state. Isolation: concurrent transactions don't interfere. Durability: committed changes persist despite failures. Implemented via locking, logging, recovery protocols."),
    ("Explain database indexing.", "Indexes speed up queries by creating sorted data structures (typically B-trees) on columns. Trade-off: faster reads, slower writes (must update index). Primary index on primary key. Secondary indexes on other columns. Covering index includes all query columns. Too many indexes hurt write performance."),
    ("What are database transactions?", "Transaction is a sequence of operations treated as single unit. BEGIN starts, COMMIT saves, ROLLBACK undoes. Isolation levels: Read Uncommitted, Read Committed, Repeatable Read, Serializable (trade consistency for concurrency). Locks prevent conflicts. Distributed transactions use two-phase commit."),
    ("Explain joins in SQL.", "Joins combine rows from tables. INNER JOIN: matching rows only. LEFT JOIN: all left rows, matching right rows (NULL if no match). RIGHT JOIN: opposite. FULL JOIN: all rows from both. CROSS JOIN: cartesian product. Join condition typically matches foreign key to primary key."),

    # Programming Paradigms
    ("What is object-oriented programming?", "OOP organizes code around objects containing data (attributes) and behavior (methods). Four pillars: Encapsulation (hide internal details), Inheritance (classes extend others), Polymorphism (same interface, different implementations), Abstraction (focus on essential features). Classes define object blueprints."),
    ("Explain functional programming.", "Functional programming treats computation as function evaluation. Key concepts: pure functions (no side effects, same input = same output), immutable data, first-class functions, higher-order functions. Benefits: easier reasoning, testing, parallelization. Languages: Haskell, Lisp, Scala, and features in Python, JavaScript."),
    ("What is recursion vs iteration?", "Recursion: function calls itself with base case to terminate. Natural for tree structures, divide-and-conquer. Uses stack space, risk of overflow. Iteration: loops with explicit state. More memory efficient. Mathematically equivalent. Tail recursion can be optimized to iteration."),
    ("Explain design patterns.", "Design patterns are reusable solutions to common problems. Creational: Singleton (one instance), Factory (create objects). Structural: Adapter (interface conversion), Decorator (add behavior). Behavioral: Observer (event subscription), Strategy (swappable algorithms). Patterns provide vocabulary and proven approaches."),
    ("What is concurrency vs parallelism?", "Concurrency: dealing with multiple things at once (structure). Parallelism: doing multiple things at once (execution). Concurrent program may run on single core via interleaving. Parallel program requires multiple cores. Concurrency is about managing complexity; parallelism is about performance."),

    # Complexity Theory
    ("What is P vs NP?", "P: problems solvable in polynomial time. NP: problems verifiable in polynomial time. P ⊆ NP, but does P = NP? Unknown, biggest open problem in CS. NP-complete problems are hardest in NP; if any solved in polynomial time, P = NP. NP-hard are at least as hard as NP-complete."),
    ("Explain time complexity classes.", "O(1): constant. O(log n): binary search. O(n): linear scan. O(n log n): efficient sorts. O(n²): nested loops. O(2ⁿ): subsets. O(n!): permutations. Polynomial (P) is 'tractable'. Exponential/factorial is 'intractable' for large inputs. Complexity determines practical limits."),
    ("What is space complexity?", "Space complexity measures memory usage vs input size. Includes input space and auxiliary space. In-place algorithms use O(1) auxiliary space. Recursive algorithms use O(depth) stack space. Trade-offs exist: memoization trades space for time. Space is often a tighter constraint than time."),
    ("Explain NP-complete problems.", "NP-complete: problems in NP where every NP problem reduces to it polynomially. Examples: SAT (first proven NP-complete), traveling salesman, graph coloring, knapsack. If any NP-complete problem has polynomial solution, all do. We use approximation algorithms, heuristics, or accept exponential time."),

    # Machine Learning Basics
    ("What is supervised vs unsupervised learning?", "Supervised: learn from labeled data (input-output pairs). Classification (discrete output), regression (continuous). Unsupervised: find patterns in unlabeled data. Clustering (group similar items), dimensionality reduction (compress features). Semi-supervised uses some labels. Reinforcement learning learns from rewards."),
    ("Explain neural networks.", "Neural networks are layers of interconnected nodes. Each node computes weighted sum of inputs plus bias, then applies activation function. Input layer receives features, hidden layers learn representations, output layer gives predictions. Training via backpropagation adjusts weights to minimize loss."),
    ("What is overfitting?", "Overfitting: model learns training data too well, including noise, and fails to generalize to new data. Signs: high training accuracy, low test accuracy. Prevention: more training data, regularization (L1/L2), dropout, early stopping, cross-validation, simpler models. Underfitting: model too simple."),
    ("Explain gradient descent.", "Gradient descent minimizes functions by iteratively moving in the direction of steepest descent (negative gradient). Learning rate controls step size. Too large: overshoot. Too small: slow convergence. Variants: stochastic (one sample), mini-batch, momentum, Adam. Finds local minima (global for convex functions)."),
    ("What is cross-validation?", "Cross-validation evaluates model generalization. K-fold: split data into k parts, train on k-1, test on 1, rotate. Average results give reliable estimate. Leave-one-out: k = n. Stratified preserves class proportions. Prevents overfitting to single train/test split. Essential for model selection."),
    ("Explain decision trees and random forests.", "Decision trees split data based on features to make predictions. Nodes are decisions, leaves are outcomes. Easy to interpret but prone to overfitting. Random forests: ensemble of trees trained on random subsets of data and features. Averaging reduces variance. More robust, less interpretable."),

    # Security
    ("What is encryption and hashing?", "Encryption is reversible transformation with a key; decrypt to recover original. Symmetric: same key (AES). Asymmetric: public/private keys (RSA). Hashing is one-way transformation producing fixed-size digest (SHA-256). Hashing verifies integrity, stores passwords. Encryption protects confidentiality."),
    ("Explain SQL injection.", "SQL injection: attacker inserts malicious SQL via user input. Example: input ' OR '1'='1 in login field bypasses authentication. Prevention: parameterized queries (prepared statements), input validation, least privilege database accounts. Never concatenate user input into SQL strings."),
    ("What is authentication vs authorization?", "Authentication verifies identity (who are you?): passwords, biometrics, tokens. Authorization determines access rights (what can you do?): roles, permissions, ACLs. Authentication comes first. Multi-factor authentication (MFA) requires multiple methods. OAuth handles authorization between services."),
    ("Explain public key cryptography.", "Public key (asymmetric) cryptography uses key pairs. Public key encrypts or verifies signatures; private key decrypts or signs. RSA: based on factoring difficulty. ECC: based on elliptic curve discrete log. Digital signatures prove authenticity and integrity. Enables secure key exchange and digital certificates."),
    ("What are common security vulnerabilities?", "OWASP Top 10 includes: Injection (SQL, command), Broken Authentication, Sensitive Data Exposure, XXE, Broken Access Control, Security Misconfiguration, XSS (Cross-Site Scripting), Insecure Deserialization, Using Components with Known Vulnerabilities, Insufficient Logging. Defense in depth addresses multiple layers."),

    # Distributed Systems
    ("What is CAP theorem?", "CAP theorem: distributed systems can guarantee at most two of three properties. Consistency: all nodes see same data. Availability: every request gets response. Partition tolerance: system works despite network failures. Since partitions happen, choose CP (consistent but may be unavailable) or AP (available but eventually consistent)."),
    ("Explain load balancing.", "Load balancers distribute traffic across servers. Algorithms: Round Robin, Least Connections, IP Hash, Weighted. Layer 4 balances by IP/port; Layer 7 by content (HTTP headers, URLs). Benefits: scalability, redundancy, maintenance without downtime. Health checks remove failed servers."),
    ("What is caching?", "Caching stores frequently accessed data closer to requesters. CPU caches, browser caches, CDNs, application caches (Redis, Memcached). Strategies: write-through (write cache and DB), write-behind (write cache, async DB), read-through. Cache invalidation is hard problem. TTL expiration common."),
    ("Explain microservices architecture.", "Microservices: application as suite of small, independently deployable services. Each service handles specific function, has own data store. Communication via APIs (REST, gRPC) or messaging. Benefits: independent scaling, technology diversity, fault isolation. Challenges: distributed complexity, data consistency, debugging."),
    ("What is consensus in distributed systems?", "Consensus: multiple nodes agreeing on value despite failures. Paxos and Raft are common algorithms. Leader election: nodes choose coordinator. Quorum: majority must agree. Byzantine fault tolerance handles malicious nodes. Blockchain uses consensus (proof of work, proof of stake). Essential for replicated state machines."),
]

# =============================================================================
# US LEGAL SYSTEM
# =============================================================================
US_LEGAL_SYSTEM = [
    # Constitutional Framework
    ("What is the US Constitution?", "The US Constitution is the supreme law of the United States, ratified in 1788. It establishes the framework of government with three branches (legislative, executive, judicial), federalism (division of power between federal and state governments), and includes the Bill of Rights (first 10 amendments) protecting individual liberties. All laws must be consistent with the Constitution."),
    ("What are the three branches of US government?", "Legislative (Congress): makes laws, consists of Senate (100 members, 2 per state) and House of Representatives (435 members, proportional to population). Executive (President): enforces laws, commands military, conducts foreign policy. Judicial (Supreme Court and federal courts): interprets laws, determines constitutionality. Separation of powers with checks and balances prevents any branch from becoming too powerful."),
    ("What is the Bill of Rights?", "The Bill of Rights is the first 10 amendments to the Constitution, ratified in 1791. Key rights: 1st - speech, religion, press, assembly, petition. 2nd - bear arms. 4th - protection from unreasonable searches. 5th - due process, self-incrimination protection, double jeopardy. 6th - speedy trial, attorney, jury. 8th - no cruel/unusual punishment. These protect individual liberties from government infringement."),
    ("What is the 14th Amendment?", "The 14th Amendment (1868) is crucial for civil rights. It establishes: citizenship for all persons born in the US, equal protection under law for all persons, due process requirements for states (not just federal government). It's the basis for incorporating Bill of Rights protections against state governments and underlies most civil rights litigation."),
    ("What is federalism?", "Federalism divides power between federal and state governments. Federal government has enumerated powers (defense, currency, interstate commerce, immigration). States have reserved powers (police power, education, family law, intrastate commerce). Some powers are concurrent (taxation, courts). The Supremacy Clause makes federal law supreme when there's conflict with state law."),

    # Court System
    ("How is the US court system structured?", "The US has dual court systems: federal and state. Federal courts: District Courts (trial), Circuit Courts of Appeals (13 circuits), Supreme Court (final authority). State courts: trial courts, appellate courts, state supreme courts. Federal courts handle federal law, constitutional issues, diversity jurisdiction. State courts handle most criminal cases, family law, contracts, property. Cases can move from state to federal system on federal questions."),
    ("What is the Supreme Court?", "The Supreme Court is the highest court in the US, with 9 justices appointed for life by the President with Senate confirmation. It has original jurisdiction in limited cases (disputes between states) and appellate jurisdiction over federal questions from lower courts. It chooses cases via certiorari (needs 4 justices to grant). Its interpretations of the Constitution are final and binding on all other courts."),
    ("What is judicial review?", "Judicial review is the power of courts to declare laws unconstitutional and void. Established in Marbury v. Madison (1803), it makes the judiciary a check on legislative and executive branches. Courts can strike down federal laws, state laws, and executive actions that violate the Constitution. This power is not explicitly stated in the Constitution but is now fundamental to American government."),
    ("What is the difference between civil and criminal law?", "Criminal law: government prosecutes individuals for crimes against society. Burden of proof is 'beyond reasonable doubt.' Penalties include imprisonment, fines, probation. Defendant has right to attorney, jury trial, protection against self-incrimination. Civil law: disputes between private parties. Burden of proof is 'preponderance of evidence' (more likely than not). Remedies include money damages, injunctions, specific performance. No right to free attorney."),
    ("What is common law?", "Common law is judge-made law based on precedent (stare decisis). Courts follow prior decisions in similar cases, creating consistent legal rules. Originated in England, adopted by US (except Louisiana, which uses civil law tradition). Statutes (legislative law) can override common law. Common law fills gaps where no statute exists and interprets ambiguous statutes."),

    # Criminal Justice
    ("What are Miranda rights?", "Miranda rights must be read before custodial interrogation: right to remain silent, anything said can be used against you, right to an attorney, if you can't afford one, one will be appointed. From Miranda v. Arizona (1966). Protects 5th Amendment right against self-incrimination. Statements obtained without Miranda warnings are generally inadmissible. You must clearly invoke these rights."),
    ("What is the difference between a felony and misdemeanor?", "Felonies are serious crimes punishable by more than one year in prison (murder, rape, robbery, burglary, drug trafficking). Misdemeanors are less serious, punishable by up to one year in jail (petty theft, simple assault, DUI first offense, disorderly conduct). Felonies result in loss of rights (voting, gun ownership, jury service). Infractions are minor violations (traffic tickets) with fines only."),
    ("What is due process?", "Due process (5th and 14th Amendments) requires fair procedures before government can deprive someone of life, liberty, or property. Procedural due process: notice, opportunity to be heard, impartial decision-maker. Substantive due process: certain fundamental rights cannot be infringed regardless of procedure (privacy, marriage, family). Applies to criminal proceedings, civil cases, and administrative actions."),
    ("What is the difference between arrest, indictment, and conviction?", "Arrest: police take someone into custody based on probable cause. Indictment: formal charges issued by grand jury (federal felonies require this) or prosecutor files information. Arraignment: defendant enters plea. Trial: prosecution must prove guilt beyond reasonable doubt. Conviction: guilty verdict or plea. Sentencing: judge determines punishment within statutory guidelines. Appeals can follow conviction."),
    ("What is bail?", "Bail is money or property deposited to ensure defendant appears for trial. 8th Amendment prohibits excessive bail. Factors: flight risk, danger to community, severity of charges, ties to community, criminal history. Bail can be denied for serious charges. Bail bondsmen post bail for fee (typically 10%). If defendant appears, bail is returned (minus fees). If defendant flees, bail is forfeited."),
    ("What is plea bargaining?", "Plea bargaining is negotiation between prosecutor and defendant to resolve case without trial. Defendant pleads guilty to lesser charge or for recommended lighter sentence. Benefits: saves court resources, provides certainty, often reduces sentence. About 90-95% of criminal cases are resolved through plea bargains. Judge must approve and ensure plea is knowing and voluntary."),
    ("What is probation vs parole?", "Probation: alternative to incarceration, defendant serves sentence in community under supervision with conditions (check-ins, drug tests, no new crimes). Parole: early release from prison, serving remainder of sentence in community under supervision. Both can be revoked for violations, resulting in incarceration. Conditions vary but typically include regular reporting, employment, travel restrictions."),

    # Civil Rights
    ("What is the Civil Rights Act of 1964?", "The Civil Rights Act of 1964 is landmark legislation prohibiting discrimination based on race, color, religion, sex, or national origin. Title II: public accommodations (hotels, restaurants). Title VI: federally funded programs. Title VII: employment discrimination, created EEOC. It ended Jim Crow segregation and is the foundation of modern civil rights law. Later amended to add protections."),
    ("What is the Equal Protection Clause?", "The Equal Protection Clause (14th Amendment) requires states to treat similarly situated people equally. Courts use different scrutiny levels: strict scrutiny for race/national origin (must be narrowly tailored to compelling interest), intermediate for gender (substantially related to important interest), rational basis for most other classifications. It's the basis for challenging discriminatory laws."),
    ("What are protected classes under employment law?", "Federal law prohibits employment discrimination based on: race, color, national origin, religion, sex (including pregnancy, sexual orientation, gender identity per recent rulings), age (40+), disability, genetic information, citizenship status. State laws may add protections (marital status, political affiliation). Applies to hiring, firing, promotion, pay, harassment, and workplace conditions."),
    ("What is the Americans with Disabilities Act?", "The ADA (1990) prohibits discrimination against people with disabilities. Title I: employment - reasonable accommodations required unless undue hardship. Title II: state/local government services must be accessible. Title III: public accommodations (businesses) must be accessible. Disability: physical or mental impairment substantially limiting major life activities. Covers both physical access and policies."),

    # Contracts and Property
    ("What makes a contract legally binding?", "A valid contract requires: offer (definite terms), acceptance (mirror image of offer), consideration (something of value exchanged), capacity (legal ability to contract - age, mental competence), legality (legal purpose). Written requirement for some contracts (Statute of Frauds): real estate, contracts over one year, goods over $500. Breach remedies: damages, specific performance, rescission."),
    ("What is the difference between civil and criminal court?", "Civil court resolves disputes between private parties seeking remedies like money damages or injunctions. Criminal court prosecutes violations of criminal law, with the government as prosecutor seeking punishment. Different burdens of proof: civil is preponderance of evidence (>50%), criminal is beyond reasonable doubt. Same conduct can result in both civil and criminal liability (e.g., O.J. Simpson cases)."),
    ("What is small claims court?", "Small claims court handles minor civil disputes, typically under $5,000-$10,000 (varies by state). Simplified procedures: no lawyers required (some states prohibit them), relaxed evidence rules, quick resolution. Common cases: landlord-tenant disputes, consumer complaints, minor car accidents, unpaid debts. Judgments are enforceable but collection may require additional steps."),
    ("What are property rights in the US?", "Property rights are fundamental in US law. Real property: land and attached structures. Personal property: movable items. Ownership includes rights to use, exclude others, transfer. Government can take property through eminent domain but must pay just compensation (5th Amendment). Property can be owned individually, jointly, or through entities. Zoning laws regulate land use."),
    ("What is a lien?", "A lien is a legal claim against property as security for debt. Voluntary liens: mortgages, car loans. Involuntary liens: tax liens, mechanic's liens (unpaid contractors), judgment liens. Liens must be satisfied before property can be sold with clear title. Lien priority usually follows recording date. Foreclosure allows lienholder to force sale to satisfy debt."),

    # Family Law
    ("How does divorce work in the US?", "All US states now have no-fault divorce (irreconcilable differences). Process: petition filed, service on spouse, response, discovery, negotiation/mediation, trial if needed, final decree. Issues resolved: property division (equitable distribution or community property depending on state), spousal support (alimony), child custody and support. Waiting periods vary by state (0-12 months)."),
    ("What is child custody?", "Child custody determines parenting arrangements after separation. Legal custody: decision-making authority (education, health, religion). Physical custody: where child lives. Joint custody: shared between parents. Sole custody: one parent has primary rights. Courts decide based on 'best interests of the child': stability, parent-child relationships, child's wishes (if old enough), safety concerns. Custody can be modified if circumstances change."),
    ("What is child support?", "Child support is financial obligation of non-custodial parent to contribute to child's expenses. Amount determined by state guidelines based on income, custody arrangement, children's needs. Continues until child reaches adulthood (18-21 depending on state) or emancipation. Enforced through wage garnishment, tax refund intercept, license suspension, contempt of court. Can be modified if circumstances change substantially."),
    ("What is adoption?", "Adoption legally creates parent-child relationship between non-biological parent and child. Types: agency adoption, private adoption, stepparent adoption, international adoption. Process: home study, background checks, court proceedings, finalization. Birth parent rights must be terminated (voluntarily or involuntarily). Adopted children have same legal rights as biological children. Open adoptions allow some birth parent contact."),

    # Immigration
    ("What are the main categories of US immigration?", "Family-based: US citizens can sponsor immediate relatives (unlimited visas) and other family (limited). Employment-based: skilled workers, investors, extraordinary ability (limited visas, often long waits). Diversity visa lottery: 50,000 annual visas for underrepresented countries. Refugees/asylum: persecution-based protection. Temporary visas: tourists (B), students (F), workers (H-1B, L). Path to citizenship: green card → 3-5 years → naturalization."),
    ("What is the difference between a visa and green card?", "Visa: temporary permission to enter US for specific purpose (tourism, work, study). Has expiration date and conditions. Green card (permanent resident card): permanent permission to live and work in US. Can lead to citizenship after 3-5 years. Green card holders can travel freely, work anywhere, but must maintain US residence and can be deported for crimes."),
    ("What is DACA?", "DACA (Deferred Action for Childhood Arrivals) is a policy protecting undocumented immigrants who arrived as children. Requirements: arrived before age 16, continuous presence since 2007, in school or graduated or military, no serious crimes. Provides work permit and deportation protection (renewable every 2 years) but not path to citizenship. Subject to ongoing legal and political challenges."),
    ("What is asylum?", "Asylum provides protection to people in the US who fear persecution in their home country based on race, religion, nationality, political opinion, or membership in particular social group. Must apply within one year of arrival. Affirmative asylum: apply proactively. Defensive asylum: apply in removal proceedings. If granted, can work, apply for green card after one year. Withholding of removal and CAT are related protections."),

    # Business Law
    ("What are the main business entity types?", "Sole proprietorship: simplest, owner personally liable. Partnership: multiple owners share profits/liability (general or limited). LLC: limited liability, pass-through taxation, flexible management. Corporation: separate legal entity, shareholders, directors, officers. S-Corp: corporate structure with pass-through taxation (restrictions apply). C-Corp: double taxation but unlimited shareholders, preferred for raising capital."),
    ("What is intellectual property?", "Intellectual property protects creations of the mind. Patents: inventions, 20 years protection, must be novel/useful/non-obvious. Copyrights: creative works (books, music, software), life + 70 years, automatic upon creation. Trademarks: brand identifiers (names, logos), can last indefinitely with use. Trade secrets: confidential business information (formulas, processes), protected indefinitely if kept secret."),
    ("What is bankruptcy?", "Bankruptcy provides legal relief for those unable to pay debts. Chapter 7: liquidation - assets sold to pay creditors, remaining debt discharged. Chapter 13: reorganization for individuals - repayment plan over 3-5 years. Chapter 11: reorganization for businesses - continue operating while restructuring debt. Automatic stay stops collections. Some debts non-dischargeable (student loans, taxes, child support). Stays on credit report 7-10 years."),
    ("What is an LLC?", "LLC (Limited Liability Company) combines corporate liability protection with partnership tax flexibility. Members (owners) not personally liable for business debts. Profits pass through to members' personal taxes (avoiding double taxation). Flexible management structure. Operating agreement governs internal affairs. Must file articles of organization with state. Good for small businesses wanting liability protection without corporate formality."),

    # Torts and Liability
    ("What is negligence?", "Negligence is failure to exercise reasonable care causing harm to another. Elements: duty (obligation to act reasonably), breach (failure to meet standard), causation (breach caused harm - actual and proximate cause), damages (actual harm suffered). Examples: car accidents, slip and fall, medical malpractice. Defenses: comparative negligence (plaintiff partly at fault), assumption of risk. Damages can include medical costs, lost wages, pain and suffering."),
    ("What is strict liability?", "Strict liability holds defendants responsible regardless of fault or intent. Applies to: abnormally dangerous activities (explosives, wild animals), product liability (defective products causing injury), statutory violations. Plaintiff must prove defect/danger and causation, not negligence. Defenses limited: assumption of risk, product misuse, alteration. Encourages safety in inherently dangerous activities."),
    ("What is defamation?", "Defamation is false statement harming someone's reputation. Libel: written defamation. Slander: spoken defamation. Elements: false statement of fact (not opinion), publication to third party, fault (negligence or actual malice for public figures), damages. Public figures must prove actual malice (knowledge of falsity or reckless disregard). Truth is absolute defense. Opinions generally protected."),
    ("What is medical malpractice?", "Medical malpractice is negligence by healthcare providers. Elements: duty (doctor-patient relationship), breach (deviation from standard of care), causation (breach caused injury), damages. Requires expert testimony on standard of care. Statute of limitations often 2-3 years from discovery. Many states have damage caps, especially on non-economic damages. Common claims: misdiagnosis, surgical errors, medication errors."),

    # Constitutional Rights
    ("What is the First Amendment?", "The First Amendment protects: freedom of religion (no establishment, free exercise), freedom of speech (including symbolic speech, with exceptions for incitement, true threats, obscenity), freedom of the press, right to peaceably assemble, right to petition government. These rights have limits - they primarily protect against government restriction, not private action. Content-based restrictions face strict scrutiny."),
    ("What is the Second Amendment?", "The Second Amendment protects the right to keep and bear arms. DC v. Heller (2008) held it protects individual right to possess firearms for self-defense in the home. McDonald v. Chicago (2010) applied this to states. Rights are not unlimited: regulations on concealed carry, prohibited persons (felons, mentally ill), dangerous weapons, and sensitive places can be constitutional. Ongoing litigation defines scope."),
    ("What is the Fourth Amendment?", "The Fourth Amendment protects against unreasonable searches and seizures. Requires warrants based on probable cause, describing place to be searched and items to be seized. Exceptions: consent, plain view, exigent circumstances, search incident to arrest, automobile exception, stop and frisk (reasonable suspicion). Exclusionary rule: illegally obtained evidence inadmissible. Applies to government actors, not private parties."),
    ("What is the Fifth Amendment?", "The Fifth Amendment provides: grand jury requirement for federal felonies, protection against double jeopardy (tried twice for same offense), privilege against self-incrimination (right to remain silent), due process requirement, takings clause (just compensation for eminent domain). 'Taking the Fifth' means refusing to answer questions that might incriminate you. Applies in criminal proceedings and civil cases where answers could lead to criminal liability."),
    ("What is the Sixth Amendment?", "The Sixth Amendment guarantees criminal defendants: speedy and public trial, impartial jury from district where crime occurred, notice of charges, right to confront witnesses, compulsory process to obtain witnesses, assistance of counsel. Gideon v. Wainwright (1963) requires free attorney for those who can't afford one in felony cases. These rights apply to 'criminal prosecutions,' not civil cases."),

    # Practical Legal Knowledge
    ("What should I do if I'm pulled over by police?", "Stay calm. Pull over safely and quickly. Turn off engine, turn on interior light if dark. Keep hands visible on steering wheel. Don't reach for documents until asked. Provide license, registration, insurance when requested. You can decline to answer questions beyond identification. You can refuse consent to search (though they may search anyway if they have probable cause). Stay polite and don't argue. You can record the interaction."),
    ("What should I do if I'm arrested?", "Remain calm and don't resist. You will be searched and handcuffed. Invoke your rights clearly: 'I am invoking my right to remain silent. I want a lawyer.' Don't answer questions without attorney present. Don't sign anything without attorney review. You're entitled to phone call. At arraignment, enter plea (usually not guilty initially). Don't discuss case with cellmates. Contact family and attorney immediately."),
    ("Do I need a lawyer?", "Consider a lawyer for: criminal charges (serious consequences), significant civil disputes (large amounts, complex issues), real estate transactions (major investment), business formation (liability protection), estate planning (ensuring wishes honored), immigration (complex rules, high stakes). Many lawyers offer free consultations. Legal aid available for those who qualify financially. Some matters (small claims, simple documents) may not require lawyers."),
    ("What is the statute of limitations?", "Statute of limitations sets deadline for filing legal claims. Varies by claim type and jurisdiction. Criminal: murder often has no limit, felonies typically 3-6 years, misdemeanors 1-2 years. Civil: personal injury 2-3 years, contracts 4-6 years, property damage 3-6 years. Clock usually starts when harm is discovered or should have been discovered. Tolling can pause the clock (defendant absent, plaintiff minor). Missing deadline bars claim."),
    ("What are my rights as a tenant?", "Tenants have rights to: habitable premises (heat, water, safety, structural integrity), quiet enjoyment (landlord can't harass or interfere), security deposit return (within statutory timeframe, itemized deductions), proper notice before entry (usually 24-48 hours except emergencies), protection from retaliation (for complaining or organizing), fair housing (no discrimination). Written lease recommended. Know your state's specific laws."),
]

# =============================================================================
# AMERICAN LIFE - Culture, Systems, and Practical Knowledge
# =============================================================================
AMERICAN_LIFE = [
    # Government and Civic Life
    ("How do US elections work?", "Federal elections held first Tuesday after first Monday in November. President: Electoral College system - voters choose electors, 270 electoral votes needed. Congress: direct popular vote in each state/district. Senators serve 6-year terms (1/3 elected every 2 years). Representatives serve 2-year terms. Primaries/caucuses select party nominees. Must register to vote. Voter ID requirements vary by state. Early voting and mail voting available in most states."),
    ("How does voting work in the US?", "Register to vote (requirements vary by state, deadlines typically 2-4 weeks before election). Check registration status. Find your polling place. Bring required ID if applicable. Vote in person on Election Day, during early voting period, or by mail/absentee. Ballot includes federal, state, and local races plus ballot measures. Results certified by states. Recounts possible if margins are small."),
    ("What is the Electoral College?", "The Electoral College elects the President. Each state gets electors equal to its congressional delegation (Senators + Representatives). 538 total electors, 270 needed to win. Most states use winner-take-all (candidate winning popular vote gets all electors). Maine and Nebraska split electors. Electors meet in December, Congress counts votes in January. A candidate can win presidency while losing national popular vote."),
    ("How do taxes work in the US?", "Federal income tax: progressive rates (10% to 37%), filed by April 15. State income tax: most states have it (0-13%), some have none. FICA: Social Security (6.2%) and Medicare (1.45%) on wages. Sales tax: state and local, on purchases (0-10%+). Property tax: local, on real estate value. Tax returns reconcile withholding with actual liability. Deductions and credits reduce tax owed."),
    ("What is Social Security?", "Social Security provides retirement, disability, and survivor benefits. Funded by payroll taxes (FICA). Retirement benefits: work 40 quarters (10 years), full benefits at 66-67 (rising), reduced at 62, increased if delayed to 70. Amount based on 35 highest-earning years. SSDI for disabled workers. SSI for low-income elderly/disabled. Apply through SSA, can do online."),
    ("What is Medicare?", "Medicare is federal health insurance for people 65+ and some disabled. Part A: hospital insurance (usually premium-free). Part B: medical insurance (monthly premium, ~$170). Part C: Medicare Advantage (private plans). Part D: prescription drugs (private plans). Enrollment periods: initial (around 65th birthday), annual (Oct-Dec). Supplemental Medigap policies cover gaps. Doesn't cover long-term care."),
    ("What is Medicaid?", "Medicaid is joint federal-state health insurance for low-income individuals. Eligibility varies by state but includes: low-income adults, children, pregnant women, elderly, disabled. Covers: hospital, doctor visits, long-term care, prescriptions. ACA expanded eligibility in many states. No or low premiums. Apply through state Medicaid office or healthcare.gov. Can have both Medicare and Medicaid."),

    # Education System
    ("How does the US education system work?", "K-12 public education is free and compulsory. Elementary school: K-5 (ages 5-11). Middle school: 6-8 (ages 11-14). High school: 9-12 (ages 14-18). Graduation requires credits in core subjects. Higher education: community colleges (2-year, associate's), universities (4-year, bachelor's). Graduate school: master's (1-2 years), doctorate (4-7 years). Public universities subsidized by states. Private schools at all levels."),
    ("How do I apply to college?", "Research schools (fit, programs, cost). Take standardized tests (SAT/ACT, though many now test-optional). Request transcripts and letters of recommendation. Write personal essays. Complete applications (Common App or school-specific). Apply for financial aid (FAFSA opens October 1). Early decision/action deadlines November, regular deadlines January. Decisions March-April. Commit by May 1. Consider community college as affordable path."),
    ("What is FAFSA?", "FAFSA (Free Application for Federal Student Aid) determines eligibility for federal financial aid. Opens October 1 for following school year. Required for federal grants (Pell), loans (Direct), and work-study. Many states and schools also require it. Uses family income/assets to calculate Expected Family Contribution (EFC). Submit early - some aid is first-come. Results in Student Aid Report (SAR). Schools use it for aid packages."),
    ("What types of student financial aid exist?", "Grants: free money, need-based (Pell Grant up to ~$7,000) or merit-based. Scholarships: free money, merit or criteria-based (many sources - schools, organizations, employers). Work-study: part-time campus jobs. Loans: must repay with interest - federal (subsidized, unsubsidized, PLUS) better terms than private. Federal loans offer income-driven repayment and forgiveness programs. Always maximize grants/scholarships before borrowing."),
    ("How do student loans work?", "Federal loans: subsidized (government pays interest while in school, need-based) and unsubsidized (interest accrues from disbursement). Interest rates set annually. Repayment begins 6 months after graduation. Plans: standard (10 years), graduated, extended, income-driven (IBR, PAYE, SAVE - payments based on income, forgiveness after 20-25 years). Public Service Loan Forgiveness after 10 years of qualifying employment. Private loans have fewer protections."),

    # Healthcare
    ("How does health insurance work in the US?", "Insurance pays for medical care in exchange for premiums. Types: employer-sponsored (most common), individual market (healthcare.gov), Medicare (65+), Medicaid (low-income). Key terms: premium (monthly payment), deductible (pay before insurance kicks in), copay (fixed amount per visit), coinsurance (percentage you pay), out-of-pocket maximum (yearly limit). In-network providers cost less. Open enrollment typically November-January."),
    ("What is the Affordable Care Act (Obamacare)?", "ACA (2010) reformed health insurance: requires coverage of pre-existing conditions, allows children on parents' plans until 26, created health insurance marketplaces (healthcare.gov), expanded Medicaid (in participating states), subsidies for those earning 100-400% of poverty level. Essential health benefits must be covered. Individual mandate penalty eliminated but some states have their own mandates."),
    ("What should I do in a medical emergency?", "Call 911 for life-threatening emergencies (chest pain, difficulty breathing, severe bleeding, stroke symptoms, serious injuries). Go to emergency room for urgent serious conditions. Urgent care for non-life-threatening issues (minor injuries, infections). Primary care for routine and preventive care. Know nearest ER location. Emergency rooms must treat regardless of ability to pay (EMTALA). Expect a bill - emergency care is expensive."),
    ("What is a Health Savings Account (HSA)?", "HSA is tax-advantaged account for medical expenses, paired with high-deductible health plan (HDHP). Triple tax benefit: contributions tax-deductible, growth tax-free, withdrawals tax-free for qualified medical expenses. 2024 limits: $4,150 individual, $8,300 family. Funds roll over and are portable. Can invest for growth. After 65, can withdraw for any purpose (taxed like IRA). FSA is similar but use-it-or-lose-it."),

    # Employment
    ("What are US employment laws?", "Key federal laws: Fair Labor Standards Act (minimum wage, overtime), Title VII (anti-discrimination), ADA (disability accommodation), FMLA (unpaid leave for family/medical), OSHA (workplace safety). Most employment is at-will (either party can end for any reason except illegal discrimination). States may have additional protections. Independent contractors have fewer protections than employees."),
    ("What is minimum wage?", "Federal minimum wage: $7.25/hour (since 2009). Many states and cities have higher minimums ($15+ in some areas). Tipped workers have lower minimum ($2.13 federal) but must reach regular minimum with tips. Some exceptions for young workers, disabled workers, students. Overtime (1.5x pay) required after 40 hours/week for non-exempt employees. Exempt employees (salaried professionals/managers over ~$35k) don't get overtime."),
    ("What is at-will employment?", "At-will employment means either employer or employee can end the relationship at any time, for any reason (or no reason), without notice. Exceptions: can't fire for illegal discrimination (race, sex, religion, etc.), retaliation (for reporting violations), or violating public policy. Employment contracts can modify at-will status. Some states require final paycheck timing. Unemployment insurance may be available if fired without cause."),
    ("How do I file for unemployment?", "File with your state's unemployment office (online usually). Eligibility: lost job through no fault of your own, earned enough wages, able and available to work, actively seeking work. Provide: SSN, work history, reason for separation. Benefits typically 26 weeks (extended in recessions), amount based on previous wages. Must certify weekly/biweekly that you're searching. Benefits are taxable income."),
    ("What is workers' compensation?", "Workers' comp provides benefits for work-related injuries/illnesses regardless of fault. Covers: medical treatment, wage replacement (typically 2/3 of wages), disability payments, death benefits. Almost all employers must carry insurance. Report injury immediately. File claim with state workers' comp board. Can't sue employer for covered injuries (exclusive remedy). Retaliation for filing is illegal."),
    ("What should I know about my 401(k)?", "401(k) is employer-sponsored retirement account. Contributions from paycheck (pre-tax traditional or post-tax Roth). 2024 limit: $23,000 (+$7,500 catch-up if 50+). Many employers match contributions (free money - always contribute enough to get full match). Investment options typically include funds. Vesting schedule may apply to employer match. Early withdrawal before 59½ has 10% penalty + taxes. Required distributions at 73."),

    # Housing
    ("How do I rent an apartment?", "Search listings (Zillow, Apartments.com, Craigslist). Visit in person. Application requires: ID, SSN, income verification (pay stubs, tax returns), credit check, references, application fee. Landlord checks credit score, income (typically 3x rent), rental history. If approved: sign lease, pay security deposit (usually 1-2 months rent) and first month's rent. Read lease carefully. Know your rights as tenant."),
    ("How do I buy a house?", "Get pre-approved for mortgage (determines budget). Find real estate agent. Search for homes. Make offer (typically with contingencies for inspection, financing). Negotiate. Get home inspection. Finalize mortgage (underwriting). Get homeowner's insurance. Closing: sign documents, pay closing costs (2-5% of price), receive keys. Down payment typically 3-20% (less requires PMI). Process takes 30-60 days after accepted offer."),
    ("What is a mortgage?", "Mortgage is loan to buy property, secured by the property itself. Types: conventional (private lender), FHA (lower down payment, government-backed), VA (for veterans), USDA (rural areas). Terms: typically 15 or 30 years. Fixed rate (same payment) vs adjustable rate (can change). Monthly payment includes principal, interest, property taxes, homeowner's insurance (PITI). Pre-approval shows sellers you're serious."),
    ("What is a credit score?", "Credit score (300-850) predicts likelihood of repaying debt. Factors: payment history (35%), amounts owed (30%), length of history (15%), new credit (10%), credit mix (10%). Good score: 670+, excellent: 740+. Affects: loan approval, interest rates, apartment rentals, insurance rates, some jobs. Build credit: pay bills on time, keep utilization low (<30%), don't close old accounts, limit applications. Check free at annualcreditreport.com."),
    ("What is renter's insurance?", "Renter's insurance covers your belongings in a rental. Protects against: theft, fire, water damage (not floods), some natural disasters. Also includes liability coverage (if someone is injured in your unit) and additional living expenses (if displaced). Typically $15-30/month. Landlord's insurance covers building, not your stuff. Document belongings with photos/receipts. Worth it for protection against loss."),

    # Transportation
    ("How do I get a driver's license?", "Requirements vary by state but generally: be minimum age (16-18), pass written test on traffic laws, pass vision test, hold learner's permit for required period (6 months-1 year), complete supervised driving hours, pass road test. Bring proof of identity, residency, SSN. Fee varies. License valid for 4-8 years. Real ID compliant license needed for domestic flights starting May 2025."),
    ("What do I need to know about car insurance?", "Car insurance is required in almost all states. Types: liability (required - covers damage you cause to others), collision (covers your car in accidents), comprehensive (theft, weather, etc.), uninsured motorist. Factors affecting rates: age, driving record, location, car type, coverage levels. Get multiple quotes. Minimum liability limits often insufficient - consider higher limits. Deductible affects premium."),
    ("What happens if I get a traffic ticket?", "Options: pay fine (admits guilt, points on license, may increase insurance), contest in court (must appear, present defense), traffic school (some violations, keeps points off license). Serious violations (DUI, reckless driving) require court appearance and may have criminal consequences. Too many points can lead to license suspension. Some tickets can be negotiated to lesser violations."),
    ("What should I do after a car accident?", "Ensure safety, move to safe location if possible. Call 911 if injuries or significant damage. Exchange information: name, contact, insurance, license, plate numbers. Document scene: photos of damage, positions, conditions. Get witness information. Don't admit fault. Report to police (required for injuries or significant damage). Notify your insurance company promptly. Seek medical attention even for minor symptoms."),

    # Daily Life
    ("How does tipping work in the US?", "Tipping is customary in service industries. Restaurants: 15-20% of pre-tax bill (more for excellent service, less for poor). Bars: $1-2 per drink or 15-20% of tab. Food delivery: 15-20% or $3-5 minimum. Hair salons: 15-20%. Taxis/rideshare: 15-20%. Hotels: $2-5/night for housekeeping, $1-2/bag for bellhop. Coffee shops: optional, $1-2 or tip jar. Service workers often rely on tips as significant income."),
    ("How does the postal system work?", "USPS is federal postal service. Services: First Class Mail (letters, 1-3 days), Priority Mail (1-3 days, flat rate available), Media Mail (books, slow, cheap), packages by weight/distance. Buy stamps at post office, online, or retailers. Mailboxes for outgoing mail. Track packages online. Also: UPS, FedEx for shipping. Forward mail when moving (online or at post office). PO Boxes available for rent."),
    ("What are typical banking services?", "Banks offer: checking accounts (daily transactions, debit card), savings accounts (interest-earning, limited transactions), CDs (fixed term, higher interest), loans, credit cards. Credit unions are member-owned alternatives, often better rates. Online banks offer higher savings rates. FDIC insures deposits up to $250,000. Direct deposit for paychecks. Online bill pay. Mobile check deposit. ATM networks."),
    ("How do I build credit in America?", "Start with: secured credit card (deposit becomes limit), become authorized user on family member's card, credit-builder loan. Use credit card for small purchases, pay full balance monthly. Keep utilization below 30%. Never miss payments. Don't apply for too much credit at once. Mix of credit types helps (cards, loans). Takes 6+ months to establish score. Monitor credit reports for errors."),
    ("What should I know about consumer rights?", "Key protections: Fair Credit Reporting Act (dispute credit report errors), Fair Debt Collection Practices Act (limits collector behavior), Truth in Lending Act (disclosure requirements), warranties (express and implied). Return policies vary by retailer. Cooling-off rule: 3 days to cancel some door-to-door sales. Dispute fraudulent charges within 60 days. Lemon laws for defective vehicles. FTC handles consumer complaints."),

    # Communication and Utilities
    ("How do phone plans work in the US?", "Major carriers: AT&T, Verizon, T-Mobile. Options: postpaid (monthly bill, credit check, may include device payment) or prepaid (pay in advance, no contract). Plans include talk, text, data (limited or unlimited). Consider coverage in your area. MVNOs (Mint, Visible) use major networks at lower cost. WiFi calling helps with poor coverage. International plans available. Port number to keep it when switching."),
    ("How do I set up utilities?", "When moving: contact utility companies to start service (electric, gas, water, trash). May require deposit based on credit. Some included in rent. Internet/cable: multiple providers usually available, compare prices and speeds. Average costs: electricity $100-150, gas $50-100, water $30-50, internet $50-100. Set up autopay to avoid late fees. Many offer budget billing (fixed monthly amount)."),
    ("What is the 911 emergency system?", "911 is the universal emergency number in the US. Call for: police (crimes, suspicious activity), fire (fire, smoke, gas smell), medical (injuries, illness, overdose). Dispatcher will ask location and nature of emergency. Stay calm, answer questions. Don't hang up until told. Text-to-911 available in many areas. Non-emergency police lines for reporting that doesn't require immediate response."),

    # Culture and Social Norms
    ("What are common American social customs?", "Greetings: handshake for business, wave or 'hi' casually. Personal space: arm's length typical. Eye contact shows engagement. Small talk is normal (weather, sports). Punctuality valued - arrive on time or slightly early. RSVP if requested. Thank-you notes appreciated. Remove shoes in some homes (ask or observe). Splitting the check is common among friends. Casual dress acceptable most places."),
    ("What are major American holidays?", "Federal holidays (banks, government closed): New Year's Day (Jan 1), MLK Day (3rd Mon Jan), Presidents Day (3rd Mon Feb), Memorial Day (last Mon May), Independence Day (July 4), Labor Day (1st Mon Sept), Columbus Day (2nd Mon Oct), Veterans Day (Nov 11), Thanksgiving (4th Thu Nov), Christmas (Dec 25). Other widely celebrated: Easter, Halloween, Valentine's Day. Many get day off for Thanksgiving and Christmas."),
    ("What should I know about American food culture?", "Diverse cuisine reflecting immigration: Mexican, Italian, Chinese, etc. plus regional American (Southern, Tex-Mex, BBQ). Fast food prevalent but health consciousness growing. Portion sizes typically large. Breakfast common (eggs, bacon, cereal, pancakes). Lunch quick (sandwiches, salads). Dinner main meal. Coffee culture strong. Tipping expected at sit-down restaurants. Food allergies taken seriously - ask about ingredients."),
    ("What are American measurement units?", "US uses Imperial system, not metric. Length: inch (2.54 cm), foot (12 inches), yard (3 feet), mile (5,280 feet/1.6 km). Weight: ounce (28g), pound (16 oz/454g), ton (2,000 lbs). Volume: cup (8 oz), pint (2 cups), quart (2 pints), gallon (4 quarts/3.8L). Temperature: Fahrenheit (32°F = 0°C, 212°F = 100°C). Quick conversions: double Celsius + 30 ≈ Fahrenheit."),

    # Practical Skills
    ("How do I file taxes in the US?", "Tax year is calendar year. File by April 15 (extensions available). W-2 from employers, 1099s for other income. Choose: tax software (TurboTax, H&R Block, free options for simple returns), professional preparer, or IRS Free File if income under ~$79,000. Standard deduction ($14,600 single, $29,200 married 2024) vs itemizing. Common credits: Earned Income, Child Tax. Pay estimated taxes if self-employed. Keep records 7 years."),
    ("What should I know about retirement savings?", "Start early - compound interest matters. Employer 401(k): contribute enough to get full match (free money). Traditional 401(k)/IRA: tax-deductible contributions, taxed on withdrawal. Roth: after-tax contributions, tax-free growth and withdrawal. IRA contribution limit $7,000 (2024). Target date funds simplify investing. Social Security provides base but often insufficient. Aim to save 10-15% of income."),
    ("How do I handle a medical bill?", "Review bill for errors (common). Request itemized bill. Check against insurance explanation of benefits (EOB). If uninsured or high cost: ask for cash discount (often 20-50% off), request financial assistance (most hospitals have programs), negotiate payment plan. Medical debt is negotiable. Don't ignore - can go to collections and affect credit. Know statute of limitations in your state."),
    ("What should I do if I can't pay bills?", "Contact creditors before missing payments - many offer hardship programs. Prioritize: housing, utilities, food, transportation, medical. Government assistance: SNAP (food), LIHEAP (utilities), Medicaid (health). Nonprofit assistance: United Way 211, food banks, community organizations. Avoid payday loans (predatory interest). Credit counseling (nonprofit) can help negotiate. Bankruptcy is last resort but provides fresh start."),
    ("How does the American job search work?", "Update resume (1-2 pages, achievements not just duties). LinkedIn profile important. Search: company websites, Indeed, LinkedIn, Glassdoor, networking. Tailor resume and cover letter for each application. Prepare for interviews: research company, practice common questions, prepare questions to ask. Follow up with thank-you email. Background and reference checks common. Negotiate salary (research market rates). Read offer letter carefully."),

    # US Geography and Regions
    ("What are the major regions of the United States?", "The US has distinct regions: Northeast (New England, Mid-Atlantic) - historic, urban, finance/education centers. Southeast/South - warmer climate, growing economy, hospitality culture. Midwest - agricultural heartland, manufacturing, friendly communities. Southwest - desert climate, Hispanic influence, growing tech. West Coast - tech industry, entertainment, progressive politics. Mountain West - outdoor recreation, ranching, lower population density. Each region has unique culture, accents, food, and traditions."),
    ("What are the largest US cities?", "By population: New York City (~8.3M, finance, media, culture), Los Angeles (~4M, entertainment, tech), Chicago (~2.7M, Midwest hub, architecture), Houston (~2.3M, energy, space), Phoenix (~1.6M, retirement, tech growth), Philadelphia (~1.6M, historic, healthcare), San Antonio (~1.5M, military, tourism), San Diego (~1.4M, military, biotech), Dallas (~1.3M, business, telecom), San Jose (~1M, Silicon Valley tech). Metro areas are much larger."),
    ("What is cost of living like in different US areas?", "Varies dramatically by location. Expensive: San Francisco, NYC, Boston, Seattle, LA - housing can be 3-5x national average. Moderate: Denver, Austin, Portland, Nashville, Atlanta - growing cities with rising costs. Affordable: Midwest (Ohio, Indiana), South (Texas, Tennessee, Georgia), rural areas - lower housing, wages may also be lower. Consider: housing (largest expense), taxes (vary by state), utilities, transportation. Use cost of living calculators when relocating."),
    ("What states have no income tax?", "Nine states have no state income tax: Alaska, Florida, Nevada, New Hampshire (dividends/interest only), South Dakota, Tennessee, Texas, Washington, Wyoming. However, these states may have higher property taxes, sales taxes, or other fees. Texas and Florida are popular for tax savings. Washington has no income tax but high sales tax. Consider total tax burden, not just income tax, when evaluating states."),
    ("What is the difference between rural and urban living in America?", "Urban: higher population density, public transit available, walking/biking possible, diverse dining and entertainment, higher costs, more job opportunities, smaller living spaces. Rural: lower density, car essential, larger properties, lower costs, fewer amenities, stronger community ties, outdoor access, limited job market. Suburban: middle ground, car-dependent, family-oriented, good schools often prioritized. Many Americans move between these throughout life."),

    # Weather and Natural Disasters
    ("What types of weather should I expect in America?", "Varies by region. Northeast: four distinct seasons, cold snowy winters, warm humid summers. Southeast: mild winters, hot humid summers, hurricanes June-November. Midwest: extreme temperature swings, cold winters, tornadoes spring/summer. Southwest: hot dry summers, mild winters, monsoon season. Pacific Northwest: mild year-round, lots of rain (west of Cascades). California: Mediterranean climate, dry summers, wildfire risk. Mountain West: cold winters, significant snow, altitude affects temperature."),
    ("How do I prepare for natural disasters in the US?", "Know your area's risks. Hurricanes: evacuation routes, supplies, board windows. Tornadoes: shelter plan (basement, interior room), NOAA weather radio. Earthquakes: secure heavy items, drop/cover/hold on, emergency kit. Wildfires: defensible space, evacuation bag, air quality masks. Floods: flood insurance (separate from homeowners), don't drive through water. Winter storms: supplies for power outages, safe heating. General: 72-hour emergency kit, water, food, first aid, important documents."),
    ("What is tornado alley?", "Tornado Alley is a region of central US with frequent tornadoes, roughly from Texas through Oklahoma, Kansas, Nebraska, and South Dakota. Warm moist Gulf air meets cold dry Canadian air, creating severe thunderstorms. Peak season: April-June. Tornadoes also occur in 'Dixie Alley' (Southeast) and elsewhere. Watch means conditions favorable; Warning means tornado spotted - take shelter immediately. Mobile homes are particularly dangerous."),
    ("What should I know about hurricane season?", "Atlantic hurricane season: June 1 - November 30, peak August-October. Affects Gulf Coast (Texas to Florida) and Atlantic Coast. Categories 1-5 based on wind speed. Prepare: know evacuation zones and routes, hurricane supplies (water, food, batteries, medications), protect property (shutters, sandbags). When approaching: follow evacuation orders, fill gas tank early, secure important documents. After: avoid floodwater, check for damage, wait for official all-clear."),
    ("How do earthquakes work in the US?", "Most earthquakes occur along fault lines. California's San Andreas fault is most famous. Pacific Northwest has Cascadia subduction zone (megaquake risk). New Madrid zone in Midwest has historical large quakes. Prepare: secure heavy furniture, know drop/cover/hold on, have emergency supplies. After: check for gas leaks, avoid damaged buildings, expect aftershocks. Earthquake insurance is separate from homeowners and expensive in high-risk areas."),

    # Sports and Entertainment
    ("What are the major American sports?", "Football (NFL): most popular, season September-February, Super Bowl is biggest TV event. Basketball (NBA): October-June, global appeal. Baseball (MLB): 'America's pastime,' April-October, 162-game season. Hockey (NHL): October-June, biggest in northern states/Canada. Soccer (MLS): growing rapidly, season March-November. College sports (especially football, basketball) have huge followings. Also popular: golf, tennis, NASCAR, UFC."),
    ("How does the NFL work?", "NFL has 32 teams in two conferences (AFC, NFC). Regular season: 17 games, September-January. Playoffs: top 7 teams each conference, single elimination. Super Bowl: conference champions, usually early February, massive cultural event. Fantasy football extremely popular. Teams: salary cap promotes parity. Draft: worst teams pick first. Famous teams: Cowboys, Patriots, Packers, 49ers, Steelers. Games mostly Sundays, plus Thursday and Monday nights."),
    ("What are popular American entertainment options?", "Streaming: Netflix, Hulu, Disney+, HBO Max, Amazon Prime. Cable TV declining but still used. Movie theaters remain popular for blockbusters. Live music: concerts, festivals (Coachella, Lollapalooza), local venues. Broadway in NYC, touring shows nationally. Theme parks: Disney, Universal, Six Flags. Museums, zoos, aquariums in major cities. Outdoor recreation varies by region. Bars, restaurants, bowling, escape rooms for social entertainment."),
    ("How does Hollywood and the American film industry work?", "Hollywood (Los Angeles area) is center of US film/TV production, though filming occurs nationwide. Major studios: Disney, Warner Bros, Universal, Paramount, Sony. Streaming has disrupted traditional model. Box office measures theatrical success. Awards: Oscars (Academy Awards) in February/March. Film ratings: G, PG, PG-13, R, NC-17. TV ratings: TV-Y through TV-MA. Independent films often debut at festivals (Sundance, SXSW)."),
    ("What is American music culture like?", "Extremely diverse. Origins: blues, jazz, country, rock and roll - all American inventions. Current genres: pop, hip-hop/rap (dominant), country, rock, R&B, electronic. Nashville: country music capital. New Orleans: jazz birthplace. Memphis: blues. Detroit: Motown. Live music culture strong - concerts, festivals, local venues. Streaming (Spotify, Apple Music) dominant. Radio still relevant, especially country and pop formats."),

    # Food and Dining
    ("What is American food culture really like?", "Incredibly diverse due to immigration. Regional cuisines: Southern (BBQ, fried chicken, biscuits), Cajun/Creole (Louisiana), Tex-Mex (Southwest), New England seafood, Midwestern comfort food. Ethnic cuisines everywhere: Chinese, Mexican, Italian, Thai, Indian, Vietnamese, Korean. Fast food originated here and is prevalent. Farm-to-table and organic movements growing. Food portions typically large. Food trucks popular in cities."),
    ("How does dining out work in America?", "Restaurants range from fast food to fine dining. Casual dining: seat yourself or wait to be seated. Server takes order, brings food, checks on you. Bill comes when requested or automatically. Tip 15-20% of pre-tax total. Fast casual (Chipotle, Panera): order at counter, food brought to table, tipping optional. Fine dining: reservations recommended, dress code possible, courses served, higher prices, tip 20%+."),
    ("What are popular American fast food chains?", "Burgers: McDonald's, Burger King, Wendy's, Five Guys, In-N-Out (West Coast). Chicken: Chick-fil-A, Popeyes, KFC. Mexican: Taco Bell, Chipotle, Qdoba. Pizza: Domino's, Pizza Hut, Papa John's. Subs: Subway, Jersey Mike's, Jimmy John's. Coffee: Starbucks, Dunkin'. Regional favorites: Whataburger (Texas), Culver's (Midwest), Wawa/Sheetz (Mid-Atlantic). Many have mobile apps, drive-throughs, delivery."),
    ("What is BBQ culture in America?", "BBQ varies significantly by region - passionate debates about which is best. Texas: beef brisket, simple salt and pepper rub, post oak wood. Kansas City: variety of meats, sweet tomato-based sauce. Memphis: pork ribs and pulled pork, dry rub or wet sauce. Carolina: pulled pork, vinegar-based (Eastern) or mustard-based (South Carolina) sauce. Low and slow cooking (hours at low temperature) is the key technique. BBQ restaurants and competitions are cultural institutions."),
    ("What are American drinking laws and culture?", "Drinking age: 21 nationwide (strictly enforced, ID required). DUI/DWI: serious criminal offense, 0.08% BAC limit (lower in some states), severe penalties. Open container laws: can't have open alcohol in vehicle in most states. Bars typically close 1-2 AM (varies by locality). Alcohol sold in stores - some states have restrictions (no Sunday sales, state-run stores). Craft beer, wine, cocktail culture strong. Designated driver concept important."),

    # Shopping and Consumer Culture
    ("How does shopping work in America?", "Retail types: big-box stores (Walmart, Target, Costco), department stores (Macy's, Nordstrom), specialty stores, malls, strip malls, outlets. E-commerce dominant (Amazon). Prices usually exclude tax (added at register). Sales common, especially holiday weekends (Black Friday, Memorial Day, Labor Day). Return policies generally generous (30-90 days with receipt). Credit cards widely accepted. Many stores have loyalty programs."),
    ("What are the major American shopping holidays?", "Black Friday: day after Thanksgiving, massive sales, starts retail holiday season. Cyber Monday: online shopping day after Black Friday. Prime Day: Amazon's summer sale event. Memorial Day, Labor Day, Presidents Day: sales on furniture, appliances, cars. Back to school: late summer sales on clothing, electronics. After-Christmas sales: deep discounts on holiday items. Tax-free weekends: some states, usually before school year."),
    ("What should I know about Costco and warehouse stores?", "Costco, Sam's Club, BJ's require annual membership ($60-120). Bulk purchases at lower per-unit prices. Best for: large families, small businesses, specific items (electronics, pharmacy, gas). Costco Kirkland brand is high quality. Gas typically cheapest. Generous return policy. Food court has very cheap hot dogs, pizza. Not worth it for small households unless buying specific high-value items."),
    ("How does Amazon work in America?", "Amazon dominates US e-commerce. Standard shipping free over $35 or with Prime. Prime ($139/year): free 2-day shipping, Prime Video, Music, other benefits. Subscribe & Save: recurring delivery discount. Amazon Fresh/Whole Foods: grocery delivery. Locker pickup available. Returns usually easy - drop at UPS, Whole Foods, Kohl's. Marketplace has third-party sellers (check ratings). Price tracking tools helpful (CamelCamelCamel)."),

    # Technology and Internet
    ("How does internet service work in the US?", "ISP options vary by location - often limited competition. Types: cable (common, 100-1000 Mbps), fiber (fastest, limited availability), DSL (older, slower), satellite (rural areas, higher latency), fixed wireless. Major providers: Comcast/Xfinity, Spectrum, AT&T, Verizon. Typical costs: $50-100/month. Bundles with TV/phone available. Equipment rental vs buying your own modem/router. Speed needs: streaming needs 25+ Mbps, gaming needs low latency."),
    ("What cell phone options exist in the US?", "Major carriers: Verizon (best coverage), AT&T, T-Mobile (best value/5G). MVNOs use major networks cheaper: Mint Mobile, Visible, Cricket, Metro by T-Mobile. Plan types: postpaid (credit check, device financing) or prepaid (no contract). Unlimited plans common but may have deprioritization. Consider coverage in your area. Most phones work on any carrier now. Keep phone number when switching (porting). eSIM increasingly common."),
    ("What streaming services are popular in America?", "Video: Netflix (originals), Hulu (next-day TV), Disney+ (Disney/Marvel/Star Wars), HBO Max (prestige), Amazon Prime Video, Apple TV+, Peacock (NBC), Paramount+. Music: Spotify, Apple Music, YouTube Music, Amazon Music. Live TV: YouTube TV, Hulu Live, Sling TV. Average household has 4+ subscriptions. Password sharing being restricted. Bundles available (Disney/Hulu/ESPN). Many have ad-supported cheaper tiers."),
    ("What should I know about social media in America?", "Platforms: Facebook (older demographics), Instagram (photos/stories), TikTok (short videos, younger), Twitter/X (news, discourse), LinkedIn (professional), YouTube (video), Snapchat (younger, ephemeral), Reddit (communities/forums). Privacy concerns ongoing. Employer may check social media. Cyberbullying is issue for kids. Digital literacy important - verify information. Screen time concerns growing, especially for children."),
    ("How does smart home technology work?", "Ecosystems: Amazon Alexa, Google Home, Apple HomeKit. Smart speakers control devices by voice. Common devices: smart lights (Philips Hue), thermostats (Nest, Ecobee), doorbells (Ring), locks, cameras, plugs. Hub may be needed for some devices. WiFi network capacity matters. Security/privacy considerations with always-listening devices. Automation possible (lights at sunset, thermostat scheduling). Growing but not yet universal."),

    # Relationships and Family
    ("How does dating work in America?", "Dating apps dominant: Tinder, Bumble, Hinge, OkCupid, Match. Traditional meeting still happens (work, friends, activities). Early dating casual - multiple dates before exclusivity. 'Talking stage' before official dating common. Typical progression: dating → exclusive → relationship → living together → engagement → marriage. Expectations vary by region, age, culture. Splitting the check increasingly common. Communication about expectations important."),
    ("What is American wedding culture like?", "Engagement: proposal with ring, typically 1-2 year engagement. Weddings range from simple courthouse to elaborate events. Average cost: $30,000+ (highly variable). Common elements: ceremony (religious or civil), reception with dinner/dancing, wedding party (bridesmaids, groomsmen). Save-the-dates, invitations sent. Registry for gifts at major stores. Rehearsal dinner night before. Bachelor/bachelorette parties. Tipping vendors expected. Destination weddings growing."),
    ("What should I know about raising children in America?", "Parental leave: no federal paid requirement (some states have it, some employers offer it). Childcare expensive: $1,000-2,500/month for daycare. Options: daycare centers, in-home daycare, nannies, au pairs, family. Pre-K: available in some areas, head start for low-income. K-12: public free, private expensive. Activities (sports, music) common but costly. College savings: 529 plans. Healthcare: kids covered by parents' insurance to 26."),
    ("How does K-12 education work for parents?", "Public schools: free, assigned by residence (school district quality affects home values). Private schools: religious or secular, tuition-based. Charter schools: public but independently run. Homeschooling: legal in all states, regulations vary. School choice varies by state. Involvement expected: PTA, conferences, volunteering. Sports, arts, clubs available. Gifted and special education services. School starts between 7-9 AM typically."),
    ("What is the college process for American families?", "Planning starts in high school: grades, standardized tests (SAT/ACT becoming optional), extracurriculars. College visits junior/senior year. Applications due November (early) or January (regular). FAFSA for financial aid. Acceptance letters March-April. Decision by May 1. Cost varies dramatically: community college ~$3,500/year, state school $10,000-25,000, private $50,000-80,000. Room and board additional. Student loans common. Community college to four-year transfer saves money."),
    ("How do Americans care for aging parents?", "Aging in place preferred by most - home modifications, in-home care. Family often provides care (usually daughters). Options: independent living, assisted living ($4,000-7,000/month), nursing homes ($8,000-10,000/month), memory care. Medicare covers medical but NOT long-term care. Medicaid covers long-term care if assets depleted. Long-term care insurance expensive and complex. Family and Medical Leave Act provides unpaid leave. Caregiver burnout is real concern."),
    ("What happens with death and funerals in America?", "When someone dies: call 911 or funeral home. Death certificate needed for legal/financial matters. Funeral home handles body, can arrange services. Options: burial (casket, cemetery plot - $7,000-12,000+), cremation (cheaper, growing preference - $2,000-5,000), green burial. Services: viewing/visitation, funeral service (religious or secular), graveside service, reception. Obituary in newspaper/online. Estate goes through probate unless avoided with trusts. Grief counseling available."),

    # Personal Health and Wellness
    ("How do Americans approach fitness and exercise?", "Gym memberships common: national chains (Planet Fitness, LA Fitness, Equinox) and local gyms. Cost: $10-200/month depending on amenities. Home gyms popular since COVID. Running, cycling, hiking popular outdoor activities. Boutique fitness: CrossFit, Orange Theory, SoulCycle, yoga studios. Sports leagues for adults (recreational). Apps: Peloton, Nike Training, fitness trackers (Fitbit, Apple Watch). Corporate wellness programs at some employers."),
    ("What should I know about mental health in America?", "Mental health awareness has increased significantly. Resources: therapists/counselors, psychiatrists (can prescribe), psychologists. Finding providers: insurance directory, Psychology Today, ask doctor. Cost: often covered by insurance with copay, sliding scale available. Telehealth therapy widely available now. Crisis resources: 988 Suicide and Crisis Lifeline, Crisis Text Line (text HOME to 741741). Stigma decreasing but still exists. Many employers offer EAP (Employee Assistance Program) with free sessions."),
    ("What is the opioid crisis?", "The opioid epidemic has killed hundreds of thousands of Americans since 1999. Started with overprescription of painkillers. Many moved to heroin, then fentanyl (extremely potent, often in other drugs). Overdose deaths peaked around 95,000/year. Naloxone (Narcan) reverses overdoses - now available without prescription. Treatment: medication-assisted treatment (Suboxone, methadone), rehab, support groups. Affects all demographics but hit rural areas hard."),
    ("What should I know about health insurance and prescriptions?", "Prescription coverage (Part D for Medicare, included in most plans). Formulary determines which drugs covered at what tier. Generics much cheaper than brand name - ask for them. GoodRx app shows pharmacy price comparisons. Mail-order pharmacies often cheaper for maintenance medications. Prior authorization may be required for expensive drugs. Manufacturer coupons available for some brand drugs. Patient assistance programs for those who qualify."),
    ("How does dental and vision insurance work?", "Often separate from medical insurance. Dental: typically covers preventive (cleanings, x-rays) 100%, basic (fillings) 80%, major (crowns) 50%. Annual maximum usually $1,000-2,000. Vision: usually covers annual exam, allowance for glasses or contacts. Discount programs exist if no insurance. Dental schools offer cheaper care. Routine dental care important - problems become expensive. Many consider dental insurance marginally useful given limits."),

    # Pets and Animals
    ("What should I know about having pets in America?", "Dogs and cats most common. Responsibilities: food, veterinary care (vaccines, annual exams, emergencies), housing that allows pets (many rentals restrict), time and attention. Costs: dog $1,000-3,000/year, cat $500-1,000/year. Adoption from shelters encouraged over buying (shelter dogs/cats need homes, puppy mills are problematic). Spay/neuter expected. Microchipping recommended. Pet insurance available ($30-100/month). Many areas have leash laws."),
    ("How does pet adoption work?", "Shelters: local animal shelters, humane societies have dogs, cats, others. Adoption fees $50-300 (covers spay/neuter, vaccines). Process: application, sometimes home check, meet the animal. Rescue organizations: breed-specific or regional, may be more selective. Petfinder.com lists adoptable pets. Foster-to-adopt available. Benefits: saving a life, usually already spayed/neutered, lower cost than breeders. Some pets have unknown history or behavioral needs."),
    ("What are typical veterinary costs?", "Routine care: annual exam $50-100, vaccines $20-100 each, spay/neuter $200-500. Dental cleaning $300-800. Emergency care can be thousands (surgery $2,000-10,000+). Pet insurance helps ($30-100/month) but has deductibles and exclusions, doesn't cover pre-existing conditions. Many vets offer payment plans. Low-cost clinics available in some areas. Preventive care (vaccines, flea/tick, heartworm) is much cheaper than treatment."),

    # Home and Property
    ("What should homeowners know about maintenance?", "Regular maintenance prevents expensive repairs. HVAC: change filters monthly, annual professional service. Plumbing: know shut-off locations, address small leaks. Roof: inspect annually, clean gutters. Exterior: maintain paint/siding, grade soil away from foundation. Appliances: clean refrigerator coils, dryer vents. Seasonal: winterize (pipes, outdoor faucets), spring (AC service, yard). Budget 1-2% of home value annually for maintenance."),
    ("How do home repairs and contractors work?", "For major work, get multiple quotes (3 is standard). Check licensing (required for many trades), insurance, references. Written contracts essential - scope, timeline, payment schedule. Don't pay everything upfront. Permits required for major work (homeowner pulls or contractor pulls). Inspections at key stages. Be present or check in regularly. Final payment after satisfactory completion. Small jobs: handyman may be more cost-effective."),
    ("What should I know about HOAs (Homeowners Associations)?", "HOAs govern many condos, townhomes, and some single-family neighborhoods. Monthly/quarterly fees ($100-500+) cover common areas, amenities, sometimes exterior maintenance. CC&Rs (rules) govern what you can do (paint colors, landscaping, parking, rentals). Board makes decisions - you can participate. Special assessments possible for major repairs. Review documents carefully before buying. Can be restrictive but maintain property values."),
    ("How does home security work?", "Options: DIY systems (Ring, SimpliSafe - $10-30/month), professional monitoring (ADT, Vivint - $40-60/month), no monitoring. Components: door/window sensors, motion detectors, cameras, smart locks, video doorbells. Smart home integration common. Police response varies by area. Many burglars deterred by visible security. Basics: good locks, exterior lighting, not advertising absence (hold mail, use timers). Neighborhood watch programs in some areas."),
    ("What should I know about property taxes?", "Property taxes fund local services (schools, fire, police). Assessed value determined by county - may differ from market value. Tax rate varies widely by location (0.5% to 2.5%+ of value). Can protest assessment if too high. Homestead exemptions reduce taxes for primary residence. Taxes typically escrowed in mortgage payment. Due dates vary (annually, semi-annually, quarterly). Delinquent taxes result in liens, eventually foreclosure."),

    # Legal and Safety
    ("How do I protect myself from scams?", "Common scams: IRS calls (IRS uses mail), tech support calls, romance scams, investment scams, grandparent scams. Red flags: urgency, unusual payment methods (gift cards, wire transfer, crypto), unsolicited contact, too good to be true. Never give personal info to incoming calls. Verify independently. Check FTC and BBB for known scams. If victimized: report to FTC (reportfraud.ftc.gov), contact bank immediately. Elderly particularly targeted."),
    ("What is identity theft and how do I prevent it?", "Identity theft: someone uses your personal info (SSN, accounts) for fraud. Prevention: protect SSN, shred documents, strong unique passwords, monitor credit, freeze credit (free at three bureaus), secure mail. Signs: unexpected bills, collections calls, credit changes. If victimized: file FTC report at IdentityTheft.gov, file police report, place fraud alerts, dispute fraudulent accounts. Consider identity theft protection services."),
    ("What are my rights when interacting with police?", "You have right to: remain silent (say 'I invoke my right to remain silent'), attorney before questioning, refuse consent to searches (say 'I don't consent to searches'), record police in public. You must: provide ID if lawfully detained in some states, follow lawful orders. Stay calm, don't argue on scene (contest in court). Keep hands visible. Ask 'Am I free to go?' If detained, ask for lawyer. File complaints for misconduct after the fact."),
    ("What is jury duty?", "Citizens can be summoned for jury duty - a civic obligation. Qualification: US citizen, 18+, resident of jurisdiction, no felony convictions. Summons comes by mail - follow instructions. Selection (voir dire): lawyers question potential jurors. Trial may last hours, days, or weeks. Employers must allow time off (paid or unpaid varies). Hardship exemptions possible. Compensation minimal ($10-50/day typically). Ignoring summons has consequences."),
    ("What should I know about guns in America?", "Second Amendment protects gun rights. Laws vary significantly by state. Federal: background checks for licensed dealer sales, prohibited for felons/domestic abusers/mentally adjudicated. States vary on: permits, concealed carry, open carry, assault weapons, magazine limits, waiting periods. Roughly 400M guns in US, ~30% of adults own guns. Gun safety basics: treat every gun as loaded, never point at anything you don't intend to destroy, store securely. Debate over gun control is ongoing."),

    # Work and Career
    ("What is work-life balance like in America?", "Compared to other developed countries, Americans work more hours with less vacation. Typical: 40-hour week (often more for salaried/professional). Vacation: average 10-15 days/year (no federal requirement, unlike Europe). Parental leave: no federal paid requirement. Work culture varies by industry and employer. Remote work expanded post-COVID. Burnout is common concern. Some companies now emphasize balance, unlimited PTO (usage varies)."),
    ("How do I negotiate salary in America?", "Research market rates: Glassdoor, LinkedIn Salary, Levels.fyi (tech), BLS. Know your value - skills, experience, achievements. Best time: with offer in hand or annual review. Negotiate base salary first (affects future raises, 401k match). Consider total compensation: bonus, equity, benefits, flexibility. Be professional and positive. Have a number in mind and justify it. Be prepared to walk away. Get final offer in writing."),
    ("What is the gig economy?", "Gig economy: flexible, temporary, or freelance jobs. Platforms: rideshare (Uber, Lyft), delivery (DoorDash, Instacart, Amazon Flex), freelance (Upwork, Fiverr), tasks (TaskRabbit). Workers are independent contractors (1099) - no benefits, pay own taxes (including self-employment tax), no unemployment insurance. Flexibility vs security trade-off. Some states reclassifying workers as employees. Can be primary or supplemental income."),
    ("How do I start a business in America?", "Steps: business plan, choose structure (LLC most common for small business), register with state, get EIN from IRS, business bank account, necessary licenses/permits, insurance. Funding: personal savings, friends/family, small business loans (SBA), investors. Legal considerations: contracts, liability, taxes. Resources: SBA.gov, SCORE (free mentoring). Many businesses fail - research and planning important. Consider starting part-time while employed."),
    ("What professional certifications are valuable?", "Depends on field. IT: AWS certifications, CompTIA (A+, Security+), Cisco (CCNA), PMP (project management). Finance: CPA (accounting), CFA (investment), CFP (financial planning). Healthcare: varies by profession, requires licensing. HR: SHRM, PHR. Real estate: license required, varies by state. Teaching: state certification. Trades: licensing/certification varies. Many certifications require continuing education. Employer may pay for relevant certifications."),

    # Community and Social Life
    ("How do Americans make friends as adults?", "Making friends after school/college is challenging for many. Common ways: work colleagues, neighbors, through children's activities, hobbies and clubs (running groups, book clubs), religious communities, volunteering, alumni networks, apps (Bumble BFF, Meetup). Takes initiative - suggest activities. Friendships often start activity-based. Geographic mobility means starting over. Loneliness is increasingly recognized issue."),
    ("What is volunteerism like in America?", "Strong tradition of volunteering. Opportunities: food banks, Habitat for Humanity, hospitals, schools, animal shelters, mentoring (Big Brothers Big Sisters), environmental cleanup, crisis hotlines. Find opportunities: VolunteerMatch.org, local organizations. Some employers give volunteer time off. Students often need service hours. Benefits: helping community, meeting people, developing skills. Required for some: court-ordered community service."),
    ("What religious life is like in America?", "Freedom of religion is foundational. ~65% identify as Christian (declining), growing 'nones' (no religion). Churches: Catholic, Protestant (Baptist, Methodist, Lutheran, Evangelical, etc.), Orthodox. Other religions: Judaism, Islam, Buddhism, Hinduism, Sikhism, others. Attendance varies widely by region (higher in South). Separation of church and state, though religion influences politics. Houses of worship often community centers offering various services."),
    ("What are American attitudes toward diversity?", "America is increasingly diverse - no majority race by 2045 projected. Legal protections against discrimination (race, religion, sex, national origin, disability). Debates ongoing about: immigration, affirmative action, systemic racism, LGBTQ+ rights. Attitudes vary significantly by region, age, urban/rural. Diversity valued by many employers. DEI (Diversity, Equity, Inclusion) programs common but controversial. Progress has been made but disparities persist."),
    ("How does charity and giving work in America?", "Americans are among most generous donors globally. Ways to give: direct donations, workplace giving, donor-advised funds, volunteering time. Tax benefits: itemized deductions for charitable contributions. Vetting charities: Charity Navigator, GuideStar ratings. Caution: scam charities exist, especially after disasters. Types: religious (largest category), education, human services, health, arts. Giving Tuesday (after Thanksgiving) promotes year-end giving."),

    # Travel and Recreation
    ("How does domestic travel work in the US?", "Air travel: book flights on airline sites or aggregators (Google Flights, Kayak). TSA security - arrive 1.5-2 hours early, liquids 3.4oz or less, ID required (Real ID deadline May 2025). Checked bag fees common. Road trips: Interstate highway system covers country, rest stops along major routes, gas stations everywhere. Train: Amtrak limited but scenic. Bus: Greyhound, FlixBus for budget travel. Rental cars widely available."),
    ("What are popular American vacation destinations?", "Theme parks: Walt Disney World (Orlando), Disneyland (Anaheim), Universal Studios. Beaches: Florida, California, Hawaii. National Parks: Yellowstone, Grand Canyon, Yosemite, Zion, Glacier. Cities: NYC, Las Vegas, San Francisco, Chicago, New Orleans, DC (monuments/museums free). Nature: Alaska cruises, Colorado skiing, Maine coast. Road trips: Route 66, Pacific Coast Highway, Blue Ridge Parkway. Cruises from Florida, California ports."),
    ("How do National Parks work?", "National Park Service operates 400+ sites. Entrance fees: $20-35 per vehicle, good for 7 days. America the Beautiful Pass: $80/year, all parks. Popular parks (Yellowstone, Grand Canyon, Zion) very crowded in summer - reserve campsites months ahead, consider shoulder seasons. Facilities vary: some have lodges, most have camping, all have trails. Leave No Trace principles. Rangers provide programs. State parks are separate system, often less crowded."),
    ("What outdoor recreation is popular in America?", "Hiking: trails everywhere from urban parks to wilderness. Camping: developed campgrounds or backcountry. Fishing: license required (state-specific), regulations vary. Hunting: license required, seasons regulated. Water sports: boating, kayaking, swimming, surfing. Winter: skiing, snowboarding (Rockies, Northeast, Northwest). Biking: road and mountain. Golf: public and private courses. Running races: 5Ks, marathons everywhere."),
    ("How does camping work in America?", "Types: developed campgrounds (bathrooms, sometimes hookups - RVs), primitive camping (minimal facilities), dispersed camping (national forests, BLM land - free, no facilities), backcountry (requires permits in many areas). Reservations: Recreation.gov for federal lands, state systems vary. Equipment: tent, sleeping bag, pad, cooking supplies. Many parks have first-come sites. Campfire rules vary (fire danger). Store food properly (bears). Popular weekends book months ahead."),

    # Money and Finance
    ("How do I create a budget in America?", "Track spending for a month to understand where money goes. Common methods: 50/30/20 (needs/wants/savings), zero-based (every dollar assigned), envelope system (cash in categories). Tools: Mint, YNAB, spreadsheets, pen and paper. Essential categories: housing (aim for under 30% of income), transportation, food, utilities, insurance, debt payments, savings. Build emergency fund (3-6 months expenses). Review and adjust monthly."),
    ("What should I know about investing in America?", "Start with employer 401(k) to get match. Then max IRA ($7,000/year limit). Taxable brokerage for additional investing. Key concepts: diversification, low-cost index funds, long-term thinking, compound growth. Common investments: stocks (individual or funds), bonds, real estate (REITs or property), cash equivalents. Major brokerages: Fidelity, Vanguard, Schwab, Robinhood. Robo-advisors (Betterment, Wealthfront) automate investing. Avoid timing the market."),
    ("What is a Roth IRA vs Traditional IRA?", "Traditional IRA: contributions tax-deductible now, pay taxes on withdrawals in retirement. Roth IRA: contributions with after-tax money, withdrawals tax-free in retirement (including growth). Roth better if: you're young (long time for tax-free growth), expect higher taxes in retirement, want flexibility (contributions can be withdrawn anytime). Income limits for Roth contributions ($161K single, $240K married). Backdoor Roth for high earners."),
    ("How does debt management work?", "Types: credit card (highest interest, 15-25%), auto loan, student loan, mortgage (lowest interest). Strategies: avalanche (pay highest interest first, saves money), snowball (pay smallest first, psychological wins). Minimum payments trap keeps you in debt. Consolidation: combines debts, sometimes at lower rate. Balance transfer: 0% intro rate cards for existing debt. Avoid: payday loans, high-interest personal loans. Credit counseling: nonprofit help negotiating with creditors."),
    ("What is an emergency fund and how do I build one?", "Emergency fund: savings for unexpected expenses (job loss, medical, car repair). Goal: 3-6 months of expenses. Start small: $500-1,000 for starter fund. Where to keep it: high-yield savings account (4-5% APY currently), accessible but not too easy. Build automatically: set up recurring transfer. Fund before aggressive debt payoff (prevents new debt). Don't invest it - stability more important than returns. Use only for true emergencies."),

    # Media and Information
    ("How does American news media work?", "Major TV networks: ABC, CBS, NBC (broadcast), CNN, MSNBC, Fox News (cable). Newspapers: local papers declining, major nationals (NYT, WSJ, Washington Post, USA Today). Online: news sites, social media (problematic for accuracy). AP, Reuters are wire services used by many outlets. Bias exists across spectrum - consume multiple sources. Local news important but struggling financially. Misinformation is significant problem - verify before sharing."),
    ("How do I identify misinformation?", "Red flags: emotional language, no sources cited, unknown website, only exists on social media, too good/bad to be true. Verify: check other reputable sources, reverse image search, check date (old stories recirculated), read beyond headline. Fact-checkers: Snopes, PolitiFact, FactCheck.org. Be especially skeptical during elections, crises. Misinformation spreads faster than corrections. Think before sharing."),
    ("What is freedom of speech in America?", "First Amendment protects speech from government restriction. It does NOT protect: true threats, incitement to imminent lawless action, defamation, fraud, obscenity, child pornography. Private companies (social media, employers) can restrict speech - not a First Amendment issue. Hate speech is generally protected unless it crosses into threats. Academic freedom protects professors. Media has strong protections (actual malice standard for public figures)."),

    # Miscellaneous American Life
    ("What should I know about moving to a new city?", "Before moving: research neighborhoods (cost, safety, commute, schools), visit if possible, understand job market. The move: hire movers or DIY, budget $1,000-5,000+ depending on distance. After: update address (USPS, DMV, voter registration, banks), transfer utilities, find new doctors/dentist, get to know area. Making connections: explore neighborhoods, join activities, be patient - takes 6-12 months to feel settled."),
    ("How do I dispose of things in America?", "Regular trash: weekly curbside pickup (included in taxes or billed). Recycling: varies by location - paper, plastic, glass, metal common (check local rules, contamination is problem). Bulk items: scheduled pickup or drop-off. Hazardous waste (paint, chemicals, batteries): special collection days. Electronics: many stores offer recycling. Donations: Goodwill, Salvation Army, Habitat ReStore. Selling: Facebook Marketplace, Craigslist, OfferUp. Junk removal services for large cleanouts."),
    ("What are common home improvement stores?", "Big boxes: Home Depot and Lowe's (similar, compete directly). Offer: lumber, tools, appliances, paint, flooring, plumbing, electrical, garden. Services: installation, rental tools, how-to workshops. Specialty: Menards (Midwest), Ace Hardware (smaller, more service), Harbor Freight (tools, lower quality/price). Pro desks for contractors. Price matching common. Both have apps, delivery. DIY culture strong - YouTube for instructions."),
    ("What should new Americans understand about US culture?", "Individualism valued - personal responsibility, self-reliance. Direct communication style (less formal than some cultures). Time is money - punctuality matters. Small talk is social lubricant (weather, sports). Personal space larger than some cultures. Tipping is expected and essential to service workers' income. Diversity is reality but integration varies. Regional differences significant. Work culture can be demanding. Optimism and 'can-do' attitude common."),
    ("How does the American Dream work in practice?", "Traditional American Dream: work hard, achieve success (home, family, prosperity). Historically meant upward mobility across generations. Reality today: more complex. Costs (education, healthcare, housing) have risen faster than wages. Social mobility varies by location and starting point. Some achieve it; many struggle. Definition evolving - may include work-life balance, experiences, purpose. Debate about whether Dream is still attainable."),

    # ==========================================================================
    # SPECIALIZED PROFESSIONAL FIELDS
    # ==========================================================================

    # Signals Intelligence (SIGINT)
    ("What is SIGINT (Signals Intelligence)?", "SIGINT is intelligence gathered from electronic signals and communications. It includes: COMINT (Communications Intelligence) - intercepting communications between people. ELINT (Electronic Intelligence) - gathering data from electronic sensors like radar. FISINT (Foreign Instrumentation Signals Intelligence) - intercepting telemetry and other technical signals. NSA is the primary US SIGINT agency. SIGINT is crucial for national security, counterterrorism, and military operations."),
    ("How does signals intelligence collection work?", "SIGINT collection involves: antenna systems and receivers to intercept radio/satellite/cellular signals, cable taps for wired communications, close-access operations for hard-to-reach targets. Processing includes: signal detection and isolation, decryption of encrypted communications, translation of foreign languages, traffic analysis (who communicates with whom, when, how often). Modern challenges include encryption, volume of data, and distinguishing relevant signals from noise."),
    ("What is traffic analysis in SIGINT?", "Traffic analysis examines communication patterns without reading content. It reveals: who communicates with whom (contact chaining), communication frequency and timing, location of transmitters, organizational structures and hierarchies, changes in activity indicating operations. Even encrypted communications reveal metadata. Traffic analysis can identify targets, predict activities, and map networks. It's valuable when content is unavailable or encrypted."),
    ("What is the difference between SIGINT and HUMINT?", "SIGINT (Signals Intelligence) collects from electronic emissions - communications, radar, telemetry. Strengths: covers vast areas, provides exact wording, hard to detect collection. Weaknesses: can be encrypted, doesn't reveal intentions. HUMINT (Human Intelligence) comes from human sources - agents, informants, interrogations. Strengths: reveals intentions/plans, context, insider access. Weaknesses: small scale, can be deceptive, risky. Best intelligence combines both."),
    ("What is communications security (COMSEC)?", "COMSEC protects communications from interception. Elements: cryptographic security (encryption algorithms, key management), transmission security (anti-jam, low probability of intercept), emission security (controlling unintentional signals), physical security of equipment. Best practices: end-to-end encryption, secure key exchange, operational security (not revealing sensitive info), regular key changes, secure facilities. COMSEC is essential for military, government, and sensitive business communications."),
    ("What are the legal frameworks for SIGINT in the US?", "US SIGINT is governed by: Executive Order 12333 (intelligence activities framework), FISA (Foreign Intelligence Surveillance Act) - requires court approval for domestic surveillance, Fourth Amendment protections for US persons, Minimization procedures to protect inadvertently collected US person data, Congressional oversight through intelligence committees. Foreign collection has different rules than domestic. FISA Court (FISC) reviews warrant applications. Reforms followed Snowden revelations."),

    # Computer Network Operations (CNO)
    ("What are Computer Network Operations (CNO)?", "CNO encompasses three disciplines: CNA (Computer Network Attack) - disrupting, denying, degrading, or destroying information in computers/networks. CND (Computer Network Defense) - protecting own networks from attack. CNE (Computer Network Exploitation) - gathering intelligence from target computer systems. These capabilities are used by military, intelligence agencies, and nation-states. Offensive and defensive operations are closely related - understanding attacks improves defense."),
    ("What is Computer Network Exploitation (CNE)?", "CNE involves gaining access to target computer systems to collect intelligence. Phases: reconnaissance (mapping networks, identifying vulnerabilities), initial access (exploiting vulnerabilities, phishing, supply chain), persistence (maintaining access, implants), collection (extracting data, monitoring), exfiltration (removing data covertly). Challenges include: avoiding detection, operating in diverse environments, legal authorities. CNE provides valuable intelligence on adversary capabilities and intentions."),
    ("What is Computer Network Attack (CNA)?", "CNA uses cyber capabilities to disrupt, deny, degrade, or destroy adversary systems. Effects range from: denial of service (temporary disruption), data manipulation (changing information), system destruction (Stuxnet-style physical damage). Considerations: proportionality, collateral damage, attribution, escalation risks. CNA can support military operations, counter terrorism, or respond to attacks. Distinction from crime: state-sponsored, authorized, strategic objectives."),
    ("What is Computer Network Defense (CND)?", "CND protects networks from intrusion and attack. Layers: perimeter defense (firewalls, IDS/IPS), endpoint protection (antivirus, EDR), network monitoring (traffic analysis, SIEM), threat hunting (proactively seeking threats), incident response (containing and remediating breaches). Key concepts: defense in depth, zero trust architecture, threat intelligence, security operations centers (SOC). Effective CND requires people, processes, and technology working together."),
    ("What is an Advanced Persistent Threat (APT)?", "APT is a sophisticated, long-term cyber intrusion typically by nation-states or well-resourced groups. Characteristics: advanced techniques (zero-days, custom malware), persistent presence (months to years), specific targets (government, defense, critical infrastructure), patient and methodical approach. APT groups are tracked by security researchers (APT28/Fancy Bear = Russia, APT1 = China, Lazarus = North Korea). Defense requires continuous monitoring, threat intelligence, and incident response capability."),
    ("What are cyber operations tools and techniques?", "Offensive tools: exploitation frameworks (Metasploit, Cobalt Strike), custom malware and implants, zero-day exploits, social engineering tools. Defensive tools: firewalls, IDS/IPS (Snort, Suricata), SIEM (Splunk, Elastic), EDR (CrowdStrike, Carbon Black), vulnerability scanners (Nessus, Qualys). Techniques follow kill chain: reconnaissance, weaponization, delivery, exploitation, installation, command and control, actions on objectives. Both sides constantly evolve."),
    ("What is the Cyber Kill Chain?", "The Cyber Kill Chain (Lockheed Martin model) describes attack phases: 1) Reconnaissance - researching target. 2) Weaponization - creating exploit/payload. 3) Delivery - transmitting weapon (phishing, USB). 4) Exploitation - triggering vulnerability. 5) Installation - establishing persistence. 6) Command & Control - remote control channel. 7) Actions on Objectives - achieving goals. Defenders can break the chain at any phase. Earlier detection means less damage. Model helps structure both offense and defense."),
    ("What is OPSEC in cyber operations?", "Operational Security (OPSEC) protects operations from adversary detection. In cyber: using anonymizing infrastructure (VPNs, Tor, bulletproof hosting), avoiding patterns (timing, techniques), sanitizing tools (no identifying artifacts), separation of operations, secure communications. Poor OPSEC leads to attribution - linking operations to specific actors. Nation-states invest heavily in OPSEC. Defenders study adversary OPSEC failures to identify threat actors."),

    # Criminal Investigations
    ("How do criminal investigations work in America?", "Criminal investigations follow general phases: initial report/complaint, preliminary investigation (first responders secure scene, gather initial evidence), follow-up investigation (detectives conduct interviews, collect evidence, identify suspects), case preparation (work with prosecutors), arrest and prosecution. Key elements: establishing corpus delicti (crime occurred), identifying and locating suspects, gathering admissible evidence, maintaining chain of custody. Investigations must respect constitutional rights."),
    ("What is the role of forensic evidence in investigations?", "Forensic evidence scientifically analyzes physical evidence. Types: DNA (biological samples, CODIS database), fingerprints (AFIS database), ballistics (matching bullets/casings to weapons), trace evidence (fibers, hair, soil), toxicology (drugs, poisons), digital forensics (computers, phones). Evidence must be properly collected, documented, and preserved. Chain of custody is critical for admissibility. Expert witnesses explain forensic findings in court. Forensics can identify, include, or exclude suspects."),
    ("What is digital forensics?", "Digital forensics recovers and analyzes electronic evidence. Process: identification (locate evidence), preservation (forensic imaging, write blockers), analysis (examine data, recover deleted files, timeline analysis), documentation, presentation. Tools: EnCase, FTK, Autopsy, Cellebrite (mobile). Evidence sources: computers, phones, cloud accounts, network logs, IoT devices. Challenges: encryption, anti-forensics, volume of data, volatile memory. Used in criminal, civil, corporate, and national security investigations."),
    ("What is the chain of custody?", "Chain of custody documents everyone who handled evidence from collection to court. Requirements: unique identification (evidence tags), documentation of transfers (who, when, why), secure storage, limited access. Breaks in chain can make evidence inadmissible - defense argues tampering or contamination possible. Documentation includes: where found, who collected, how packaged, storage location, analysis records. Critical for forensic evidence, drugs, weapons, and all physical evidence."),
    ("How do police interrogations work?", "Interrogations are custodial questioning of suspects. Requirements: Miranda warnings, voluntary statements, no coercion. Techniques: rapport building, cognitive interviewing, Reid Technique (controversial - can produce false confessions), PEACE model (non-confrontational). Recording increasingly required. Suspect can invoke rights (silence, attorney) at any time - questioning must stop. Confessions must be corroborated. Defense attorneys challenge confession voluntariness. Effective interrogation is about truth-finding, not confession-getting."),
    ("What is the difference between investigation and prosecution?", "Investigation (law enforcement): gather evidence, identify suspects, make arrests. Goal is determining what happened and who is responsible. Prosecution (district/US attorneys): review evidence, file charges, present case in court, seek conviction. Prosecutors decide whether evidence supports charges and case is winnable. They have ethical obligation to seek justice, not just convictions. Close coordination needed, but different roles. Defense provides adversarial check on both."),
    ("What are federal vs state criminal investigations?", "State/local police investigate: most crimes (assault, robbery, burglary, murder), state law violations. Federal agencies investigate: federal crimes (bank robbery, counterfeiting, tax evasion), interstate crimes, crimes on federal property, violations of federal law. Key federal agencies: FBI (broad jurisdiction), DEA (drugs), ATF (firearms, explosives), Secret Service (counterfeiting, protection), HSI (immigration, customs). Joint task forces combine resources. Some crimes can be prosecuted federally or by states."),
    ("What is undercover investigation?", "Undercover operations involve officers concealing identity to investigate crimes. Used for: drug trafficking, organized crime, corruption, terrorism. Legal considerations: entrapment (government can't induce crime person wouldn't otherwise commit), authorization requirements, danger to officers. Types: deep cover (long-term, complete identity change), light cover (short-term, limited scope). Informants provide similar access without law enforcement status. Strict policies govern undercover operations due to risks."),
    ("What is surveillance in criminal investigations?", "Surveillance monitors suspect activities. Physical surveillance: following subjects (mobile), observing locations (static). Technical surveillance: wiretaps (Title III warrant required), GPS tracking (warrant required per Jones decision), pole cameras, pen registers (phone numbers dialed). Challenges: resources, detection by targets, constitutional limits. Surveillance provides evidence, identifies associates, and reveals patterns. Extensive documentation required. Electronic surveillance has strict warrant requirements protecting privacy."),
    ("How do search warrants work?", "Search warrants authorize law enforcement to search specific locations for specific evidence. Requirements: probable cause (reasonable belief evidence will be found), particularity (specific place and items), neutral magistrate approval. Process: officer submits affidavit, judge reviews and signs if warranted. Exceptions to warrant requirement: consent, exigent circumstances, plain view, search incident to arrest, automobile exception. Knock and announce typically required. Evidence from invalid warrants may be suppressed (exclusionary rule)."),

    # Intelligence Community
    ("What is the US Intelligence Community?", "The IC consists of 18 agencies: CIA (foreign intelligence), NSA (SIGINT, cybersecurity), DIA (military intelligence), NGA (geospatial), NRO (reconnaissance satellites), FBI (domestic intelligence/counterintelligence), plus military service intelligence, DEA, Treasury, DHS components, and others. DNI (Director of National Intelligence) coordinates. Each agency has specific focus. Intelligence cycle: planning, collection, processing, analysis, dissemination. IC supports policymakers and military with decision-making information."),
    ("What is the CIA and what does it do?", "CIA (Central Intelligence Agency) is the primary foreign intelligence agency. Functions: collecting foreign intelligence through human sources (HUMINT), analyzing all-source intelligence, conducting covert action when directed by President. Organized into: Directorate of Operations (collection), Directorate of Analysis, Directorate of Science & Technology, Directorate of Support. No domestic law enforcement authority. Headquarters in Langley, Virginia. Reports to DNI and President."),
    ("What is the NSA and what does it do?", "NSA (National Security Agency) focuses on SIGINT and cybersecurity. Missions: collecting signals intelligence on foreign targets, protecting US government communications and systems (cybersecurity). Part of DoD but serves entire IC. Headquarters at Fort Meade, Maryland. Works with Five Eyes partners (UK, Canada, Australia, New Zealand). CYBERCOM (Cyber Command) co-located, shares director. Largest intelligence agency by budget and personnel."),
    ("What is the FBI's intelligence role?", "FBI has dual roles: law enforcement and domestic intelligence. Intelligence functions: counterterrorism, counterintelligence (foreign spies), cyber threats. After 9/11, FBI created intelligence branch and Field Intelligence Groups. Works with IC on domestic threats while respecting civil liberties. FBI is only IC agency that can arrest people and collect evidence for prosecution. Balances intelligence collection with law enforcement evidence standards. Joint Terrorism Task Forces combine FBI with other agencies."),
    ("What is counterintelligence?", "Counterintelligence (CI) identifies, assesses, and counters foreign intelligence threats. Activities: detecting foreign spies, protecting classified information, investigating espionage, security education. Offensive CI: penetrating foreign services, turning their agents. FBI leads domestic CI; CIA handles foreign. Insider threats are major concern. CI integrates human and technical security. Recent priorities include: China's intelligence activities, Russia operations, cyber espionage. CI protects all IC and defense activities."),
    ("What is the intelligence cycle?", "The intelligence cycle is the process of producing intelligence: 1) Planning/Direction - policymakers identify needs. 2) Collection - gathering information (HUMINT, SIGINT, GEOINT, OSINT). 3) Processing - converting raw data to usable form (decryption, translation). 4) Analysis - evaluating and interpreting information, producing assessments. 5) Dissemination - delivering to consumers. 6) Feedback - users provide guidance for future collection. Cycle is continuous. Good intelligence is timely, accurate, relevant, and actionable."),
    ("What security clearances exist in the US?", "Federal security clearances: Confidential (lowest, damage to national security), Secret (serious damage), Top Secret (grave damage). TS requires Single Scope Background Investigation (SSBI) covering 10 years. Above TS: SCI (Sensitive Compartmented Information) for intelligence, SAP (Special Access Programs) for sensitive projects. Clearance process: SF-86 form, background investigation, adjudication. Factors: foreign contacts, finances, criminal history, drug use, mental health. Polygraph required for some positions."),
    ("How do you get a job in US intelligence?", "Intelligence careers require US citizenship and ability to obtain clearance. Paths: direct hire (agency websites), military intelligence (then civilian transition), contractors (easier entry, less job security). Education: any degree useful, especially languages, STEM, international relations. Skills valued: analytical thinking, writing, languages, technical skills, area expertise. Process: application, testing, interviews, polygraph, full background investigation (6-18 months). Agencies recruit at universities. Prior military/government experience helpful."),

    # Law Enforcement Careers
    ("How do you become a police officer in America?", "Requirements vary by department but typically: US citizen, 21+ years old (some 18 for academy), high school diploma (many require some college), driver's license, no felony convictions. Process: written test, physical fitness test, background investigation, polygraph, psychological evaluation, medical exam, oral board interview, academy training (4-6 months). Field training follows (3-6 months). Career progression: patrol → specialized units → detective → supervisor → command. Federal positions require degree."),
    ("What are federal law enforcement agencies?", "Major federal agencies: FBI (broad criminal jurisdiction, counterterrorism), DEA (drug enforcement), ATF (firearms, explosives, arson), US Marshals (fugitives, court security, witness protection), Secret Service (counterfeiting, presidential protection), CBP (border protection), ICE/HSI (immigration enforcement, investigations), Postal Inspection Service. Requirements: typically degree, age limits, extensive background check, academy training. Most are under DHS or DOJ. Each has specific jurisdiction."),
    ("What does the FBI do?", "FBI is the primary federal investigative agency with broad jurisdiction: terrorism, counterintelligence, cybercrime, public corruption, civil rights, organized crime, white-collar crime, violent crime. Also provides services: forensic labs, NCIC database, training at Quantico. 56 field offices nationwide. Agents are Special Agents (investigators) or Intelligence Analysts. Requirements: degree, 23-36 age range, rigorous background check, 20-week training at Quantico. About 35,000 employees."),
    ("What is the Secret Service?", "Secret Service has dual missions: protection (President, VP, families, former presidents, major candidates, visiting heads of state) and investigations (counterfeiting, financial crimes, cybercrime). Originally part of Treasury, now under DHS. Uniformed Division protects White House, VP residence, foreign embassies. Special Agents handle both missions throughout career. Training at FLETC and Secret Service academy. Known for protective operations but investigation is larger part of work."),
    ("What is the US Marshals Service?", "US Marshals Service is the oldest federal law enforcement agency. Missions: fugitive operations (arresting federal fugitives, including 15 Most Wanted), court security and prisoner transport, witness security (WITSEC) program, asset forfeiture, Special Operations Group (high-risk operations). Marshals work in judicial districts. Deputy US Marshals are the operational force. Known for hunting fugitives - arrest most federal fugitives. Work closely with state/local on fugitive task forces."),
    ("What is a detective/investigator role?", "Detectives/investigators specialize in solving specific crime types: homicide, robbery, burglary, sex crimes, financial crimes, cyber crimes. Promoted from patrol or hired directly (federal). Duties: reviewing cases, interviewing witnesses/suspects, collecting and analyzing evidence, coordinating with forensics, writing reports, working with prosecutors, testifying in court. Require strong analytical, communication, and interpersonal skills. Caseloads can be heavy. Solve rate varies by crime type."),

    # Cybersecurity Careers
    ("What cybersecurity careers exist in America?", "Major roles: Security Analyst (monitor, detect threats), Penetration Tester (ethical hacking), Security Engineer (build security systems), Incident Responder (handle breaches), Security Architect (design security), CISO (executive leadership), Forensics Analyst, Malware Analyst, Threat Intelligence Analyst, GRC Analyst (governance, risk, compliance). Employers: companies (all industries), consulting firms, government, military, MSPs. Strong demand, good salaries. Entry through IT, certifications, or degree programs."),
    ("What certifications are important for cybersecurity?", "Entry-level: CompTIA Security+, CySA+ (analyst), Network+. Intermediate: CEH (Certified Ethical Hacker), GSEC (GIAC Security Essentials). Advanced: CISSP (management, requires experience), OSCP (penetration testing, hands-on), GPEN (penetration testing), CISM (management). Specialized: GCIH (incident handling), GREM (malware), AWS/Azure security certs. Certifications demonstrate knowledge but experience matters most. Many require continuing education. Employers often pay for certifications."),
    ("How do I start a career in cybersecurity?", "Paths: IT experience (help desk, system admin) → security roles. Degree in cybersecurity, CS, or IT. Self-study with home labs, CTF competitions, certifications. Boot camps (accelerated training). Military (trains many security professionals). Entry roles: SOC analyst, security administrator, IT with security responsibilities. Build skills: networking, operating systems, scripting, security tools. Get Security+ certification. Join local security groups (BSides, OWASP). Create portfolio with projects and writeups."),
    ("What is penetration testing as a career?", "Penetration testers (ethical hackers) attempt to breach systems to find vulnerabilities before attackers do. Types: external network, internal network, web application, mobile, social engineering, physical, red team (adversary simulation). Skills needed: networking, web technologies, programming/scripting, common vulnerabilities, tools (Burp, Metasploit, Nmap). Career path: IT → security analyst → junior pentester → senior pentester → lead/principal → consultant. OSCP certification highly valued. Consulting or internal positions available."),
    ("What is incident response as a career?", "Incident responders handle security breaches. Activities: detection (monitoring alerts), triage (assess severity), containment (stop spread), eradication (remove threat), recovery (restore operations), lessons learned (improve defenses). Work in SOCs, IR teams, or consulting (DFIR). Skills: forensics, malware analysis, log analysis, network analysis, communication. Stressful during incidents. Path: SOC analyst → IR analyst → senior IR → IR lead → manager. GCIH, GCFA certifications valuable. Often on-call rotation."),
    ("What is threat intelligence?", "Threat intelligence analyzes cyber threats to inform defenses. Levels: tactical (IOCs - indicators of compromise), operational (TTPs - tactics, techniques, procedures), strategic (threat landscape, trends). Activities: monitoring threat actors, analyzing malware and campaigns, producing reports, integrating with security tools. Employers: large enterprises, government, vendors, consulting. Skills: analytical thinking, research, writing, technical security knowledge, OSINT. Certifications: CTIA, GCTI. Combines technical and analytical work."),

    # Military Careers
    ("What military branches exist in the US?", "Six branches: Army (land warfare, largest branch), Navy (sea warfare, carrier operations), Marine Corps (amphibious, expeditionary), Air Force (air and space operations), Space Force (space operations, newest branch), Coast Guard (maritime law enforcement, homeland security, under DHS in peacetime). Each has active duty, reserve, and guard (Army/Air) components. Different cultures, missions, and career opportunities. Unified under Department of Defense (except Coast Guard)."),
    ("How do you join the US military?", "Enlisted: age 17-39 (varies by branch), high school diploma/GED, pass ASVAB (aptitude test), medical exam (MEPS), choose job/MOS based on scores and availability, basic training (8-13 weeks), job training. Officers: bachelor's degree required, commission through ROTC, service academies, or OCS/OTS after college. Requirements: US citizen or permanent resident, meet fitness standards, no disqualifying medical/legal issues. Recruiters guide the process."),
    ("What is military intelligence?", "Military intelligence supports commanders with information on adversaries and battlespace. Disciplines: HUMINT, SIGINT, GEOINT (geospatial), MASINT (measurement and signature), OSINT. Each service has intel components; DIA coordinates defense intelligence. Roles: analyst (produce assessments), collector (gather information), counterintelligence. Clearance required. Career paths: enlisted specialists or officers. Training at intelligence schools (Fort Huachuca for Army). Many transition to civilian intel careers."),
    ("What is Special Operations?", "Special Operations Forces (SOF) conduct specialized, high-risk missions: direct action, special reconnaissance, counterterrorism, foreign internal defense, unconventional warfare. Units: Army Special Forces (Green Berets), Rangers, Delta Force; Navy SEALs; Air Force PJs, Combat Controllers, Special Tactics; Marine Raiders. Selection is extremely demanding (high attrition). Requires top physical fitness, mental toughness, teamwork. Career path: serve in conventional forces first, then attempt selection."),
    ("What cyber roles exist in the US military?", "Military cyber roles: Cyber Operations (offensive/defensive operations), Cyber Intelligence (threat analysis), Network Operations (maintaining networks), Information Warfare. Organizations: US Cyber Command, service cyber components. Enlisted roles do technical work; officers lead. Training: initial military training plus cyber schooling. Clearance required. 17C (Army Cyber Operations Specialist), CTN (Navy Cryptologic Technician Networks) are key enlisted jobs. Growing field with civilian equivalence for skills."),

    # Emergency Services
    ("How does the fire service work in America?", "Fire departments handle fires, medical emergencies (most calls), hazmat, rescue, and disasters. Types: career (paid), volunteer (common in rural areas), combination. Ranks: firefighter → engineer/driver → lieutenant → captain → battalion chief → chief. Hiring: written test, physical agility test, interview, background check, medical, academy training. Most departments require EMT certification. Work schedule often 24 hours on, 48-72 off. Unions common in career departments."),
    ("How does EMS work in America?", "Emergency Medical Services provides pre-hospital care. Levels: EMR (Emergency Medical Responder), EMT (Basic Life Support), AEMT, Paramedic (Advanced Life Support). Delivery models: fire-based (most common), private ambulance, hospital-based, third service (government EMS). 911 dispatches appropriate resources. Protocols guide treatment. Transport to hospitals. Career path: EMT certification (few months), paramedic program (1-2 years), supervisory roles. Field is demanding physically and emotionally."),
    ("What is emergency management?", "Emergency management coordinates disaster preparedness, response, recovery, and mitigation. Framework: FEMA (federal), state emergency management agencies, local emergency management. Phases: mitigation (reduce risk), preparedness (plan and train), response (during disaster), recovery (return to normal). Incident Command System (ICS) standardizes response. Careers: emergency managers, planners, coordinators. Degree programs available. FEMA provides training (Emergency Management Institute). Coordinates all hazards: natural disasters, terrorism, accidents."),
    ("How does 911 dispatch work?", "911 dispatchers (telecommunicators) answer emergency calls, gather information, dispatch appropriate resources (police, fire, EMS). Skills: remaining calm, multitasking, typing while talking, making quick decisions. Training: CPR, agency protocols, CAD (computer-aided dispatch) systems, radio procedures. Stressful job - hearing emergencies, criticism if mistakes. Career path: dispatcher → senior dispatcher → supervisor → communications manager. Law enforcement or fire experience helpful. Some agencies require testing and backgrounds."),

    # Legal Careers
    ("What types of lawyers exist in America?", "Major practice areas: criminal (prosecutors defend society, defense attorneys protect accused), civil litigation (lawsuits between parties), corporate/business (transactions, compliance), family (divorce, custody), personal injury (plaintiff or defense), real estate, immigration, intellectual property, tax, employment, environmental. Lawyers may be in: law firms, solo practice, corporate in-house, government, public interest. Specialization comes after law school through practice."),
    ("How do you become a lawyer in America?", "Path: bachelor's degree (any major), LSAT (entrance exam), law school (3 years, JD degree), bar exam (state-specific), character and fitness review. Law school first year: core courses (contracts, torts, civil procedure, criminal law). Upper years: electives and clinics. Summer associate positions at firms are pathway to hiring. Bar exam is 2-day test (MBE and state-specific). Some states allow diploma privilege (no bar exam from in-state schools). Continuing legal education required."),
    ("What do prosecutors do?", "Prosecutors (District Attorneys for state, US Attorneys for federal) represent the government in criminal cases. Duties: review police cases, decide whether to file charges, present cases at trial, recommend sentences, handle appeals. Ethical duty: seek justice, not just convictions (must disclose exculpatory evidence). Career: many start in DA office for trial experience. Heavy caseloads, lower pay than private practice, but meaningful work. Path: law school → bar → assistant DA → senior positions."),
    ("What do criminal defense attorneys do?", "Defense attorneys protect the accused. Public defenders represent those who can't afford lawyers (appointed by court, heavy caseloads). Private defense attorneys charge clients. Duties: advise clients on rights and options, investigate cases, negotiate plea deals, try cases, handle appeals. Constitutional role: ensure fair process, hold government accountable. Challenging: unpopular clients, emotional cases, workload. Most cases plead out. Trial skills crucial for negotiating leverage."),
    ("What is the court system structure?", "Federal courts: District Courts (trial level, 94 districts), Circuit Courts of Appeals (13 circuits), Supreme Court (final authority). State courts: trial courts (may be divided by type - criminal, civil, family), appellate courts, state supreme courts. Federal handles: federal crimes, constitutional issues, suits between states, diversity jurisdiction. State handles: most criminal cases, family law, contracts, torts. Cases can move from state to federal on federal questions."),

    # Healthcare Careers
    ("How do you become a doctor in America?", "Path: bachelor's degree with pre-med courses (biology, chemistry, physics, organic chemistry), MCAT (entrance exam), medical school (4 years, MD or DO degree), residency (3-7 years depending on specialty), optional fellowship for subspecialties. Total: 11-18 years after high school. Extremely competitive: high GPA and MCAT needed. Match Day assigns residencies. Board exams during and after training. Licensing by state. Continuing education required. Enormous debt common ($200,000+)."),
    ("What types of nurses exist?", "Nursing levels: CNA (Certified Nursing Assistant - basic care), LPN/LVN (Licensed Practical Nurse - 1 year program), RN (Registered Nurse - associate's or bachelor's), BSN (Bachelor's of Science in Nursing - 4 years), NP (Nurse Practitioner - master's or doctorate, can prescribe). Specialties: ICU, ER, OR, pediatrics, oncology, many others. Career progression through education and experience. Strong demand, good salaries. Shift work typically 12-hour shifts."),
    ("What is a paramedic vs EMT?", "EMT (Emergency Medical Technician): Basic Life Support - CPR, basic airway, splinting, automated external defibrillator, assisting with certain medications. Training: ~150 hours. Paramedic: Advanced Life Support - IV therapy, cardiac monitoring, advanced airways, medication administration, interpreting EKGs. Training: 1,200-1,800 hours after EMT. Paramedics make more critical decisions, have more interventions available. Both transport patients. Paramedics typically in ALS ambulances; EMTs in BLS or as paramedic partners."),
    ("What is a physician assistant?", "Physician Assistants (PAs) practice medicine under physician supervision. Education: bachelor's degree (often with healthcare experience), PA program (3 years, master's degree). Can: examine patients, diagnose, treat, prescribe medications, assist in surgery. Work in all specialties. Supervision ranges from direct to indirect depending on state/setting. Growing field with good salaries and work-life balance. Similar scope to Nurse Practitioners but different training path. Collaborative practice model."),

    # Technical Trades
    ("What are skilled trades careers?", "Skilled trades: electricians, plumbers, HVAC technicians, welders, carpenters, machinists, automotive technicians, lineworkers, and more. Training through: apprenticeships (paid learning, 4-5 years), vocational schools, community colleges, military. Apprenticeships combine on-the-job training with classroom. Licensing required for many (electricians, plumbers). Strong demand, good pay ($50,000-100,000+), job security. Physical work, can be dangerous. Union or non-union shops."),
    ("How do apprenticeships work in America?", "Apprenticeships are earn-while-you-learn programs in skilled trades. Structure: work for employer at increasing wages while completing related instruction. Duration: 1-6 years (typically 4). Registered with Department of Labor or state. Apply through: union halls (IBEW, UA), employer programs, apprenticeship.gov. Highly competitive for union programs. Complete apprenticeship → journeyman status. Benefits: no student debt, salary during training, valuable credential. Common in construction, manufacturing, utilities."),
    ("What does an electrician do?", "Electricians install, maintain, and repair electrical systems. Types: residential (homes), commercial (businesses), industrial (factories), lineworkers (power lines). Path: apprenticeship (4-5 years) or vocational training, then licensing exam. License types vary by state: journeyman, master. Duties: reading blueprints, installing wiring, troubleshooting, code compliance. Safety critical - working with electricity is dangerous. Growing demand with construction and electric vehicles. Union (IBEW) or non-union."),
    ("What do HVAC technicians do?", "HVAC (Heating, Ventilation, Air Conditioning) technicians install, maintain, and repair climate control systems. Training: vocational programs (6 months - 2 years) or apprenticeships. EPA certification required for refrigerant handling. Duties: installing units, diagnosing problems, repairing equipment, performing maintenance. Work in residential, commercial, industrial. Strong demand especially in summer/winter. Career progression: technician → senior technician → supervisor → contractor. Physical work in uncomfortable conditions."),
]

# =============================================================================
# REASONING AND LOGIC
# =============================================================================
REASONING = [
    # Logical puzzles
    ("If all dogs are mammals and all mammals are animals, what can we conclude about dogs?", "We can conclude that all dogs are animals. This is a syllogism: if A (dogs) is a subset of B (mammals), and B is a subset of C (animals), then A must be a subset of C."),
    ("There are three switches outside a room. One controls a light inside. You can only enter once. How do you determine which switch controls the light?", "Turn switch 1 ON for 10 minutes, then turn it OFF. Turn switch 2 ON. Enter the room. If the light is ON, it's switch 2. If OFF and warm, it's switch 1. If OFF and cold, it's switch 3."),
    ("You have 8 balls. One is heavier. With a balance scale, what's the minimum weighings to find it?", "Two weighings. Divide into groups of 3, 3, and 2. Weigh the two groups of 3. If balanced, weigh the remaining 2 to find the heavier one. If unbalanced, take the heavier group of 3, weigh any 2 against each other. If balanced, it's the third; if not, it's the heavier one."),

    # Critical thinking
    ("What is correlation vs causation?", "Correlation means two things tend to occur together. Causation means one actually causes the other. Correlation doesn't imply causation. For example, ice cream sales and drowning deaths both increase in summer, but ice cream doesn't cause drowning—hot weather is the common cause."),
    ("What is confirmation bias?", "Confirmation bias is the tendency to search for, interpret, and remember information that confirms existing beliefs while ignoring contradicting evidence. To counter it, actively seek opposing viewpoints, consider alternative explanations, and evaluate evidence objectively."),
    ("How do you evaluate the credibility of a source?", "Evaluate credibility by checking: 1) Authority - who wrote it and their expertise, 2) Accuracy - are claims supported by evidence, 3) Purpose - is it informing or persuading, 4) Currency - is it up-to-date, 5) Corroboration - do other reliable sources agree."),

    # Problem-solving
    ("How do you approach a complex problem?", "Break it down: 1) Understand the problem fully, 2) Identify what you know and don't know, 3) Break it into smaller sub-problems, 4) Solve each sub-problem, 5) Combine solutions, 6) Verify the result works. If stuck, try a different approach or simplify the problem."),
    ("What is the scientific method?", "The scientific method is: 1) Observe and question, 2) Form a hypothesis (testable prediction), 3) Design and conduct experiments, 4) Analyze data, 5) Draw conclusions, 6) Share results for peer review. If results don't support the hypothesis, revise and retest."),
    ("How do you make good decisions?", "Good decision-making involves: 1) Define the decision clearly, 2) Gather relevant information, 3) Identify alternatives, 4) Weigh pros and cons of each, 5) Consider your values and priorities, 6) Make the decision, 7) Review and learn from outcomes."),
]

# =============================================================================
# CODING AND TECHNICAL
# =============================================================================
CODING = [
    # Python basics
    ("How do I read a file in Python?", "```python\n# Using with statement (recommended)\nwith open('file.txt', 'r') as f:\n    content = f.read()  # Read entire file\n\n# Read line by line\nwith open('file.txt', 'r') as f:\n    for line in f:\n        print(line.strip())\n```"),
    ("How do I handle exceptions in Python?", "```python\ntry:\n    result = risky_operation()\nexcept ValueError as e:\n    print(f'Value error: {e}')\nexcept Exception as e:\n    print(f'Unexpected error: {e}')\nelse:\n    print('Success!')  # Runs if no exception\nfinally:\n    cleanup()  # Always runs\n```"),
    ("What is a list comprehension?", "List comprehension is a concise way to create lists:\n```python\n# Traditional\nsquares = []\nfor x in range(10):\n    squares.append(x**2)\n\n# List comprehension\nsquares = [x**2 for x in range(10)]\n\n# With condition\nevens = [x for x in range(20) if x % 2 == 0]\n```"),
    ("How do I use dictionaries in Python?", "```python\n# Create\nuser = {'name': 'Alice', 'age': 30}\n\n# Access\nname = user['name']  # KeyError if missing\nname = user.get('name', 'Unknown')  # Default value\n\n# Modify\nuser['age'] = 31\nuser['email'] = 'alice@email.com'\n\n# Iterate\nfor key, value in user.items():\n    print(f'{key}: {value}')\n```"),
    ("How do I create a class in Python?", "```python\nclass Dog:\n    def __init__(self, name, breed):\n        self.name = name\n        self.breed = breed\n    \n    def bark(self):\n        return f'{self.name} says woof!'\n    \n    def __str__(self):\n        return f'{self.name} the {self.breed}'\n\n# Usage\nmy_dog = Dog('Buddy', 'Golden Retriever')\nprint(my_dog.bark())\n```"),

    # JavaScript
    ("How do I fetch data in JavaScript?", "```javascript\n// Using fetch with async/await\nasync function getData() {\n    try {\n        const response = await fetch('https://api.example.com/data');\n        if (!response.ok) throw new Error('Network error');\n        const data = await response.json();\n        return data;\n    } catch (error) {\n        console.error('Error:', error);\n    }\n}\n```"),
    ("What is the difference between let, const, and var?", "`var` is function-scoped and can be redeclared. `let` is block-scoped and can be reassigned. `const` is block-scoped and cannot be reassigned (but objects/arrays can be mutated). Best practice: use `const` by default, `let` when you need to reassign, avoid `var`."),
    ("How do arrow functions work in JavaScript?", "```javascript\n// Traditional function\nfunction add(a, b) {\n    return a + b;\n}\n\n// Arrow function\nconst add = (a, b) => a + b;\n\n// With body\nconst greet = name => {\n    const msg = `Hello, ${name}!`;\n    return msg;\n};\n\n// Arrow functions don't bind their own 'this'\n```"),

    # General programming
    ("What is recursion?", "Recursion is when a function calls itself to solve a problem by breaking it into smaller instances. It needs a base case (stopping condition) and a recursive case.\n```python\ndef factorial(n):\n    if n <= 1:  # Base case\n        return 1\n    return n * factorial(n - 1)  # Recursive case\n```"),
    ("What is Big O notation?", "Big O describes algorithm efficiency as input grows. Common complexities:\n- O(1): Constant - array access\n- O(log n): Logarithmic - binary search\n- O(n): Linear - simple loop\n- O(n log n): Linearithmic - efficient sorting\n- O(n²): Quadratic - nested loops\n- O(2^n): Exponential - recursive fibonacci"),
    ("What is the difference between a stack and a queue?", "Stack (LIFO - Last In, First Out): Like a stack of plates. Last item added is first removed. Operations: push (add), pop (remove). Used for: undo operations, recursion.\n\nQueue (FIFO - First In, First Out): Like a line. First item added is first removed. Operations: enqueue (add), dequeue (remove). Used for: task scheduling, BFS."),
    ("What is version control?", "Version control tracks changes to code over time, enabling collaboration and history. Git is the most popular system. Key concepts: repository (project), commit (save point), branch (parallel development), merge (combine branches), pull request (propose changes). Commands: git init, add, commit, push, pull, branch, merge."),
    ("What is an API?", "API (Application Programming Interface) is a set of rules for software to communicate. REST APIs use HTTP methods (GET, POST, PUT, DELETE) to access resources at URLs. APIs return data (often JSON). Example: GET /users/123 returns user 123's data. APIs enable apps to share data and functionality."),
]

# =============================================================================
# MATHEMATICS
# =============================================================================
MATH = [
    # Arithmetic
    ("What is 15% of 80?", "15% of 80 = 0.15 × 80 = 12"),
    ("What is 7 × 8?", "7 × 8 = 56"),
    ("What is 144 ÷ 12?", "144 ÷ 12 = 12"),
    ("What is 25² (25 squared)?", "25² = 25 × 25 = 625"),
    ("What is the square root of 169?", "√169 = 13 (since 13 × 13 = 169)"),

    # Algebra
    ("Solve for x: 2x + 5 = 17", "2x + 5 = 17\n2x = 17 - 5\n2x = 12\nx = 6"),
    ("Solve for x: 3x - 7 = 2x + 4", "3x - 7 = 2x + 4\n3x - 2x = 4 + 7\nx = 11"),
    ("Factor: x² - 9", "x² - 9 = (x + 3)(x - 3)\nThis is a difference of squares: a² - b² = (a+b)(a-b)"),
    ("What is the quadratic formula?", "For ax² + bx + c = 0:\nx = (-b ± √(b² - 4ac)) / 2a\n\nThe discriminant (b² - 4ac) determines the number of solutions:\n- Positive: 2 real solutions\n- Zero: 1 real solution\n- Negative: 2 complex solutions"),

    # Concepts
    ("What is a prime number?", "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. First primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29... The number 2 is the only even prime."),
    ("What is the Pythagorean theorem?", "For a right triangle with legs a and b and hypotenuse c:\na² + b² = c²\n\nExample: If a=3 and b=4:\n3² + 4² = 9 + 16 = 25 = 5²\nSo c = 5"),
    ("What is a factorial?", "Factorial (n!) is the product of all positive integers up to n:\nn! = n × (n-1) × (n-2) × ... × 1\n\nExamples:\n5! = 5×4×3×2×1 = 120\n0! = 1 (by definition)"),
    ("What is logarithm?", "A logarithm answers: 'What power gives this number?'\nlog_b(x) = y means b^y = x\n\nExamples:\nlog₂(8) = 3 (because 2³ = 8)\nlog₁₀(100) = 2 (because 10² = 100)\n\nCommon bases: 10 (log), e (ln), 2 (in computing)"),
    ("What is the order of operations?", "PEMDAS (or BODMAS):\n1. Parentheses (Brackets)\n2. Exponents (Orders)\n3. Multiplication and Division (left to right)\n4. Addition and Subtraction (left to right)\n\nExample: 2 + 3 × 4 = 2 + 12 = 14 (not 20)"),
]

# =============================================================================
# WRITING AND COMMUNICATION
# =============================================================================
WRITING = [
    ("How do I write a good introduction?", "A good introduction should: 1) Hook the reader with an interesting opening (question, fact, anecdote), 2) Provide context/background, 3) State your thesis or main point clearly, 4) Preview your main arguments. Keep it concise—about 10% of total length."),
    ("What makes a strong thesis statement?", "A strong thesis: 1) Takes a clear position (not just states a fact), 2) Is specific and focused, 3) Is arguable (others could disagree), 4) Previews your main points. Example: 'Remote work increases productivity because it eliminates commute time, reduces distractions, and improves work-life balance.'"),
    ("How do I write a professional email?", "Professional email structure:\n1. Clear subject line\n2. Appropriate greeting (Dear/Hi [Name])\n3. Purpose in first sentence\n4. Body: concise, one topic per paragraph\n5. Clear call to action if needed\n6. Professional closing (Best regards/Thank you)\n7. Signature\n\nProofread before sending!"),
    ("What is active vs passive voice?", "Active voice: Subject does the action. 'The dog bit the man.'\nPassive voice: Subject receives the action. 'The man was bitten by the dog.'\n\nActive is usually clearer and more direct. Use passive when the action or receiver is more important than the actor, or when the actor is unknown."),
    ("How do I improve my writing?", "To improve writing: 1) Read widely to absorb good style, 2) Write regularly—daily if possible, 3) Edit ruthlessly—cut unnecessary words, 4) Get feedback from others, 5) Study grammar and style guides, 6) Read your writing aloud to catch issues, 7) Learn from writers you admire."),
    ("How do I structure a paragraph?", "A well-structured paragraph has: 1) Topic sentence (main idea), 2) Supporting sentences (evidence, examples, explanations), 3) Concluding sentence (summary or transition). Each paragraph should focus on one main idea. Connect paragraphs with transitions."),
]

# =============================================================================
# PROFESSIONAL SKILLS
# =============================================================================
PROFESSIONAL = [
    ("How do I prepare for a job interview?", "Interview prep: 1) Research the company thoroughly, 2) Review the job description and match your skills, 3) Prepare STAR stories (Situation, Task, Action, Result), 4) Practice common questions aloud, 5) Prepare thoughtful questions to ask, 6) Plan your outfit and route, 7) Bring copies of your resume."),
    ("How do I write a resume?", "Resume tips: 1) Keep to 1-2 pages, 2) Lead with a summary of qualifications, 3) List experience in reverse chronological order, 4) Use action verbs and quantify achievements, 5) Tailor to each job, 6) Include relevant skills and education, 7) Proofread carefully, 8) Use clean, readable formatting."),
    ("What is project management?", "Project management is planning, organizing, and overseeing work to achieve goals within constraints (time, budget, scope). Key elements: defining objectives, creating timelines, allocating resources, managing risks, tracking progress, communicating with stakeholders. Common frameworks: Agile, Scrum, Waterfall."),
    ("How do I give effective feedback?", "Effective feedback: 1) Be specific, not vague, 2) Focus on behavior, not personality, 3) Give it promptly, 4) Balance positive and constructive, 5) Offer suggestions for improvement, 6) Make it a dialogue, not a lecture, 7) Follow up to see progress."),
    ("How do I manage my time better?", "Time management tips: 1) Prioritize using the Eisenhower Matrix (urgent/important), 2) Time-block your calendar, 3) Tackle hardest tasks when you're most alert, 4) Minimize distractions and multitasking, 5) Break large tasks into smaller steps, 6) Learn to say no, 7) Review and plan weekly."),
    ("What is emotional intelligence?", "Emotional intelligence (EQ) is the ability to recognize, understand, and manage emotions in yourself and others. Components: 1) Self-awareness, 2) Self-regulation, 3) Motivation, 4) Empathy, 5) Social skills. High EQ improves relationships, leadership, and decision-making."),
]

# =============================================================================
# CREATIVE AND FUN
# =============================================================================
CREATIVE = [
    ("Tell me a fun fact", "Honey never spoils! Archaeologists have found 3,000-year-old honey in Egyptian tombs that was still perfectly edible. Honey's low moisture content and acidic pH create an environment where bacteria can't survive."),
    ("Give me a creative writing prompt", "Write a story that begins: 'The last library on Earth held only one book, and today someone finally came to read it.' Consider: What's in the book? Why is it the last library? Who is the reader?"),
    ("What's an interesting paradox?", "The Ship of Theseus: If you replace every plank of a ship over time, is it still the same ship? And if you rebuild a ship from all the old planks, which one is the 'real' ship? This raises questions about identity and continuity."),
    ("Tell me a riddle", "I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?\n\nAnswer: An echo."),
    ("Give me a tongue twister", "She sells seashells by the seashore. The shells she sells are seashells, I'm sure. So if she sells shells on the seashore, then I'm sure she sells seashore shells."),
    ("What are some ways to be more creative?", "Boost creativity: 1) Expose yourself to diverse experiences and ideas, 2) Keep a journal for thoughts and observations, 3) Practice brainstorming without judgment, 4) Take breaks to let ideas incubate, 5) Combine unrelated concepts, 6) Ask 'what if?' questions, 7) Embrace constraints as creative challenges."),
]

# =============================================================================
# PRACTICAL ADVICE
# =============================================================================
PRACTICAL = [
    ("How do I learn a new skill faster?", "Learn faster with: 1) Deliberate practice—focus on weak areas, 2) Spaced repetition—review over increasing intervals, 3) Teach what you learn—explaining solidifies understanding, 4) Get immediate feedback, 5) Break skills into sub-skills, 6) Practice in varied contexts, 7) Stay consistent over intensity."),
    ("How do I stay motivated?", "Stay motivated by: 1) Setting clear, meaningful goals, 2) Breaking big goals into small wins, 3) Tracking your progress visibly, 4) Finding an accountability partner, 5) Connecting tasks to your values, 6) Building habits and routines, 7) Celebrating achievements, 8) Managing energy, not just time."),
    ("How do I deal with stress?", "Manage stress by: 1) Exercise regularly—even short walks help, 2) Practice deep breathing or meditation, 3) Get adequate sleep, 4) Talk to someone you trust, 5) Break overwhelming tasks into steps, 6) Limit caffeine and alcohol, 7) Make time for activities you enjoy, 8) Set realistic expectations."),
    ("How do I form good habits?", "Form habits with: 1) Start tiny—make it so easy you can't say no, 2) Attach to existing routines (habit stacking), 3) Make it obvious (visual cues), 4) Make it attractive (pair with something you like), 5) Make it easy (reduce friction), 6) Make it satisfying (immediate reward), 7) Track your streak."),
    ("How do I read more effectively?", "Read effectively: 1) Preview—skim headings, summaries, conclusions first, 2) Ask questions—what do you want to learn?, 3) Read actively—highlight, annotate, question, 4) Summarize in your own words, 5) Review periodically to retain, 6) Apply what you learn, 7) Adjust speed to content difficulty."),
    ("How do I make better decisions?", "Better decisions: 1) Define the decision clearly, 2) Gather information but set a deadline, 3) Consider multiple options, 4) Think about second-order effects, 5) Use a pros/cons list or decision matrix, 6) Get input from others, 7) Trust your gut for certain decisions, 8) Accept that perfect info isn't possible."),
]

# =============================================================================
# MILITARY AND INTELLIGENCE OPERATIONS
# =============================================================================
MILITARY_INTEL_OPS = [
    # Psychological Operations (PSYOP/MISO)
    ("What are psychological operations (PSYOP)?", """Psychological Operations (PSYOP), now termed Military Information Support Operations (MISO), are planned operations to convey selected information and indicators to foreign audiences to influence their emotions, motives, objective reasoning, and ultimately behavior of foreign governments, organizations, groups, and individuals.

Key components of PSYOP:
1. Target Audience Analysis (TAA): Identifying and understanding the psychological vulnerabilities, cultural factors, and decision-making processes of target populations
2. Product Development: Creating persuasive content (leaflets, broadcasts, digital media, face-to-face communication) tailored to the target audience
3. Dissemination: Delivering messages through appropriate media channels (radio, television, internet, print, loudspeakers)
4. Assessment: Measuring effectiveness through behavioral indicators and feedback

PSYOP operates at three levels:
- Strategic: Long-term campaigns targeting national-level audiences to achieve broad policy objectives
- Operational: Theater-level operations supporting military campaign objectives
- Tactical: Immediate battlefield influence supporting combat operations

Legal framework: PSYOP targeting foreign audiences is authorized under Title 10 (military operations). Domestic influence operations on US persons are prohibited. All PSYOP must be truthful—'white' propaganda attributed to US government sources."""),

    ("What is the difference between white, gray, and black propaganda?", """Propaganda is classified by attribution:

WHITE PROPAGANDA:
- Source is openly acknowledged and identified
- Content is factually accurate
- Example: Voice of America broadcasts clearly identified as US government programming
- Used for credibility and long-term influence
- Legal requirement for US PSYOP operations

GRAY PROPAGANDA:
- Source is not identified or deliberately ambiguous
- Content may be true, false, or mixed
- Example: News articles placed without revealing government sponsorship
- Provides plausible deniability
- Can be more persuasive as it appears independent

BLACK PROPAGANDA:
- Falsely attributed to a source other than the true origin
- Often attributes content to the adversary to discredit them
- Example: Fake enemy communications designed to demoralize or deceive
- High risk if exposed—damages credibility
- Typically conducted by intelligence agencies rather than military

The US officially commits to truthful (white) propaganda in acknowledged operations, though intelligence agencies have historically conducted gray and black operations. The distinction matters for legal authorities, oversight, and long-term credibility."""),

    ("How does the US military conduct information operations?", """US Military Information Operations (IO) integrate multiple information-related capabilities to influence, disrupt, corrupt, or usurp the decision-making of adversaries while protecting our own.

Core Information-Related Capabilities (IRCs):
1. MISO (Military Information Support Operations): Psychological influence on foreign audiences
2. OPSEC (Operations Security): Protecting friendly information from adversary collection
3. Military Deception (MILDEC): Misleading adversary decision-makers
4. Electronic Warfare (EW): Using electromagnetic spectrum for attack, protection, and exploitation
5. Cyberspace Operations: Offensive and defensive cyber activities
6. Civil Affairs: Engaging civilian populations and governments
7. Public Affairs: Truthful information to domestic and international media
8. Combat Camera: Visual documentation for multiple purposes

IO Planning Process:
1. Define information environment and key actors
2. Identify adversary decision-making processes and vulnerabilities
3. Develop desired behavioral outcomes
4. Synchronize IRCs to achieve effects
5. Assess and adapt based on feedback

Modern IO emphasizes the cognitive dimension—affecting how people think and decide—rather than just physical destruction of communication infrastructure."""),

    ("What is military deception (MILDEC)?", """Military Deception (MILDEC) consists of actions executed to deliberately mislead adversary military, paramilitary, or violent extremist organization decision makers, causing them to take actions (or inactions) that contribute to the success of friendly operations.

Principles of MILDEC:
1. Focus: Target specific adversary decision-makers
2. Objective: Support the overall operation plan
3. Centralized Control: Single authority coordinates all deception
4. Security: Protect the deception plan as classified
5. Timeliness: Allow enough time for adversary to receive, process, and act
6. Integration: Synchronize with other operations

Types of MILDEC:
- Feints: Limited attacks to deceive about main effort location
- Demonstrations: Shows of force without decisive engagement
- Ruses: Tricks using enemy uniforms, false communications, dummy equipment
- Displays: Static deception using decoys and camouflage
- Disinformation: False information fed to adversary intelligence

Historical Example - Operation Fortitude (WWII):
Created fictional First US Army Group under Patton threatening Pas-de-Calais, using inflatable tanks, fake radio traffic, and double agents. Kept German reserves away from Normandy before and after D-Day.

MILDEC must be legally reviewed—perfidy (misusing protected symbols like Red Cross) is prohibited under Law of Armed Conflict."""),

    ("How do influence operations work on social media?", """Social media influence operations exploit digital platforms to shape public opinion, often covertly. Understanding these operations is essential for defense and detection.

Tactics, Techniques, and Procedures (TTPs):
1. Persona Development: Creating fake accounts with believable histories, photos (often AI-generated), and activity patterns
2. Network Building: Fake accounts follow each other to appear legitimate; gradual audience growth
3. Content Seeding: Introducing narratives through memes, articles, videos designed for sharing
4. Amplification: Coordinated sharing/liking to boost algorithmic visibility
5. Engagement Farming: Provocative content generates authentic engagement, increasing reach
6. Hashtag Hijacking: Inserting messages into trending conversations
7. Platform Arbitrage: Starting content on fringe platforms, migrating to mainstream

Detection Indicators:
- Coordinated inauthentic behavior (accounts acting in unison)
- Unusual posting patterns (timing, volume)
- Stolen or AI-generated profile images
- Network analysis reveals artificial clustering
- Content traces to known state-sponsored sources
- Narrative alignment with foreign government positions

Notable Operations:
- Russia's Internet Research Agency (2016 US election)
- China's 50 Cent Army and Spamouflage network
- Iran's International Union of Virtual Media

Defense requires platform cooperation, media literacy, and attribution capabilities."""),

    ("What is counterintelligence and how does it work?", """Counterintelligence (CI) encompasses activities to identify, deceive, exploit, disrupt, or protect against espionage, other intelligence activities, sabotage, or assassinations conducted by foreign powers, organizations, or their agents.

CI Functions:
1. Detection: Identifying foreign intelligence threats through analysis, surveillance, penetrations
2. Investigation: Determining scope of compromise, identifying perpetrators
3. Neutralization: Arresting spies, expelling diplomats, disrupting operations
4. Exploitation: Turning detected operations to friendly advantage (double agents, feeding disinformation)
5. Protection: Hardening targets, security awareness, vetting personnel

CI Disciplines:
- Offensive CI: Actively penetrating foreign intelligence services
- Defensive CI: Protecting own personnel, facilities, and information
- Collection CI: Protecting intelligence sources and methods

Key Concepts:
- Mole Hunt: Investigating suspected penetration of own organization
- Double Agent: Spy who pretends to work for adversary but actually serves friendly service
- Dangle: Officer presented to foreign service as potential recruit (controlled operation)
- False Flag: Recruiting source who believes they're working for different country

US CI Organizations:
- FBI: Lead for CI within US territory
- CIA: Lead for CI abroad
- Military CI: Service-specific (Army CID, NCIS, OSI, etc.)
- NCSC: Coordinates national CI policy

CI is inherently paranoid by necessity—must assume adversaries are always attempting penetration."""),

    ("How does the CIA conduct covert action?", """Covert Action is activity to influence political, economic, or military conditions abroad where US government role is not intended to be apparent or acknowledged. It's authorized under Title 50 USC and requires Presidential Finding and Congressional notification.

Types of Covert Action:
1. Propaganda/Influence: Media placement, funding journalists, support for political parties
2. Political Action: Supporting/opposing foreign political movements, leaders, elections
3. Economic Action: Disrupting adversary economies, supporting friendly ones covertly
4. Paramilitary/Military: Arming and training insurgent forces, direct action
5. Cyber Operations: Computer network attacks with non-acknowledged attribution

Legal Framework:
- Presidential Finding: Written authorization required before action
- Congressional Notification: Gang of Eight or full intelligence committees
- Prohibition on Assassination: Executive Order 12333 (though 'targeted killing' in armed conflict differs)
- No US Person Targeting: Cannot target Americans

Historical Examples:
- Operation AJAX (1953): Overthrow of Iranian PM Mossadegh
- Bay of Pigs (1961): Failed Cuban invasion
- Operation Cyclone (1980s): Arming Afghan mujahideen
- Contemporary: Drone programs, cyber operations against adversaries

Covert action is controversial—short-term gains may create long-term blowback. Effectiveness depends on deniability, local conditions, and alignment with broader policy."""),

    ("What is the intelligence cycle?", """The Intelligence Cycle is the process of developing raw information into finished intelligence for policymakers. While depicted as linear, it's actually iterative and continuous.

Phases of the Intelligence Cycle:

1. PLANNING AND DIRECTION
- Policymakers identify information needs (Priority Intelligence Requirements - PIRs)
- Intelligence Community translates into collection requirements
- Resources allocated to highest priorities

2. COLLECTION
- HUMINT: Human sources (spies, diplomats, travelers)
- SIGINT: Signals intelligence (communications, electronic signals)
- GEOINT: Imagery and geospatial data (satellites, drones, maps)
- MASINT: Measurement and signature intelligence (radar, nuclear, chemical)
- OSINT: Open source intelligence (media, academic, commercial)

3. PROCESSING AND EXPLOITATION
- Converting raw collection into usable form
- Translation, decryption, image interpretation
- Database entry and initial sorting

4. ANALYSIS AND PRODUCTION
- All-source analysts integrate collection from multiple disciplines
- Assess reliability, resolve conflicts, identify gaps
- Produce finished intelligence products (assessments, estimates, briefings)

5. DISSEMINATION
- Deliver to appropriate consumers in timely manner
- Classification and need-to-know considerations
- Multiple formats: PDB, NIEs, current intelligence, warnings

6. EVALUATION AND FEEDBACK
- Consumers provide feedback on utility
- Identify new requirements, adjust priorities
- Lessons learned improve future cycles

The cycle never truly ends—intelligence is continuous, and each answer generates new questions."""),

    ("What are the 18 members of the US Intelligence Community?", """The US Intelligence Community (IC) comprises 18 organizations:

INDEPENDENT AGENCIES:
1. Office of the Director of National Intelligence (ODNI) - Coordinates IC, produces national intelligence
2. Central Intelligence Agency (CIA) - HUMINT collection abroad, covert action, all-source analysis

DEPARTMENT OF DEFENSE:
3. Defense Intelligence Agency (DIA) - Military intelligence, defense attachés
4. National Security Agency (NSA) - SIGINT collection and cybersecurity
5. National Geospatial-Intelligence Agency (NGA) - Imagery and geospatial intelligence
6. National Reconnaissance Office (NRO) - Designs and operates reconnaissance satellites
7. Army Intelligence (G-2/INSCOM)
8. Navy Intelligence (ONI)
9. Marine Corps Intelligence
10. Air Force Intelligence (A-2)
11. Space Force Intelligence (S-2)

DEPARTMENT OF JUSTICE:
12. Federal Bureau of Investigation (FBI) - Domestic counterintelligence and counterterrorism
13. Drug Enforcement Administration (DEA) - Drug-related intelligence

DEPARTMENT OF HOMELAND SECURITY:
14. Office of Intelligence and Analysis (I&A) - Threat information to state/local
15. Coast Guard Intelligence

DEPARTMENT OF STATE:
16. Bureau of Intelligence and Research (INR) - Diplomatic intelligence, alternative analysis

DEPARTMENT OF TREASURY:
17. Office of Intelligence and Analysis - Financial intelligence, sanctions

DEPARTMENT OF ENERGY:
18. Office of Intelligence and Counterintelligence - Nuclear weapons, energy security

Each has distinct authorities, capabilities, and focus areas but shares intelligence through IC-wide systems."""),

    ("How do special operations forces conduct unconventional warfare?", """Unconventional Warfare (UW) consists of activities conducted to enable a resistance movement or insurgency to coerce, disrupt, or overthrow a government or occupying power by operating through or with an underground, auxiliary, and guerrilla force in a denied area.

Phases of UW:
1. PREPARATION: Assess potential resistance movements, develop plans, train personnel
2. INITIAL CONTACT: Infiltrate operational area, contact resistance leadership
3. INFILTRATION: Insert additional personnel, equipment, and communications
4. ORGANIZATION: Structure resistance into underground, auxiliary, and guerrilla components
5. BUILDUP: Train, equip, and expand resistance forces
6. EMPLOYMENT: Direct resistance operations against adversary
7. TRANSITION: Shift to conventional operations or post-conflict stabilization

Resistance Structure:
- Underground: Clandestine cells conducting sabotage, intelligence, subversion
- Auxiliary: Civilian supporters providing safe houses, logistics, intelligence
- Guerrilla Force: Armed irregular military conducting combat operations

Key UW Tasks:
- Training indigenous forces in tactics, weapons, communications
- Advising resistance leadership on strategy and operations
- Providing material support (weapons, ammunition, medical supplies)
- Coordinating with conventional forces and theater command
- Psychological operations to build support and undermine adversary

US Army Special Forces (Green Berets) are primary UW force, organized into Operational Detachments Alpha (ODA) of 12 soldiers specializing in language, culture, and unconventional tactics.

UW is politically sensitive—requires clear policy guidance on end states and acceptable methods."""),

    ("What is electronic warfare?", """Electronic Warfare (EW) consists of military action involving electromagnetic spectrum (EMS) to control the spectrum, attack an adversary, or impede adversary operations.

Three Divisions of EW:

1. ELECTRONIC ATTACK (EA)
- Jamming: Overwhelming enemy receivers with noise or deceptive signals
- Directed Energy: Lasers and high-power microwaves to damage equipment
- Anti-radiation Missiles: Home on enemy radar emissions
- Electromagnetic Pulse (EMP): Broad-spectrum disruption
- Spoofing: Transmitting false signals (GPS spoofing, radar deception)

2. ELECTRONIC PROTECTION (EP)
- Frequency hopping: Rapidly changing frequencies to avoid jamming
- Spread spectrum: Distributing signal across wide bandwidth
- Emission control (EMCON): Limiting own electromagnetic signatures
- Hardening: Shielding equipment from EMP and directed energy
- Redundancy: Multiple communication paths

3. ELECTRONIC WARFARE SUPPORT (ES)
- Signals interception and direction finding
- Identifying and locating emitters
- Threat warning receivers
- Electronic Order of Battle development
- Real-time intelligence for EA and EP

EW Platforms:
- EA-18G Growler: Navy's primary EW aircraft
- EC-130H Compass Call: Airborne jamming and PSYOP
- Ground-based jammers: Counter-IED, counter-communications
- Space-based: Satellite jamming capabilities (both ways)

Modern EW integrates with cyber operations—the distinction between electromagnetic and cyber effects is blurring in software-defined radio and networked systems."""),

    ("How does signals intelligence (SIGINT) collection work?", """Signals Intelligence (SIGINT) is intelligence derived from electronic signals and systems, comprising communications intelligence (COMINT) and electronic intelligence (ELINT).

COMINT - Communications Intelligence:
- Intercepts communications between people (voice, text, email, messaging)
- Targets: diplomatic cables, military communications, terrorist networks, criminal organizations
- Methods: Satellite intercept, undersea cable tapping, network exploitation, close-access operations
- Processing: Decryption, translation, voice recognition, traffic analysis

ELINT - Electronic Intelligence:
- Non-communications electromagnetic emissions (radar, weapons systems)
- Technical parameters: frequency, pulse width, scan pattern
- Used to develop electronic order of battle and countermeasures
- Critical for designing jamming systems and stealth technology

SIGINT Process:
1. Collection: Sensors intercept signals of interest
2. Processing: Decrypt, filter, convert to usable form
3. Analysis: Identify communicants, interpret content, assess significance
4. Reporting: Disseminate to consumers with appropriate classification

Collection Platforms:
- Satellites: Geosynchronous and low-earth orbit collectors
- Aircraft: RC-135 Rivet Joint, EP-3 Aries, drones
- Ground stations: Global network of intercept facilities
- Naval vessels: Surface ships and submarines
- Close access: Implants in target networks and facilities

NSA is the primary US SIGINT agency. Legal authorities include FISA for domestic-foreign communications and Executive Order 12333 for foreign intelligence abroad.

Traffic analysis—studying patterns without content—can reveal significant intelligence even when encryption prevents reading content."""),

    ("What is the role of human intelligence (HUMINT) in modern intelligence?", """Human Intelligence (HUMINT) is intelligence derived from human sources through clandestine collection, overt contact, or debriefing. Despite technological advances, HUMINT remains essential for understanding intentions, plans, and decision-making.

Types of HUMINT Sources:
1. Recruited Agents: Foreign nationals secretly working for US intelligence
2. Access Agents: Provide access to targets or other potential sources
3. Principal Agents: Run networks of sub-sources
4. Defectors: Walk-ins who switch sides permanently
5. Diplomats/Attachés: Overt collectors with official cover
6. Travelers: Businesspeople, academics with natural access

The Recruitment Cycle:
1. SPOTTING: Identifying individuals with access and potential vulnerabilities
2. ASSESSING: Determining suitability, access, motivation
3. DEVELOPING: Building relationship, testing reliability
4. RECRUITING: Formal pitch for cooperation
5. HANDLING: Managing ongoing relationship, tasking, security
6. TERMINATION: Ending relationship safely

Cover Types:
- Official Cover: Diplomatic or government position (legal protection if caught)
- Non-Official Cover (NOC): Commercial or private persona (no diplomatic immunity)
- Deep Cover: Extended cover building before operational use

HUMINT Advantages:
- Access to intentions and plans (not just capabilities)
- Answers specific questions (targetable)
- Context and interpretation
- Can influence as well as collect

HUMINT Challenges:
- Time-intensive to develop
- Security risks (double agents, dangles)
- Deception and fabrication
- Legal and ethical constraints

CIA's Directorate of Operations (now National Clandestine Service) is primary HUMINT collector abroad; FBI and military services also conduct HUMINT."""),

    ("What are influence operations and how are they conducted?", """Influence Operations are coordinated efforts to affect target audience perceptions, attitudes, and behaviors to achieve political, military, or economic objectives. They integrate multiple capabilities across the information environment.

Components of Influence Operations:
1. Public Diplomacy: Overt government communication to foreign publics
2. PSYOP/MISO: Military psychological operations
3. Covert Influence: Deniable propaganda and political action
4. Cyber-enabled Influence: Social media manipulation, hacking and leaking
5. Economic Leverage: Sanctions, aid, trade as influence tools
6. Cultural Programs: Exchanges, broadcasting, educational initiatives

Influence Operation Planning:
1. Define Objectives: What behavior change is desired?
2. Target Audience Analysis: Who needs to be influenced? What are their vulnerabilities?
3. Message Development: What narratives will resonate?
4. Channel Selection: How to reach audience credibly?
5. Synchronization: Coordinate multiple efforts for reinforcement
6. Assessment: Measure behavioral change, adapt approach

Key Concepts:
- Narrative Warfare: Competing to establish dominant story/interpretation
- Perception Management: Conveying selected information to influence attitudes
- Strategic Communications: Aligning words and actions for consistent message
- Reflexive Control: Causing adversary to make decisions that favor your objectives

Foreign Influence Examples:
- Soviet 'active measures' during Cold War
- Russian interference in 2016 US election
- Chinese 'United Front' operations
- Iranian social media campaigns

Defense requires whole-of-society approach: media literacy, platform policies, government transparency, and attribution capabilities."""),

    ("How do cyber operations support military objectives?", """Cyber Operations are the employment of cyberspace capabilities to achieve objectives in or through cyberspace. Military cyber integrates with kinetic operations to create combined effects.

Types of Military Cyber Operations:

OFFENSIVE CYBER OPERATIONS (OCO):
- Disruption: Temporarily degrading adversary systems (DDoS, logic bombs)
- Destruction: Permanently damaging systems or data
- Manipulation: Altering data for deception or operational effect
- Espionage: Penetrating networks for intelligence collection
- Examples: Stuxnet (Iranian centrifuges), attacks on ISIS networks

DEFENSIVE CYBER OPERATIONS (DCO):
- Network defense: Protecting DOD systems from intrusion
- Threat hunting: Proactively searching for adversary presence
- Incident response: Containing and remediating compromises
- Vulnerability management: Patching and hardening systems

CYBERSPACE EXPLOITATION:
- Intelligence collection through network access
- Preparing access for future operations
- Mapping adversary networks and capabilities

Integration with Military Operations:
- Cyber effects synchronized with kinetic strikes
- Electronic warfare and cyber overlap (software-defined radio)
- Information operations amplified through cyber
- Space-cyber integration (satellite communications, GPS)

Legal Framework:
- Title 10: Military cyber operations
- Title 50: Intelligence cyber operations
- Law of Armed Conflict applies to cyber weapons
- Sovereignty considerations for peacetime operations

US Cyber Command (USCYBERCOM) commands military cyber forces, with service components (ARCYBER, FLTCYBER, AFCYBER, MARFORCYBER). NSA relationship provides unique signals intelligence capabilities."""),

    ("What is the Cyber Kill Chain and how is it used?", """The Cyber Kill Chain, developed by Lockheed Martin, models stages of cyber intrusions, enabling defenders to identify and disrupt attacks at each phase.

Seven Stages:

1. RECONNAISSANCE
- Attacker researches target: employees, technologies, vulnerabilities
- OSINT gathering, social engineering, scanning
- Defense: Limit public information, monitor for reconnaissance activity

2. WEAPONIZATION
- Creating malicious payload tailored to target
- Combining exploit with backdoor/RAT
- Defense: Threat intelligence on adversary tools, sandboxing

3. DELIVERY
- Transmitting weapon to target environment
- Vectors: phishing email, watering hole, USB drop, supply chain
- Defense: Email filtering, web proxies, USB policies, user training

4. EXPLOITATION
- Triggering vulnerability to execute code
- Browser exploits, document exploits, OS vulnerabilities
- Defense: Patching, application whitelisting, exploit protection

5. INSTALLATION
- Installing persistent backdoor on victim system
- Registry modifications, scheduled tasks, bootkit
- Defense: Endpoint detection, application control, integrity monitoring

6. COMMAND AND CONTROL (C2)
- Establishing communication channel to attacker infrastructure
- HTTP/HTTPS, DNS tunneling, social media, custom protocols
- Defense: Network monitoring, DNS analysis, egress filtering

7. ACTIONS ON OBJECTIVES
- Attacker achieves goal: data exfiltration, destruction, manipulation
- Lateral movement to additional targets
- Defense: Data loss prevention, segmentation, insider threat detection

Using the Kill Chain:
- Intelligence-Driven Defense: Map adversary TTPs to kill chain stages
- Courses of Action Matrix: Identify detection/mitigation options per phase
- Measure Effectiveness: Track which stages defenses are stopping
- Prioritize Investment: Address weakest points in defensive coverage

Defenders win by breaking ANY link in the chain. Attackers must succeed at every stage."""),

    ("How do nation-states conduct advanced persistent threats (APTs)?", """Advanced Persistent Threats (APTs) are sophisticated, long-term cyber campaigns conducted by nation-states or state-sponsored groups targeting specific organizations for strategic objectives.

APT Characteristics:
- Advanced: Custom tools, zero-day exploits, operational security
- Persistent: Long dwell times (months to years), re-compromise after detection
- Threat: Organized, resourced, strategic objectives (not opportunistic crime)

APT Lifecycle:
1. Target Selection: Strategic value assessment, reconnaissance
2. Initial Compromise: Spear-phishing, watering holes, supply chain
3. Establish Foothold: Implants, backdoors, legitimate tool abuse
4. Escalate Privileges: Credential theft, exploitation
5. Internal Recon: Network mapping, identifying high-value targets
6. Lateral Movement: Spreading through network using stolen credentials
7. Maintain Presence: Multiple backdoors, dormant implants
8. Complete Mission: Exfiltrate data, position for future operations

Notable APT Groups:
- APT28/Fancy Bear (Russia/GRU): Political/military targets, election interference
- APT29/Cozy Bear (Russia/SVR): Government, think tanks, espionage
- APT1/Comment Crew (China/PLA): Economic espionage, IP theft
- APT41 (China): Dual espionage and financial crime
- Lazarus Group (North Korea): Financial theft, destructive attacks
- APT33 (Iran): Energy sector, destructive malware

Detection Strategies:
- Behavioral analysis over signature detection
- Network traffic anomaly detection
- Endpoint detection and response (EDR)
- Threat hunting based on adversary TTPs
- Intelligence sharing and attribution

APT defense requires assuming breach—focus on detection, response, and limiting damage rather than prevention alone."""),

    ("What is counterterrorism intelligence and how does it work?", """Counterterrorism (CT) intelligence focuses on identifying, locating, and enabling action against terrorist threats before attacks occur.

CT Intelligence Functions:
1. Strategic Analysis: Understanding terrorist group ideology, organization, capabilities
2. Operational Intelligence: Supporting specific CT operations and investigations
3. Tactical Intelligence: Real-time support to raids, captures, interdiction
4. Warning Intelligence: Alerting to imminent threats

Collection for CT:
- HUMINT: Penetrating terrorist organizations, recruiting sources
- SIGINT: Intercepting terrorist communications, monitoring financiers
- GEOINT: Tracking facilities, movements, activity patterns
- Financial Intelligence: Following money flows, identifying donors
- OSINT: Monitoring propaganda, social media, public statements

CT Analytical Methods:
- Link Analysis: Mapping relationships between individuals and organizations
- Pattern Analysis: Identifying operational signatures, attack precursors
- Social Network Analysis: Understanding group structure and key nodes
- Geospatial Analysis: Tracking movements, identifying safe houses
- Behavioral Analysis: Radicalization indicators, pre-attack behaviors

Key CT Organizations:
- NCTC (National Counterterrorism Center): Integration and strategic analysis
- FBI: Domestic CT investigations and intelligence
- CIA/CTC: Foreign CT intelligence and operations
- DIA: Military CT intelligence
- Treasury: Terrorist financing

Post-9/11 Reforms:
- Information sharing mandated across agencies
- NCTC created for integration
- FBI shifted from prosecution to prevention
- Military increased CT role (JSOC)
- Surveillance authorities expanded (FISA, PATRIOT Act)

Challenges: Homegrown violent extremists, encrypted communications, balancing security with civil liberties, evolving threat landscape."""),

    ("How does military intelligence support tactical operations?", """Military Intelligence (MI) provides commanders with information about enemy forces, terrain, and civil considerations to enable tactical decision-making.

Intelligence Preparation of the Battlefield (IPB):
1. Define the Operational Environment: Area of operations, area of interest
2. Describe Environmental Effects: Terrain, weather, civil considerations
3. Evaluate the Threat: Order of battle, capabilities, doctrine
4. Determine Threat Courses of Action: Most likely, most dangerous enemy actions

Tactical Intelligence Collection:
- Ground Reconnaissance: Scouts, patrols, observation posts
- Aerial ISR: Drones (RQ-7, MQ-1C), aircraft, satellites
- SIGINT: Ground-based intercept, electronic warfare units
- HUMINT: Tactical questioning, civil engagement, captured documents
- Geospatial: Mapping, terrain analysis, change detection

Intelligence Products:
- Intelligence Summary (INTSUM): Periodic threat assessment
- Spot Reports: Immediate reporting of significant activity
- Target Folders: Detailed information for strike coordination
- Pattern of Life Analysis: Understanding normal activity for anomaly detection
- High-Value Target (HVT) Packets: Information to support capture/kill operations

Processing, Exploitation, and Dissemination (PED):
- Analysis cells process raw collection
- Full-motion video analysts support ongoing operations
- Document and media exploitation (DOMEX)
- Sensitive site exploitation (SSE) after raids

MI Organizations:
- S-2/G-2: Battalion/brigade/division intelligence staff
- Military Intelligence Battalion: Collection and analysis at division
- Theater Intelligence: National assets support to tactical units
- INSCOM: Army's strategic intelligence command

Key challenge: Pushing intelligence to lowest levels fast enough to enable tactical action while maintaining security."""),

    ("What is the role of special access programs (SAPs) in intelligence?", """Special Access Programs (SAPs) are security protocols providing enhanced protection for exceptionally sensitive information beyond standard classification levels.

Types of SAPs:
1. Acquisition SAPs (ACQ-SAP): Protect sensitive technologies, weapons systems
2. Intelligence SAPs (IN-SAP): Protect intelligence sources and methods
3. Operations SAPs (OP-SAP): Protect military operations and planning

SAP Security Features:
- 'Need to Know' strictly enforced beyond clearance level
- Separate personnel security vetting (polygraph, investigation)
- Compartmentalization: Access limited to specific compartments
- Special handling procedures and dedicated facilities
- Separate classification guides and marking requirements

Acknowledged vs Unacknowledged:
- Acknowledged SAP: Existence can be publicly confirmed, though details classified
- Unacknowledged SAP: Existence itself is classified (waived SAP = no Congressional notification)
- Cover stories may be created for unacknowledged programs

Access Process:
1. Nomination by program manager
2. Security investigation and adjudication
3. Indoctrination briefing and NDA
4. Access granted to specific compartments
5. Periodic reinvestigation and polygraph
6. Debriefing upon termination

Historical SAP Examples:
- MANHATTAN PROJECT (atomic bomb development)
- OXCART/SR-71 development
- F-117 stealth fighter program
- Various NSA collection programs

SAP oversight includes Congressional notifications (Gang of Eight for most sensitive), Inspector General access, and Executive Branch review. Critics argue SAPs can reduce accountability; proponents argue necessary to protect critical capabilities."""),

    ("How do intelligence agencies conduct technical surveillance?", """Technical Surveillance refers to the use of technology to monitor communications, activities, and locations of intelligence targets. Methods and legal frameworks vary by target type and location.

Communications Surveillance:
- Telephone interception: Wiretaps, switch-based collection, SS7 exploitation
- Email/Internet: Network taps, server access, provider cooperation
- Mobile: Cell tower simulators (Stingrays), baseband vulnerabilities
- Encrypted communications: Lawful access demands, endpoint exploitation
- Metadata collection: Call records, location data, connection logs

Physical Surveillance Technology:
- Audio: Room bugs, laser microphones, parabolic collectors
- Video: Covert cameras, drone surveillance, satellite imagery
- Tracking: GPS devices, mobile phone location, vehicle trackers
- Biometrics: Facial recognition, gait analysis, voice identification

Computer Network Exploitation:
- Implants: Persistent access to target systems
- Supply chain interdiction: Compromising hardware before delivery
- Offensive cyber tools: Exploiting vulnerabilities for access
- Cloud access: Provider cooperation or exploitation

Legal Framework (US):
- Title III: Criminal wiretaps require court order
- FISA: Foreign intelligence surveillance requires FISC approval
- Section 702: Collection targeting foreigners abroad (incidental US person collection)
- Executive Order 12333: Intelligence activities abroad
- Consent exceptions: If one party consents

Oversight Mechanisms:
- FISA Court (FISC) for domestic targeting
- Congressional intelligence committees
- Inspectors General
- Privacy and Civil Liberties Oversight Board

Technical surveillance capabilities have expanded dramatically with digital technology, creating ongoing debates about privacy, security, and civil liberties."""),

    ("What is the difference between strategic and tactical intelligence?", """Intelligence is categorized by level of war and consumer needs, from national strategic to small-unit tactical.

STRATEGIC INTELLIGENCE:
Purpose: Inform national policy and strategy
Consumers: President, Cabinet, Congress, senior military leaders
Scope: Global, long-term trends and capabilities
Products: National Intelligence Estimates (NIEs), Presidential Daily Brief (PDB)
Timeframe: Months to years
Sources: All disciplines, heavy national technical means
Organizations: CIA, DIA, INR, ODNI, service intelligence centers

Examples:
- Assessment of China's military modernization
- North Korean nuclear program analysis
- Russian political stability estimates
- Long-term terrorism trends

OPERATIONAL INTELLIGENCE:
Purpose: Support campaign and theater planning
Consumers: Combatant commanders, component commanders
Scope: Theater-wide, medium-term
Products: Joint Intelligence Preparation of Operational Environment (JIPOE)
Timeframe: Weeks to months
Sources: National and theater collection integrated
Organizations: Joint Intelligence Centers, Theater Intelligence Groups

Examples:
- Enemy order of battle for planned campaign
- Critical infrastructure analysis for targeting
- Adversary commander personality profiles

TACTICAL INTELLIGENCE:
Purpose: Support immediate combat operations
Consumers: Brigade and below commanders, operators
Scope: Local, immediate threat focus
Products: INTSUMs, target packets, threat updates
Timeframe: Hours to days
Sources: Organic collection, direct support from higher
Organizations: Battalion/Brigade S-2, tactical HUMINT, ISR units

Examples:
- Enemy positions for imminent assault
- IED threat on patrol route
- Pattern of life for raid planning

Intelligence must flow both up (tactical collectors feeding strategic analysis) and down (national assets supporting tactical operations). Modern operations blur these distinctions, with strategic assets providing real-time tactical support."""),

    ("How do false flag operations work?", """False flag operations are covert activities designed to appear as if conducted by another party, typically to frame an adversary or provide pretext for action.

Types of False Flag Operations:
1. Attribution Deception: Conducting attack while leaving evidence pointing to another actor
2. Recruiting Under False Flag: Intelligence officer pretends to represent different country when recruiting sources
3. Provocation: Staging incident to justify response (e.g., Gleiwitz incident before WWII)
4. Propaganda Attribution: Creating content appearing to originate from adversary

Historical Examples:
- Operation Northwoods (proposed, not executed): US plan for false Cuban attacks on Americans
- Lavon Affair (1954): Israeli operatives bombed Western targets in Egypt to blame Muslim Brotherhood
- Gleiwitz Incident (1939): Nazi SS staged Polish attack on German radio station
- Various Cold War operations by both sides

Modern Considerations:
- Cyber false flags: Planting malware artifacts, IP addresses suggesting different origin
- Information operations: Creating fake personas/accounts appearing to be from adversary
- Chemical weapons: Accusations of false flags common in Syria conflict

Detection Challenges:
- Technical forensics can be deceived
- Narratives favor preexisting biases
- Proof of true origin difficult to establish
- Even exposed false flags may achieve objectives through confusion

Legal and Ethical Issues:
- Violation of international law if used as pretext for aggression
- Potential war crimes if civilians harmed
- Undermines trust in all attributed actions
- Can backfire catastrophically if exposed

Intelligence analysis must consider false flag possibility when attributing hostile actions, requiring multiple independent sources and careful tradecraft analysis."""),
]

# =============================================================================
# COMPLEX ENGINEERING AND SCIENCE PROBLEM SOLVING
# =============================================================================
COMPLEX_STEM = [
    ("Walk me through designing a heat exchanger for an industrial process.", """HEAT EXCHANGER DESIGN - COMPLETE ENGINEERING PROCESS

STEP 1: DEFINE REQUIREMENTS
- Hot fluid: Process stream at 150°C, flow rate 5 kg/s, specific heat 2.5 kJ/kg·K
- Cold fluid: Cooling water at 20°C, outlet target 40°C
- Required duty: Cool hot fluid to 60°C
- Pressure constraints: Hot side max 500 kPa, cold side max 300 kPa
- Fouling considerations: Process fluid has moderate fouling tendency

STEP 2: CALCULATE HEAT DUTY
Q = ṁ_hot × c_p,hot × (T_hot,in - T_hot,out)
Q = 5 kg/s × 2.5 kJ/kg·K × (150 - 60)°C
Q = 5 × 2.5 × 90 = 1125 kW

STEP 3: DETERMINE COLD WATER FLOW RATE
Q = ṁ_cold × c_p,cold × (T_cold,out - T_cold,in)
1125 kW = ṁ_cold × 4.18 kJ/kg·K × (40 - 20)°C
ṁ_cold = 1125 / (4.18 × 20) = 13.5 kg/s

STEP 4: CALCULATE LOG MEAN TEMPERATURE DIFFERENCE (LMTD)
For counterflow arrangement:
ΔT₁ = T_hot,in - T_cold,out = 150 - 40 = 110°C
ΔT₂ = T_hot,out - T_cold,in = 60 - 20 = 40°C
LMTD = (ΔT₁ - ΔT₂) / ln(ΔT₁/ΔT₂)
LMTD = (110 - 40) / ln(110/40) = 70 / 1.01 = 69.3°C

STEP 5: ESTIMATE OVERALL HEAT TRANSFER COEFFICIENT
For shell-and-tube with water/organic fluid:
- Hot side (organic): h_hot ≈ 500 W/m²·K (forced convection)
- Cold side (water): h_cold ≈ 3000 W/m²·K
- Tube wall (steel): k = 50 W/m·K, thickness 2mm
- Fouling factors: R_f,hot = 0.0003, R_f,cold = 0.0001 m²·K/W

1/U = 1/h_hot + R_f,hot + t/k + R_f,cold + 1/h_cold
1/U = 1/500 + 0.0003 + 0.002/50 + 0.0001 + 1/3000
1/U = 0.002 + 0.0003 + 0.00004 + 0.0001 + 0.00033 = 0.00277
U = 361 W/m²·K

STEP 6: CALCULATE REQUIRED HEAT TRANSFER AREA
Q = U × A × LMTD
A = Q / (U × LMTD)
A = 1,125,000 W / (361 W/m²·K × 69.3 K)
A = 45.0 m²

STEP 7: SIZE THE EXCHANGER
Select tube dimensions: 19.05 mm OD, 16.56 mm ID (3/4" BWG 14)
Tube length: 4.88 m (16 ft standard)
Area per tube: π × 0.01905 × 4.88 = 0.292 m²
Number of tubes: 45.0 / 0.292 = 154 tubes

Shell diameter estimation (for 25% baffle cut, triangular pitch 1.25×OD):
Using tube count correlations: D_shell ≈ 0.45 m (18 inch)

STEP 8: VERIFY PRESSURE DROPS
Tube side (water - 2 passes):
- Velocity: v = ṁ/(ρ × A_flow) = 13.5/(1000 × 77 × π × 0.01656²/4) = 0.82 m/s
- Reynolds: Re = ρvD/μ = 1000 × 0.82 × 0.01656 / 0.001 = 13,600 (turbulent)
- Friction factor: f = 0.079 × Re^(-0.25) = 0.0073
- ΔP_tube = 4f(L/D)(ρv²/2) × 2 passes + minor losses ≈ 15 kPa ✓

Shell side pressure drop requires detailed baffle calculations...

STEP 9: MECHANICAL DESIGN CONSIDERATIONS
- Material selection: Carbon steel (ASME SA-214 tubes)
- Design pressure: 1.5 × operating pressure
- Thermal expansion: Floating head design
- Tube-to-tubesheet joint: Roller expanded with seal weld
- Shell thickness: Per ASME Section VIII

FINAL SPECIFICATION:
- Type: BEM shell-and-tube (TEMA designation)
- Duty: 1125 kW
- Area: 45 m² (with 10% margin: specify 50 m²)
- Tubes: 170 × 19.05mm OD × 4.88m, 2-pass
- Shell: 450mm ID, 4 baffles at 25% cut
- Materials: Carbon steel throughout"""),

    ("Derive the Navier-Stokes equations from first principles.", """DERIVATION OF THE NAVIER-STOKES EQUATIONS

The Navier-Stokes equations describe the motion of viscous fluids. We derive them from conservation principles.

PART 1: CONSERVATION OF MASS (CONTINUITY EQUATION)

Consider a fixed control volume V with surface S. Mass conservation states:
∂/∂t ∫∫∫_V ρ dV + ∫∫_S ρ(v⃗·n̂) dS = 0

Applying the divergence theorem to the surface integral:
∫∫∫_V [∂ρ/∂t + ∇·(ρv⃗)] dV = 0

Since this holds for any arbitrary volume:
∂ρ/∂t + ∇·(ρv⃗) = 0

Expanding the divergence:
∂ρ/∂t + ρ(∇·v⃗) + v⃗·∇ρ = 0

Or using the material derivative Dρ/Dt = ∂ρ/∂t + v⃗·∇ρ:
Dρ/Dt + ρ(∇·v⃗) = 0

For incompressible flow (ρ = constant):
∇·v⃗ = 0

PART 2: CONSERVATION OF MOMENTUM

Newton's second law for a fluid element:
ρ(Dv⃗/Dt) = Σ Forces per unit volume

The material derivative expands to:
Dv⃗/Dt = ∂v⃗/∂t + (v⃗·∇)v⃗

Forces acting on a fluid element:
1. Body forces: f⃗_body = ρg⃗ (gravity)
2. Surface forces: Stress tensor σ

Surface force per unit volume = ∇·σ

The stress tensor for a Newtonian fluid:
σ_ij = -pδ_ij + τ_ij

where p is pressure and τ is the viscous stress tensor.

PART 3: CONSTITUTIVE RELATION FOR NEWTONIAN FLUID

For a Newtonian fluid, viscous stress is proportional to strain rate:
τ_ij = μ(∂u_i/∂x_j + ∂u_j/∂x_i) + λ(∇·v⃗)δ_ij

where:
- μ = dynamic viscosity
- λ = second viscosity coefficient (bulk viscosity)
- For incompressible flow, the λ term vanishes since ∇·v⃗ = 0

PART 4: ASSEMBLING THE MOMENTUM EQUATION

Substituting into momentum conservation:
ρ[∂v⃗/∂t + (v⃗·∇)v⃗] = -∇p + ∇·τ + ρg⃗

For the viscous term (incompressible, constant μ):
∇·τ = μ∇²v⃗

This uses the identity: ∇·[μ(∇v⃗ + (∇v⃗)ᵀ)] = μ∇²v⃗ when ∇·v⃗ = 0

PART 5: THE NAVIER-STOKES EQUATIONS

For incompressible, Newtonian fluid with constant properties:

Continuity:
∇·v⃗ = 0

Momentum (vector form):
ρ[∂v⃗/∂t + (v⃗·∇)v⃗] = -∇p + μ∇²v⃗ + ρg⃗

In Cartesian components (x-momentum):
ρ[∂u/∂t + u(∂u/∂x) + v(∂u/∂y) + w(∂u/∂z)] = -∂p/∂x + μ[∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²] + ρg_x

Similar equations for y and z components.

PART 6: DIMENSIONLESS FORM

Introducing characteristic scales L (length), U (velocity), ρU² (pressure):
x* = x/L, v⃗* = v⃗/U, t* = tU/L, p* = p/(ρU²)

The dimensionless equations become:
∇*·v⃗* = 0

∂v⃗*/∂t* + (v⃗*·∇*)v⃗* = -∇*p* + (1/Re)∇*²v⃗* + (1/Fr²)ĝ

where:
- Reynolds number: Re = ρUL/μ (inertia/viscous forces)
- Froude number: Fr = U/√(gL) (inertia/gravity forces)

These dimensionless groups govern dynamic similarity between flows.

SIGNIFICANCE:
The Navier-Stokes equations are nonlinear PDEs with no general analytical solution. The nonlinearity arises from the convective term (v⃗·∇)v⃗. This is why turbulence remains one of the unsolved problems in classical physics—the equations are deterministic but solutions can be chaotic."""),

    ("Design a complete structural analysis for a cantilever beam under combined loading.", """STRUCTURAL ANALYSIS: CANTILEVER BEAM UNDER COMBINED LOADING

PROBLEM SPECIFICATION:
- Beam length: L = 3 m
- Cross-section: Wide-flange W200×46 (I-beam)
- Material: A992 structural steel (Fy = 345 MPa, E = 200 GPa)
- Loading:
  * Uniformly distributed load: w = 15 kN/m (including self-weight)
  * Concentrated load at tip: P = 25 kN
  * Axial compressive load: N = 100 kN

STEP 1: SECTION PROPERTIES (W200×46)
From steel tables:
- Depth: d = 203 mm
- Flange width: b_f = 203 mm
- Web thickness: t_w = 7.24 mm
- Flange thickness: t_f = 11.0 mm
- Area: A = 5890 mm²
- Moment of inertia: I_x = 45.8 × 10⁶ mm⁴
- Section modulus: S_x = 451 × 10³ mm³
- Plastic section modulus: Z_x = 497 × 10³ mm³
- Radius of gyration: r_x = 88.1 mm, r_y = 51.0 mm

STEP 2: REACTION FORCES AND MOMENTS
At the fixed support (x = 0):

Vertical reaction:
V_A = wL + P = 15(3) + 25 = 70 kN

Moment reaction:
M_A = wL²/2 + PL = 15(3)²/2 + 25(3) = 67.5 + 75 = 142.5 kN·m

Axial reaction:
H_A = N = 100 kN (compression)

STEP 3: INTERNAL FORCE DIAGRAMS

Shear force V(x):
V(x) = V_A - wx = 70 - 15x kN
V(0) = 70 kN, V(3) = 25 kN

Bending moment M(x):
M(x) = M_A - V_A·x + wx²/2 = 142.5 - 70x + 7.5x² kN·m
M(0) = 142.5 kN·m, M(3) = 0

Maximum moment at fixed end: M_max = 142.5 kN·m

STEP 4: STRESS ANALYSIS

Axial stress (uniform across section):
σ_axial = N/A = 100,000 / 5890 = 16.98 MPa (compression)

Maximum bending stress (at extreme fiber):
σ_bending = M_max × c / I_x = M_max / S_x
σ_bending = 142.5 × 10⁶ / (451 × 10³) = 316.0 MPa

Combined stress at extreme fiber (compression side):
σ_max = σ_axial + σ_bending = 16.98 + 316.0 = 333.0 MPa

Check: σ_max = 333 MPa < F_y = 345 MPa ✓ (within elastic range)

Maximum shear stress (at neutral axis):
τ_max = VQ / (I_x × t_w)
where Q = first moment of area above neutral axis

Q = b_f × t_f × (d/2 - t_f/2) + t_w × (d/2 - t_f)² / 2
Q = 203 × 11 × (101.5 - 5.5) + 7.24 × (101.5 - 11)² / 2
Q = 214,368 + 296,516 = 510,884 mm³ = 511 × 10³ mm³

τ_max = 70,000 × 511 × 10³ / (45.8 × 10⁶ × 7.24) = 108 MPa

Shear yield stress: τ_y = 0.6 × F_y = 207 MPa
τ_max = 108 MPa < τ_y ✓

STEP 5: DEFLECTION ANALYSIS

Using superposition for tip deflection:

From distributed load:
δ_w = wL⁴ / (8EI) = 15 × 3000⁴ / (8 × 200,000 × 45.8 × 10⁶)
δ_w = 1.215 × 10¹⁵ / (7.328 × 10¹³) = 16.6 mm

From concentrated load:
δ_P = PL³ / (3EI) = 25,000 × 3000³ / (3 × 200,000 × 45.8 × 10⁶)
δ_P = 6.75 × 10¹⁴ / (2.748 × 10¹³) = 24.6 mm

Total tip deflection:
δ_total = δ_w + δ_P = 16.6 + 24.6 = 41.2 mm

Deflection limit (typical L/180 for serviceability):
δ_allowable = 3000/180 = 16.7 mm

δ_total = 41.2 mm > δ_allowable ✓✗ (FAILS serviceability)

STEP 6: STABILITY CHECK (LATERAL-TORSIONAL BUCKLING)

For cantilever with tip load, effective length factor K ≈ 2.0
Unbraced length: L_b = 3000 mm

Limiting lengths (AISC):
L_p = 1.76 × r_y × √(E/F_y) = 1.76 × 51 × √(200,000/345) = 2162 mm
L_r = computed from AISC equations ≈ 6500 mm

Since L_p < L_b < L_r, inelastic LTB governs.

Nominal moment capacity (reduced for LTB):
M_n = C_b × [M_p - (M_p - 0.7F_y×S_x) × (L_b - L_p)/(L_r - L_p)]

For cantilever with uniform moment gradient, C_b ≈ 1.0
M_p = F_y × Z_x = 345 × 497 × 10³ = 171.5 kN·m

M_n = 1.0 × [171.5 - (171.5 - 0.7×345×451×10³/10⁶) × (3000-2162)/(6500-2162)]
M_n = 171.5 - (171.5 - 109.0) × 0.193 = 159.4 kN·m

LRFD capacity: φM_n = 0.9 × 159.4 = 143.5 kN·m
M_max = 142.5 kN·m < φM_n = 143.5 kN·m ✓ (barely adequate)

STEP 7: DESIGN RECOMMENDATIONS

1. STRENGTH: Adequate for combined axial and bending (utilization 97%)
2. DEFLECTION: Exceeds limit—consider:
   - Increase section to W250×58 (I = 87.3 × 10⁶ mm⁴)
   - Add intermediate support
   - Pre-camber beam 20mm upward
3. STABILITY: Marginal LTB capacity—add lateral bracing at midspan

CONCLUSION:
The W200×46 section is adequate for strength but fails serviceability deflection limits. Recommend upgrading to W250×58 or adding lateral bracing and pre-camber."""),

    ("Explain the complete thermodynamic analysis of a Rankine power cycle with reheat.", """THERMODYNAMIC ANALYSIS: RANKINE CYCLE WITH REHEAT

PROBLEM SPECIFICATION:
- Working fluid: Water/steam
- Boiler pressure: P₁ = 10 MPa
- Condenser pressure: P₄ = 10 kPa
- Reheat pressure: P_reheat = 2 MPa
- Turbine inlet temperature: T₁ = 500°C
- Reheat temperature: T₃ = 500°C
- Net power output: Ẇ_net = 100 MW
- Pump and turbine isentropic efficiencies: η_pump = 0.85, η_turbine = 0.88

STEP 1: STATE POINT IDENTIFICATION

State 1: Superheated vapor entering HP turbine (10 MPa, 500°C)
State 2s: Isentropic expansion to reheat pressure (2 MPa)
State 2: Actual HP turbine exit
State 3: Superheated vapor entering LP turbine (2 MPa, 500°C)
State 4s: Isentropic expansion to condenser pressure (10 kPa)
State 4: Actual LP turbine exit
State 5: Saturated liquid leaving condenser (10 kPa)
State 6s: Isentropic compression to boiler pressure
State 6: Actual pump exit (10 MPa)

STEP 2: PROPERTY DETERMINATION

State 1 (10 MPa, 500°C) - From steam tables:
h₁ = 3373.7 kJ/kg
s₁ = 6.5966 kJ/kg·K

State 2s (Isentropic to 2 MPa, s₂s = s₁ = 6.5966):
At 2 MPa: s_f = 2.4474, s_fg = 4.1014, s_g = 6.5488 kJ/kg·K
Since s₂s > s_g, state 2s is superheated

Interpolating in superheated tables at 2 MPa:
T₂s ≈ 285°C, h₂s = 2996.5 kJ/kg

State 2 (Actual HP turbine exit):
η_HP = (h₁ - h₂)/(h₁ - h₂s) = 0.88
h₂ = h₁ - η_HP(h₁ - h₂s) = 3373.7 - 0.88(3373.7 - 2996.5)
h₂ = 3373.7 - 332.0 = 3041.7 kJ/kg

State 3 (2 MPa, 500°C) - Reheat:
h₃ = 3467.6 kJ/kg
s₃ = 7.4317 kJ/kg·K

State 4s (Isentropic to 10 kPa, s₄s = s₃ = 7.4317):
At 10 kPa: s_f = 0.6493, s_fg = 7.5009 kJ/kg·K
x₄s = (s₄s - s_f)/s_fg = (7.4317 - 0.6493)/7.5009 = 0.904
h_f = 191.83, h_fg = 2392.8 kJ/kg
h₄s = h_f + x₄s × h_fg = 191.83 + 0.904 × 2392.8 = 2354.9 kJ/kg

State 4 (Actual LP turbine exit):
η_LP = (h₃ - h₄)/(h₃ - h₄s) = 0.88
h₄ = h₃ - η_LP(h₃ - h₄s) = 3467.6 - 0.88(3467.6 - 2354.9)
h₄ = 3467.6 - 979.2 = 2488.4 kJ/kg

Quality check at state 4:
x₄ = (h₄ - h_f)/h_fg = (2488.4 - 191.83)/2392.8 = 0.960
Steam quality 96.0%—acceptable (>88% avoids blade erosion)

State 5 (Saturated liquid at 10 kPa):
h₅ = h_f = 191.83 kJ/kg
v₅ = v_f = 0.00101 m³/kg

State 6 (Pump exit at 10 MPa):
w_pump,s = v₅(P₆ - P₅) = 0.00101(10,000 - 10) = 10.09 kJ/kg
w_pump = w_pump,s/η_pump = 10.09/0.85 = 11.87 kJ/kg
h₆ = h₅ + w_pump = 191.83 + 11.87 = 203.70 kJ/kg

STEP 3: ENERGY ANALYSIS

HP Turbine work:
w_HP = h₁ - h₂ = 3373.7 - 3041.7 = 332.0 kJ/kg

LP Turbine work:
w_LP = h₃ - h₄ = 3467.6 - 2488.4 = 979.2 kJ/kg

Total turbine work:
w_turbine = w_HP + w_LP = 332.0 + 979.2 = 1311.2 kJ/kg

Pump work:
w_pump = 11.87 kJ/kg

Net work output:
w_net = w_turbine - w_pump = 1311.2 - 11.87 = 1299.3 kJ/kg

Heat input (boiler + reheater):
q_in = (h₁ - h₆) + (h₃ - h₂)
q_in = (3373.7 - 203.70) + (3467.6 - 3041.7)
q_in = 3170.0 + 425.9 = 3595.9 kJ/kg

Heat rejected (condenser):
q_out = h₄ - h₅ = 2488.4 - 191.83 = 2296.6 kJ/kg

STEP 4: EFFICIENCY CALCULATIONS

Thermal efficiency:
η_th = w_net/q_in = 1299.3/3595.9 = 0.3613 = 36.13%

Comparison with simple Rankine (no reheat, same pressures):
Without reheat, expansion 10 MPa → 10 kPa:
x₄_simple ≈ 0.82 (excessive moisture!)
η_simple ≈ 33%

Reheat benefit: +3% efficiency, +14% quality improvement

STEP 5: MASS FLOW RATE AND COMPONENT SIZING

For 100 MW net output:
ṁ = Ẇ_net/w_net = 100,000 kW / 1299.3 kJ/kg = 77.0 kg/s

Component heat rates:
Q̇_boiler = ṁ(h₁ - h₆) = 77.0 × 3170.0 = 244.1 MW
Q̇_reheater = ṁ(h₃ - h₂) = 77.0 × 425.9 = 32.8 MW
Q̇_condenser = ṁ(h₄ - h₅) = 77.0 × 2296.6 = 176.8 MW
Ẇ_pump = ṁ × w_pump = 77.0 × 11.87 = 0.91 MW

Energy balance check:
Q̇_in = 244.1 + 32.8 = 276.9 MW
Q̇_out + Ẇ_net = 176.8 + 100.0 = 276.8 MW ✓

STEP 6: SECOND LAW ANALYSIS (EXERGY)

Dead state: T₀ = 25°C = 298 K, P₀ = 100 kPa
h₀ = 104.89 kJ/kg, s₀ = 0.3674 kJ/kg·K

Specific exergy at each state:
ψᵢ = (hᵢ - h₀) - T₀(sᵢ - s₀)

ψ₁ = (3373.7 - 104.89) - 298(6.5966 - 0.3674) = 3268.8 - 1856.3 = 1412.5 kJ/kg
ψ₃ = (3467.6 - 104.89) - 298(7.4317 - 0.3674) = 3362.7 - 2105.2 = 1257.5 kJ/kg
ψ₄ = (2488.4 - 104.89) - 298(7.6681 - 0.3674) = 2383.5 - 2175.6 = 207.9 kJ/kg

Exergy destruction rates:
HP Turbine: Ẋ_dest,HP = ṁT₀(s₂ - s₁) = 77 × 298 × (6.789 - 6.597) = 4.4 MW
LP Turbine: Ẋ_dest,LP = ṁT₀(s₄ - s₃) = 77 × 298 × (7.668 - 7.432) = 5.4 MW
Condenser: Ẋ_dest,cond = ṁT₀(s₅ - s₄) + Q̇_cond/T₀ ≈ -21.3 + 0 = significant

Second law efficiency:
η_II = Ẇ_net / Ẋ_in = 100 / (ṁ × ψ_fuel,in) ≈ 0.55

SUMMARY OF RESULTS:
- Thermal efficiency: 36.13%
- Steam flow rate: 77.0 kg/s
- Turbine exit quality: 96.0% (acceptable)
- Total heat input: 276.9 MW
- Condenser heat rejection: 176.8 MW
- Major irreversibilities: Boiler combustion, turbine expansion, condenser"""),

    ("Derive Maxwell's equations from experimental laws and explain their physical meaning.", """DERIVATION AND PHYSICAL INTERPRETATION OF MAXWELL'S EQUATIONS

Maxwell's equations unify electricity, magnetism, and optics into a single theoretical framework. We derive each from experimental foundations.

PART 1: GAUSS'S LAW FOR ELECTRICITY

Experimental basis: Coulomb's Law
F⃗ = (1/4πε₀) × (q₁q₂/r²) r̂

Electric field from point charge:
E⃗ = (1/4πε₀) × (q/r²) r̂

Consider electric flux through closed surface around charge q:
Φ_E = ∮ E⃗·dA⃗

For a spherical surface of radius r:
Φ_E = E × 4πr² = (q/4πε₀r²) × 4πr² = q/ε₀

Generalizing to any closed surface (using solid angle arguments):
∮ E⃗·dA⃗ = Q_enclosed/ε₀

This is Gauss's Law in integral form.

Applying the divergence theorem:
∮ E⃗·dA⃗ = ∫∫∫ ∇·E⃗ dV = ∫∫∫ (ρ/ε₀) dV

Therefore, in differential form:
∇·E⃗ = ρ/ε₀

PHYSICAL MEANING: Electric field lines originate from positive charges and terminate on negative charges. The divergence of E⃗ at any point equals the charge density divided by ε₀. Electric charges are sources/sinks of the electric field.

PART 2: GAUSS'S LAW FOR MAGNETISM

Experimental basis: No magnetic monopoles have ever been observed.

All magnetic field lines form closed loops (no beginning or end).

For any closed surface:
∮ B⃗·dA⃗ = 0

Applying divergence theorem:
∫∫∫ ∇·B⃗ dV = 0

Since this holds for any volume:
∇·B⃗ = 0

PHYSICAL MEANING: Magnetic field lines always form closed loops—they never start or end. There are no magnetic monopoles (isolated north or south poles). Every magnetic north pole is accompanied by a south pole.

PART 3: FARADAY'S LAW OF INDUCTION

Experimental basis: Faraday's experiments (1831)
A changing magnetic flux through a circuit induces an EMF.

EMF = -dΦ_B/dt

where Φ_B = ∫∫ B⃗·dA⃗ is magnetic flux through the circuit.

The EMF around a closed loop equals the circulation of E⃗:
∮ E⃗·dl⃗ = EMF = -dΦ_B/dt = -d/dt ∫∫ B⃗·dA⃗

For a fixed loop:
∮ E⃗·dl⃗ = -∫∫ (∂B⃗/∂t)·dA⃗

Applying Stokes' theorem to the left side:
∫∫ (∇×E⃗)·dA⃗ = -∫∫ (∂B⃗/∂t)·dA⃗

Since this holds for any surface:
∇×E⃗ = -∂B⃗/∂t

PHYSICAL MEANING: A changing magnetic field creates a circulating electric field. The curl of E⃗ equals the negative time rate of change of B⃗. This is the principle behind transformers, generators, and induction.

PART 4: AMPÈRE-MAXWELL LAW

Experimental basis: Ampère's Law (1820s)
A steady current I creates a magnetic field circulating around it.

∮ B⃗·dl⃗ = μ₀I_enclosed = μ₀ ∫∫ J⃗·dA⃗

Maxwell's contribution: The displacement current

Problem with original Ampère's Law: Consider a charging capacitor.
Current I flows in the wire, but no current crosses between the plates.
Applying Ampère's Law to two surfaces bounded by the same loop gives different answers—a contradiction!

Maxwell resolved this by adding the displacement current:
J⃗_d = ε₀ ∂E⃗/∂t

Between capacitor plates, E changes with time, so J_d ≠ 0 even though J = 0.

The complete Ampère-Maxwell Law:
∮ B⃗·dl⃗ = μ₀(I + ε₀ dΦ_E/dt) = μ₀ ∫∫ (J⃗ + ε₀ ∂E⃗/∂t)·dA⃗

Applying Stokes' theorem:
∇×B⃗ = μ₀J⃗ + μ₀ε₀ ∂E⃗/∂t

PHYSICAL MEANING: Magnetic fields are created by electric currents AND by changing electric fields. The displacement current term is essential for consistency and leads to electromagnetic waves.

PART 5: MAXWELL'S EQUATIONS (COMPLETE SET)

Differential form (SI units):
1. ∇·E⃗ = ρ/ε₀ (Gauss - electric)
2. ∇·B⃗ = 0 (Gauss - magnetic)
3. ∇×E⃗ = -∂B⃗/∂t (Faraday)
4. ∇×B⃗ = μ₀J⃗ + μ₀ε₀ ∂E⃗/∂t (Ampère-Maxwell)

Integral form:
1. ∮ E⃗·dA⃗ = Q/ε₀
2. ∮ B⃗·dA⃗ = 0
3. ∮ E⃗·dl⃗ = -dΦ_B/dt
4. ∮ B⃗·dl⃗ = μ₀(I + ε₀ dΦ_E/dt)

PART 6: ELECTROMAGNETIC WAVES

In vacuum (ρ = 0, J⃗ = 0), Maxwell's equations become:
∇·E⃗ = 0, ∇·B⃗ = 0
∇×E⃗ = -∂B⃗/∂t
∇×B⃗ = μ₀ε₀ ∂E⃗/∂t

Taking curl of Faraday's Law:
∇×(∇×E⃗) = -∂/∂t(∇×B⃗) = -μ₀ε₀ ∂²E⃗/∂t²

Using vector identity ∇×(∇×E⃗) = ∇(∇·E⃗) - ∇²E⃗ = -∇²E⃗:
∇²E⃗ = μ₀ε₀ ∂²E⃗/∂t²

This is the wave equation with speed:
c = 1/√(μ₀ε₀) = 2.998 × 10⁸ m/s

Maxwell calculated this and found it matched the measured speed of light—his greatest insight: LIGHT IS AN ELECTROMAGNETIC WAVE.

SIGNIFICANCE:
Maxwell's equations:
- Unify electricity, magnetism, and optics
- Predict electromagnetic waves of all frequencies
- Are Lorentz invariant (led to special relativity)
- Form the foundation of all electromagnetic technology
- Represent one of the greatest achievements in physics"""),

    ("Design a PID controller for a temperature control system with complete tuning analysis.", """PID CONTROLLER DESIGN FOR TEMPERATURE CONTROL SYSTEM

SYSTEM SPECIFICATION:
- Process: Industrial heating furnace
- Controlled variable: Chamber temperature
- Setpoint: 500°C
- Disturbances: Door openings, ambient temperature changes
- Actuator: Electric heating elements (0-100% power)
- Sensor: Type K thermocouple with transmitter (4-20 mA)
- Desired performance: <2% overshoot, settling time <10 minutes

STEP 1: PROCESS IDENTIFICATION

Open-loop step response test performed:
- Initial temperature: 200°C (steady state with 30% heater power)
- Step input: Increase heater to 50% power
- Final temperature: 350°C (steady state)

Recorded response data analysis:
- Process gain: K_p = ΔT/Δu = (350-200)/(50-30) = 7.5 °C/%
- Time delay (dead time): L = 45 seconds
- Time constant: τ = 180 seconds (63.2% of final value)

First-Order Plus Dead Time (FOPDT) model:
G_p(s) = K_p × e^(-Ls) / (τs + 1)
G_p(s) = 7.5 × e^(-45s) / (180s + 1)

Process characteristics:
- Dead time ratio: L/τ = 45/180 = 0.25 (moderate dead time)
- Controllability index: τ/L = 4.0 (good controllability)

STEP 2: PID CONTROLLER STRUCTURE

Standard PID (ISA form):
u(t) = K_c [e(t) + (1/T_i)∫e(t)dt + T_d × de(t)/dt]

Transfer function:
G_c(s) = K_c (1 + 1/(T_i×s) + T_d×s)

Where:
- K_c = Controller gain (proportional)
- T_i = Integral time (seconds)
- T_d = Derivative time (seconds)

STEP 3: INITIAL TUNING - ZIEGLER-NICHOLS METHOD

Method 1: Open-loop (Process Reaction Curve)
Using FOPDT parameters:
K_c = 1.2τ/(K_p×L) = 1.2×180/(7.5×45) = 0.64
T_i = 2L = 90 seconds
T_d = 0.5L = 22.5 seconds

Method 2: Closed-loop (Ultimate Gain)
Increase K_c with T_i = ∞, T_d = 0 until sustained oscillation.
Ultimate gain: K_u = 2.8 (measured)
Ultimate period: P_u = 120 seconds (measured)

ZN closed-loop tuning:
K_c = 0.6×K_u = 0.6×2.8 = 1.68
T_i = P_u/2 = 60 seconds
T_d = P_u/8 = 15 seconds

STEP 4: REFINED TUNING - IMC METHOD

Internal Model Control provides better performance with less overshoot.

Choose closed-loop time constant: τ_c = max(0.1τ, 0.8L) = max(18, 36) = 36 s

IMC-PID tuning rules for FOPDT:
K_c = τ / (K_p × (τ_c + L)) = 180 / (7.5 × (36 + 45)) = 0.296
T_i = τ = 180 seconds
T_d = L/2 = 22.5 seconds

STEP 5: SIMULATION AND COMPARISON

Simulating closed-loop response to setpoint change (200°C → 500°C):

ZN Open-Loop Settings (K_c=0.64, T_i=90, T_d=22.5):
- Rise time: 180 s
- Overshoot: 45% (unacceptable!)
- Settling time: 900 s
- Oscillatory response

ZN Closed-Loop Settings (K_c=1.68, T_i=60, T_d=15):
- Rise time: 120 s
- Overshoot: 55% (worse!)
- Settling time: 1200 s
- More aggressive oscillation

IMC Settings (K_c=0.296, T_i=180, T_d=22.5):
- Rise time: 250 s
- Overshoot: 5%
- Settling time: 450 s
- Smooth response

STEP 6: FINE-TUNING FOR SPECIFICATIONS

Target: <2% overshoot, settling time <600s

Adjust IMC time constant to τ_c = 50s (more conservative):
K_c = 180 / (7.5 × 95) = 0.253
T_i = 180 seconds
T_d = 22.5 seconds

Simulation results:
- Rise time: 280 s
- Overshoot: 1.8% ✓
- Settling time: 520 s ✓
- Smooth, well-damped response

STEP 7: ROBUSTNESS ANALYSIS

Gain margin and phase margin (frequency domain):

Open-loop transfer function:
G_OL(s) = G_c(s) × G_p(s)

At crossover frequency ω_c where |G_OL(jω_c)| = 1:
Using numerical analysis: ω_c = 0.015 rad/s

Phase at crossover:
∠G_OL(jω_c) = -135° - 45×0.015×(180/π)° = -135° - 39° = -174°
Phase margin = 180° - 174° = 6° (marginal!)

Gain at phase = -180°:
ω_180 ≈ 0.018 rad/s
|G_OL(jω_180)| = 0.7
Gain margin = 1/0.7 = 1.43 = 3.1 dB (low)

To improve robustness, reduce K_c to 0.20:
New phase margin: 25°
New gain margin: 6 dB
Overshoot increases to 3%, still acceptable

STEP 8: PRACTICAL IMPLEMENTATION CONSIDERATIONS

Derivative filter:
Add filter to derivative term to reduce noise amplification:
G_d(s) = T_d×s / (1 + (T_d/N)×s), where N = 10

Anti-windup:
Limit integral accumulation when output saturates:
- Tracking anti-windup: Reset integral when output at limit
- Conditional integration: Stop integrating when output saturated

Setpoint weighting:
Reduce overshoot on setpoint changes:
- Apply proportional only to error: b×SP - PV, where b = 0.5
- Apply derivative only to PV (not error) to avoid derivative kick

Sampling and scan rate:
- Sample time: T_s ≤ T_d/10 = 2.25 seconds
- Recommend T_s = 1 second for this process

Output limits:
- Heater power: 0-100%
- Rate limit: 5%/second to protect heating elements

FINAL CONTROLLER PARAMETERS:
- K_c = 0.20 (dimensionless, with normalized I/O)
- T_i = 180 seconds
- T_d = 22.5 seconds
- Derivative filter N = 10
- Setpoint weight b = 0.5
- Sample time = 1 second
- Anti-windup: Tracking with back-calculation

IMPLEMENTATION (Pseudo-code):
```
// Discrete PID with anti-windup
error = SP - PV
P_term = Kc * (b*SP - PV)
I_term = I_term + Kc*Ts/Ti * error + Ts/Tt * (output_limited - output)
D_term = Td/(Td + N*Ts) * D_term_prev - Kc*Td*N/(Td + N*Ts) * (PV - PV_prev)
output = P_term + I_term + D_term
output_limited = clamp(output, 0, 100)
```

This design provides robust temperature control meeting all specifications."""),
]


def get_general_training_data() -> List[Tuple[str, str]]:
    """
    Return comprehensive general training data.
    Combines all categories with appropriate weighting.
    """
    all_data = []

    # Identity - high weight for consistent persona
    all_data.extend(IDENTITY * 100)

    # Conversational skills - very high weight
    all_data.extend(CONVERSATION * 50)

    # Knowledge - medium-high weight
    all_data.extend(KNOWLEDGE * 30)

    # Physics - very high weight for core principles
    all_data.extend(PHYSICS * 80)

    # Truthfulness - very high weight for honest, accurate responses
    all_data.extend(TRUTHFULNESS * 100)

    # Engineering - high weight for technical knowledge
    all_data.extend(ENGINEERING * 40)

    # Science - high weight for scientific literacy
    all_data.extend(SCIENCE * 40)

    # Mathematics - high weight for mathematical competence
    all_data.extend(MATHEMATICS * 50)

    # Computer Science - very high weight for CS fundamentals
    all_data.extend(COMPUTER_SCIENCE * 60)

    # US Legal System - high weight for legal literacy
    all_data.extend(US_LEGAL_SYSTEM * 50)

    # American Life - high weight for practical knowledge
    all_data.extend(AMERICAN_LIFE * 50)

    # Reasoning - high weight for intelligence
    all_data.extend(REASONING * 40)

    # Coding - high weight for technical capability
    all_data.extend(CODING * 35)

    # Math - medium-high weight
    all_data.extend(MATH * 30)

    # Writing - medium weight
    all_data.extend(WRITING * 25)

    # Professional - medium weight
    all_data.extend(PROFESSIONAL * 20)

    # Creative - lower weight but important for variety
    all_data.extend(CREATIVE * 15)

    # Practical advice - medium weight
    all_data.extend(PRACTICAL * 25)

    # Military and Intelligence Operations - high weight for specialized knowledge
    all_data.extend(MILITARY_INTEL_OPS * 60)

    # Complex STEM problem solving - very high weight for advanced technical capability
    all_data.extend(COMPLEX_STEM * 80)

    return all_data


def get_data_stats():
    """Print statistics about the training data."""
    categories = {
        'Identity': IDENTITY,
        'Conversation': CONVERSATION,
        'Knowledge': KNOWLEDGE,
        'Physics': PHYSICS,
        'Truthfulness': TRUTHFULNESS,
        'Engineering': ENGINEERING,
        'Science': SCIENCE,
        'Mathematics': MATHEMATICS,
        'Computer Science': COMPUTER_SCIENCE,
        'US Legal System': US_LEGAL_SYSTEM,
        'American Life': AMERICAN_LIFE,
        'Reasoning': REASONING,
        'Coding': CODING,
        'Math': MATH,
        'Writing': WRITING,
        'Professional': PROFESSIONAL,
        'Creative': CREATIVE,
        'Practical': PRACTICAL,
        'Military Intel Ops': MILITARY_INTEL_OPS,
        'Complex STEM': COMPLEX_STEM,
    }

    print("General Training Data Statistics:")
    print("-" * 40)
    total = 0
    for name, data in categories.items():
        print(f"  {name}: {len(data)} pairs")
        total += len(data)
    print("-" * 40)
    print(f"  Total unique pairs: {total}")

    full_data = get_general_training_data()
    print(f"  Total with weights: {len(full_data)}")


if __name__ == "__main__":
    get_data_stats()
