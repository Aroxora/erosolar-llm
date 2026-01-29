#!/usr/bin/env python3
"""
CONCEPT HARVESTER
=================
Extracts foundational concepts for AI training using a comprehensive taxonomy.
A general-purpose AI needs deep understanding across ALL domains of human knowledge.

Output: optional_unverified_concepts/foundational_concepts.json as {concept: training_value} dict
"""

import json
import os
import sys
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, Set, List, Optional
from collections import Counter
import re

# Import base data
try:
    from data import get_all_training_data
except ImportError:
    sys.path.append(os.getcwd())
    from data import get_all_training_data

OUTPUT_FILE = "optional_unverified_concepts/foundational_concepts.json"
LEXICON_SAMPLE_FILE = "optional_unverified_concepts/lexicon_sample.json"

# =============================================================================
# COMPREHENSIVE FOUNDATIONAL KNOWLEDGE TAXONOMY
# A general-purpose AI needs understanding across ALL human knowledge domains
# =============================================================================

FOUNDATIONAL_TAXONOMY = {
    # =========================================================================
    # REASONING, LOGIC & EPISTEMOLOGY
    # =========================================================================
    "Logic & Reasoning": [
        "Deductive Reasoning", "Inductive Reasoning", "Abductive Reasoning",
        "Logical Fallacies", "Syllogism", "Propositional Logic", "Predicate Logic",
        "Boolean Logic", "Modus Ponens", "Modus Tollens", "Proof by Contradiction",
        "Mathematical Induction", "Causality", "Correlation vs Causation",
        "Counterfactual Reasoning", "Analogical Reasoning", "Modal Logic",
        "Fuzzy Logic", "Probabilistic Reasoning", "Bayesian Reasoning",
        "Heuristics", "Cognitive Biases", "Critical Thinking", "Metacognition",
        "Formal Verification", "Logical Consistency", "Soundness", "Completeness",
    ],

    "Epistemology": [
        "Knowledge", "Belief", "Justification", "Truth", "Evidence",
        "Skepticism", "Empiricism", "Rationalism", "Foundationalism",
        "Coherentism", "Reliabilism", "Virtue Epistemology", "Social Epistemology",
        "Scientific Method", "Hypothesis Testing", "Falsifiability", "Paradigm Shift",
        "Confirmation Bias", "Epistemic Humility", "Uncertainty Quantification",
    ],

    # =========================================================================
    # MATHEMATICS
    # =========================================================================
    "Arithmetic & Number Theory": [
        "Arithmetic", "Prime Numbers", "Factorization", "Divisibility",
        "Modular Arithmetic", "Number Systems", "Integers", "Rational Numbers",
        "Real Numbers", "Complex Numbers", "Infinity", "Cardinality",
        "Diophantine Equations", "Cryptographic Number Theory",
    ],

    "Algebra": [
        "Algebra", "Linear Algebra", "Abstract Algebra", "Group Theory",
        "Ring Theory", "Field Theory", "Vector Spaces", "Matrices",
        "Eigenvalues", "Eigenvectors", "Linear Transformations",
        "Polynomials", "Equations", "Inequalities", "Systems of Equations",
    ],

    "Calculus & Analysis": [
        "Calculus", "Limits", "Derivatives", "Integrals", "Differential Equations",
        "Partial Derivatives", "Multivariable Calculus", "Vector Calculus",
        "Real Analysis", "Complex Analysis", "Functional Analysis",
        "Measure Theory", "Fourier Analysis", "Laplace Transform",
        "Taylor Series", "Convergence", "Continuity",
    ],

    "Geometry & Topology": [
        "Geometry", "Euclidean Geometry", "Non-Euclidean Geometry",
        "Trigonometry", "Analytic Geometry", "Differential Geometry",
        "Topology", "Manifolds", "Metric Spaces", "Homeomorphism",
        "Fractals", "Symmetry", "Transformations",
    ],

    "Discrete Mathematics": [
        "Discrete Mathematics", "Combinatorics", "Graph Theory", "Set Theory",
        "Relations", "Functions", "Permutations", "Combinations",
        "Recurrence Relations", "Generating Functions", "Pigeonhole Principle",
        "Boolean Algebra", "Lattice Theory",
    ],

    "Probability & Statistics": [
        "Probability", "Statistics", "Random Variables", "Distributions",
        "Expected Value", "Variance", "Standard Deviation", "Correlation",
        "Regression", "Hypothesis Testing", "Confidence Intervals",
        "Bayesian Statistics", "Frequentist Statistics", "Statistical Inference",
        "Sampling", "Central Limit Theorem", "Law of Large Numbers",
        "Monte Carlo Methods", "Markov Chains", "Stochastic Processes",
    ],

    "Applied Mathematics": [
        "Optimization", "Linear Programming", "Convex Optimization",
        "Numerical Methods", "Numerical Analysis", "Approximation Theory",
        "Mathematical Modeling", "Simulation", "Game Theory",
        "Decision Theory", "Operations Research", "Queueing Theory",
        "Information Theory", "Coding Theory", "Signal Processing",
    ],

    # =========================================================================
    # COMPUTER SCIENCE
    # =========================================================================
    "Algorithms & Data Structures": [
        "Algorithm", "Data Structure", "Time Complexity", "Space Complexity",
        "Big O Notation", "Recursion", "Iteration", "Divide and Conquer",
        "Dynamic Programming", "Greedy Algorithms", "Backtracking",
        "Binary Search", "Sorting Algorithms", "Graph Algorithms",
        "Tree Traversal", "Hash Tables", "Heaps", "Stacks", "Queues",
        "Linked Lists", "Arrays", "Trees", "Graphs", "Tries",
        "Balanced Trees", "B-Trees", "Red-Black Trees",
    ],

    "Theory of Computation": [
        "Computability", "Turing Machine", "Finite Automata", "Pushdown Automata",
        "Regular Languages", "Context-Free Languages", "Chomsky Hierarchy",
        "Halting Problem", "Decidability", "Complexity Classes",
        "P vs NP", "NP-Complete", "NP-Hard", "Polynomial Time",
        "Exponential Time", "Space Complexity Classes",
    ],

    "Programming Paradigms": [
        "Imperative Programming", "Declarative Programming", "Object-Oriented Programming",
        "Functional Programming", "Logic Programming", "Procedural Programming",
        "Event-Driven Programming", "Concurrent Programming", "Parallel Programming",
        "Reactive Programming", "Aspect-Oriented Programming",
    ],

    "Programming Languages": [
        "Python", "JavaScript", "Java", "C", "C++", "Rust", "Go", "TypeScript",
        "Ruby", "PHP", "Swift", "Kotlin", "Scala", "Haskell", "Lisp", "Prolog",
        "SQL", "R", "MATLAB", "Julia", "Perl", "Shell Scripting",
        "Assembly Language", "WebAssembly",
    ],

    "Programming Concepts": [
        "Variable", "Function", "Class", "Object", "Method", "Inheritance",
        "Polymorphism", "Encapsulation", "Abstraction", "Interface",
        "Exception Handling", "Error Handling", "Debugging", "Testing",
        "Unit Testing", "Integration Testing", "Test-Driven Development",
        "Refactoring", "Code Review", "Version Control", "Git",
        "Design Patterns", "SOLID Principles", "Clean Code",
        "Memory Management", "Garbage Collection", "Pointers", "References",
        "Type Systems", "Static Typing", "Dynamic Typing", "Type Inference",
    ],

    "Software Engineering": [
        "Software Development Lifecycle", "Agile Methodology", "Scrum", "Kanban",
        "Waterfall Model", "DevOps", "CI/CD", "Continuous Integration",
        "Continuous Deployment", "Microservices", "Monolithic Architecture",
        "Service-Oriented Architecture", "API Design", "REST", "GraphQL",
        "Documentation", "Technical Writing", "Requirements Engineering",
        "Software Architecture", "System Design", "Scalability",
    ],

    "Databases": [
        "Database", "Relational Database", "SQL", "NoSQL", "ACID Properties",
        "Transactions", "Normalization", "Denormalization", "Indexing",
        "Query Optimization", "Joins", "Views", "Stored Procedures",
        "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Cassandra",
        "Elasticsearch", "Graph Databases", "Time-Series Databases",
        "Data Warehousing", "Data Lakes", "ETL",
    ],

    "Systems & Infrastructure": [
        "Operating System", "Kernel", "Process", "Thread", "Scheduling",
        "Memory Management", "Virtual Memory", "File System", "I/O",
        "Linux", "Unix", "Windows", "macOS", "Android", "iOS",
        "Virtualization", "Hypervisor", "Container", "Docker", "Kubernetes",
        "Cloud Computing", "AWS", "Azure", "Google Cloud", "Serverless",
        "Load Balancing", "Caching", "CDN", "Proxy", "Reverse Proxy",
    ],

    "Networking": [
        "Computer Networks", "OSI Model", "TCP/IP", "HTTP", "HTTPS",
        "DNS", "DHCP", "NAT", "Firewall", "Router", "Switch",
        "IP Address", "Subnet", "VLAN", "VPN", "SSL/TLS",
        "WebSocket", "gRPC", "REST API", "SOAP", "RPC",
        "Bandwidth", "Latency", "Throughput", "Packet", "Protocol",
    ],

    "Security": [
        "Cybersecurity", "Information Security", "Network Security",
        "Application Security", "Cryptography", "Encryption", "Decryption",
        "Symmetric Encryption", "Asymmetric Encryption", "Hashing",
        "Digital Signatures", "Certificates", "PKI", "SSL/TLS",
        "Authentication", "Authorization", "Access Control", "IAM",
        "OAuth", "JWT", "Session Management", "Password Security",
        "Vulnerability", "Exploit", "Penetration Testing", "Security Audit",
        "Threat Modeling", "Risk Assessment", "Incident Response",
        "Malware", "Phishing", "Social Engineering", "Zero Trust",
    ],

    "Web Development": [
        "HTML", "CSS", "JavaScript", "DOM", "Web Browser",
        "Frontend Development", "Backend Development", "Full Stack",
        "React", "Vue", "Angular", "Node.js", "Express",
        "Web APIs", "AJAX", "Fetch API", "WebSocket",
        "Responsive Design", "Accessibility", "SEO", "Web Performance",
        "Progressive Web Apps", "Single Page Applications",
    ],

    # =========================================================================
    # ARTIFICIAL INTELLIGENCE & MACHINE LEARNING
    # =========================================================================
    "Machine Learning Fundamentals": [
        "Machine Learning", "Supervised Learning", "Unsupervised Learning",
        "Semi-Supervised Learning", "Reinforcement Learning", "Self-Supervised Learning",
        "Training", "Inference", "Model", "Features", "Labels",
        "Training Data", "Test Data", "Validation Data", "Cross-Validation",
        "Overfitting", "Underfitting", "Bias-Variance Tradeoff",
        "Regularization", "L1 Regularization", "L2 Regularization", "Dropout",
        "Hyperparameters", "Hyperparameter Tuning", "Grid Search", "Random Search",
    ],

    "Neural Networks": [
        "Neural Network", "Artificial Neuron", "Perceptron", "Activation Function",
        "Sigmoid", "ReLU", "Tanh", "Softmax", "Layer", "Hidden Layer",
        "Feedforward Network", "Backpropagation", "Gradient Descent",
        "Stochastic Gradient Descent", "Adam Optimizer", "Learning Rate",
        "Loss Function", "Cross-Entropy Loss", "Mean Squared Error",
        "Batch Normalization", "Weight Initialization",
    ],

    "Deep Learning": [
        "Deep Learning", "Convolutional Neural Network", "CNN",
        "Recurrent Neural Network", "RNN", "LSTM", "GRU",
        "Transformer", "Attention Mechanism", "Self-Attention",
        "Multi-Head Attention", "Positional Encoding", "Encoder-Decoder",
        "Autoencoder", "Variational Autoencoder", "GAN", "Diffusion Models",
        "ResNet", "VGG", "Inception", "EfficientNet", "BERT", "GPT",
        "Transfer Learning", "Fine-Tuning", "Pre-Training",
    ],

    "NLP & Language Models": [
        "Natural Language Processing", "Tokenization", "Embedding",
        "Word2Vec", "GloVe", "FastText", "BERT", "GPT", "T5",
        "Language Model", "Large Language Model", "Prompt Engineering",
        "Text Classification", "Named Entity Recognition", "Sentiment Analysis",
        "Machine Translation", "Text Generation", "Summarization",
        "Question Answering", "Information Retrieval", "Semantic Search",
        "Part-of-Speech Tagging", "Dependency Parsing", "Coreference Resolution",
    ],

    "Computer Vision": [
        "Computer Vision", "Image Classification", "Object Detection",
        "Image Segmentation", "Semantic Segmentation", "Instance Segmentation",
        "Face Recognition", "Pose Estimation", "Optical Character Recognition",
        "Image Generation", "Style Transfer", "Super Resolution",
        "Feature Extraction", "Edge Detection", "Convolution", "Pooling",
    ],

    "AI Applications": [
        "Recommendation Systems", "Speech Recognition", "Speech Synthesis",
        "Robotics", "Autonomous Vehicles", "Medical AI", "Drug Discovery",
        "Fraud Detection", "Anomaly Detection", "Predictive Maintenance",
        "AI Ethics", "AI Safety", "Explainable AI", "Fairness in AI",
        "AI Alignment", "AI Governance",
    ],

    # =========================================================================
    # PHYSICS
    # =========================================================================
    "Classical Mechanics": [
        "Mechanics", "Newton's Laws", "Force", "Mass", "Acceleration",
        "Momentum", "Energy", "Work", "Power", "Conservation Laws",
        "Kinematics", "Dynamics", "Statics", "Friction", "Gravity",
        "Projectile Motion", "Circular Motion", "Oscillation", "Waves",
        "Harmonic Motion", "Pendulum", "Rigid Body Dynamics",
    ],

    "Thermodynamics": [
        "Thermodynamics", "Temperature", "Heat", "Entropy", "Enthalpy",
        "Laws of Thermodynamics", "Heat Transfer", "Conduction", "Convection",
        "Radiation", "Phase Transitions", "Ideal Gas Law", "Carnot Cycle",
        "Statistical Mechanics", "Boltzmann Distribution",
    ],

    "Electromagnetism": [
        "Electromagnetism", "Electric Field", "Magnetic Field", "Electric Charge",
        "Current", "Voltage", "Resistance", "Capacitance", "Inductance",
        "Ohm's Law", "Kirchhoff's Laws", "Maxwell's Equations",
        "Electromagnetic Waves", "Light", "Optics", "Refraction", "Reflection",
        "Interference", "Diffraction", "Polarization",
    ],

    "Modern Physics": [
        "Quantum Mechanics", "Wave-Particle Duality", "Uncertainty Principle",
        "Schrödinger Equation", "Quantum Superposition", "Quantum Entanglement",
        "Quantum Computing", "Qubit", "Quantum Gates",
        "Relativity", "Special Relativity", "General Relativity",
        "Spacetime", "Time Dilation", "Length Contraction", "Mass-Energy Equivalence",
        "Particle Physics", "Standard Model", "Quarks", "Leptons", "Bosons",
        "Nuclear Physics", "Radioactivity", "Fission", "Fusion",
    ],

    "Astrophysics & Cosmology": [
        "Astronomy", "Astrophysics", "Cosmology", "Stars", "Planets",
        "Galaxies", "Black Holes", "Dark Matter", "Dark Energy",
        "Big Bang", "Cosmic Microwave Background", "Expansion of Universe",
        "Solar System", "Exoplanets", "Stellar Evolution", "Supernovae",
    ],

    # =========================================================================
    # CHEMISTRY
    # =========================================================================
    "General Chemistry": [
        "Atom", "Molecule", "Element", "Compound", "Mixture",
        "Periodic Table", "Atomic Number", "Atomic Mass", "Isotopes",
        "Electron Configuration", "Valence Electrons", "Chemical Bonding",
        "Ionic Bond", "Covalent Bond", "Metallic Bond", "Hydrogen Bond",
        "Van der Waals Forces", "Electronegativity",
    ],

    "Chemical Reactions": [
        "Chemical Reaction", "Reactants", "Products", "Stoichiometry",
        "Balancing Equations", "Mole", "Avogadro's Number",
        "Reaction Rate", "Reaction Kinetics", "Activation Energy", "Catalyst",
        "Chemical Equilibrium", "Le Chatelier's Principle",
        "Acids and Bases", "pH", "Buffer Solutions", "Titration",
        "Oxidation", "Reduction", "Redox Reactions", "Electrochemistry",
    ],

    "Organic Chemistry": [
        "Organic Chemistry", "Hydrocarbons", "Alkanes", "Alkenes", "Alkynes",
        "Functional Groups", "Alcohols", "Aldehydes", "Ketones", "Carboxylic Acids",
        "Esters", "Amines", "Aromatics", "Benzene", "Polymers",
        "Isomers", "Stereochemistry", "Chirality",
    ],

    "Biochemistry": [
        "Biochemistry", "Proteins", "Amino Acids", "Enzymes", "Carbohydrates",
        "Lipids", "Nucleic Acids", "ATP", "Metabolism", "Glycolysis",
        "Krebs Cycle", "Electron Transport Chain", "Photosynthesis",
    ],

    # =========================================================================
    # BIOLOGY
    # =========================================================================
    "Cell Biology": [
        "Cell", "Cell Theory", "Prokaryote", "Eukaryote", "Cell Membrane",
        "Nucleus", "Mitochondria", "Ribosome", "Endoplasmic Reticulum",
        "Golgi Apparatus", "Cytoplasm", "Cell Division", "Mitosis", "Meiosis",
    ],

    "Genetics & Molecular Biology": [
        "DNA", "RNA", "Gene", "Chromosome", "Genome", "Genetics",
        "Heredity", "Mendel's Laws", "Dominant", "Recessive", "Allele",
        "Genotype", "Phenotype", "Mutation", "Gene Expression",
        "Transcription", "Translation", "Genetic Code", "Codon",
        "CRISPR", "Gene Editing", "Epigenetics", "Genomics", "Proteomics",
    ],

    "Evolution & Ecology": [
        "Evolution", "Natural Selection", "Adaptation", "Speciation",
        "Phylogenetics", "Common Ancestor", "Fossil Record",
        "Ecology", "Ecosystem", "Food Chain", "Food Web", "Trophic Levels",
        "Biodiversity", "Conservation", "Endangered Species",
        "Population Dynamics", "Carrying Capacity", "Symbiosis",
    ],

    "Human Biology": [
        "Anatomy", "Physiology", "Organ Systems", "Nervous System",
        "Circulatory System", "Respiratory System", "Digestive System",
        "Immune System", "Endocrine System", "Musculoskeletal System",
        "Brain", "Heart", "Lungs", "Liver", "Kidneys",
        "Hormones", "Neurotransmitters", "Homeostasis",
    ],

    "Microbiology": [
        "Microbiology", "Bacteria", "Virus", "Fungi", "Parasites",
        "Microbiome", "Pathogen", "Infection", "Antibiotics", "Vaccines",
        "Immunity", "Immune Response", "Antibodies", "Antigens",
    ],

    # =========================================================================
    # MEDICINE & HEALTH
    # =========================================================================
    "Medicine": [
        "Medicine", "Diagnosis", "Treatment", "Prognosis", "Symptoms",
        "Disease", "Chronic Disease", "Acute Disease", "Syndrome",
        "Pharmacology", "Drug", "Dosage", "Side Effects", "Drug Interactions",
        "Surgery", "Anesthesia", "Medical Imaging", "X-Ray", "MRI", "CT Scan",
        "Clinical Trials", "Evidence-Based Medicine", "Medical Ethics",
    ],

    "Public Health": [
        "Public Health", "Epidemiology", "Pandemic", "Epidemic", "Endemic",
        "Vaccination", "Herd Immunity", "Quarantine", "Contact Tracing",
        "Health Policy", "Healthcare Systems", "Primary Care",
        "Preventive Medicine", "Screening", "Health Education",
    ],

    "Mental Health": [
        "Mental Health", "Psychology", "Psychiatry", "Anxiety", "Depression",
        "PTSD", "Bipolar Disorder", "Schizophrenia", "ADHD", "Autism",
        "Therapy", "Cognitive Behavioral Therapy", "Psychotherapy",
        "Counseling", "Mindfulness", "Stress Management",
    ],

    "Nutrition & Wellness": [
        "Nutrition", "Diet", "Calories", "Macronutrients", "Micronutrients",
        "Vitamins", "Minerals", "Metabolism", "BMI", "Obesity",
        "Exercise", "Fitness", "Sleep", "Hydration", "Wellness",
    ],

    # =========================================================================
    # EARTH & ENVIRONMENTAL SCIENCE
    # =========================================================================
    "Earth Science": [
        "Geology", "Plate Tectonics", "Earthquakes", "Volcanoes", "Rocks",
        "Minerals", "Fossils", "Erosion", "Weathering", "Soil",
        "Oceanography", "Ocean Currents", "Tides", "Marine Biology",
        "Meteorology", "Weather", "Climate", "Atmosphere", "Precipitation",
    ],

    "Environmental Science": [
        "Environmental Science", "Climate Change", "Global Warming",
        "Greenhouse Effect", "Carbon Cycle", "Carbon Footprint",
        "Renewable Energy", "Solar Energy", "Wind Energy", "Hydropower",
        "Pollution", "Air Quality", "Water Quality", "Waste Management",
        "Recycling", "Sustainability", "Conservation", "Deforestation",
        "Biodiversity Loss", "Extinction", "Ecological Footprint",
    ],

    # =========================================================================
    # SOCIAL SCIENCES
    # =========================================================================
    "Psychology": [
        "Psychology", "Cognition", "Perception", "Memory", "Learning",
        "Attention", "Intelligence", "Emotion", "Motivation", "Personality",
        "Developmental Psychology", "Social Psychology", "Cognitive Psychology",
        "Behavioral Psychology", "Positive Psychology", "Neuropsychology",
        "Conditioning", "Reinforcement", "Cognitive Dissonance",
    ],

    "Sociology": [
        "Sociology", "Society", "Culture", "Social Structure", "Social Norms",
        "Socialization", "Social Institutions", "Family", "Education",
        "Social Stratification", "Social Class", "Social Mobility",
        "Deviance", "Social Control", "Collective Behavior",
        "Urbanization", "Globalization", "Social Change",
    ],

    "Anthropology": [
        "Anthropology", "Cultural Anthropology", "Physical Anthropology",
        "Archaeology", "Linguistics", "Ethnography", "Human Evolution",
        "Cultural Relativism", "Ethnocentrism", "Kinship", "Ritual",
    ],

    "Political Science": [
        "Political Science", "Government", "Democracy", "Authoritarianism",
        "Constitution", "Legislature", "Executive", "Judiciary",
        "Elections", "Voting", "Political Parties", "Public Policy",
        "International Relations", "Diplomacy", "Sovereignty",
        "Political Ideology", "Liberalism", "Conservatism", "Socialism",
    ],

    "Economics": [
        "Economics", "Microeconomics", "Macroeconomics", "Supply and Demand",
        "Market", "Price", "Competition", "Monopoly", "Oligopoly",
        "GDP", "Inflation", "Unemployment", "Economic Growth", "Recession",
        "Fiscal Policy", "Monetary Policy", "Central Banking", "Interest Rates",
        "Trade", "Globalization", "Comparative Advantage", "Exchange Rates",
        "Behavioral Economics", "Game Theory", "Public Goods",
    ],

    # =========================================================================
    # HUMANITIES
    # =========================================================================
    "Philosophy": [
        "Philosophy", "Ethics", "Metaethics", "Normative Ethics", "Applied Ethics",
        "Utilitarianism", "Deontology", "Virtue Ethics", "Consequentialism",
        "Metaphysics", "Ontology", "Free Will", "Determinism",
        "Philosophy of Mind", "Consciousness", "Qualia", "Dualism", "Physicalism",
        "Philosophy of Science", "Philosophy of Language", "Aesthetics",
        "Political Philosophy", "Social Contract", "Justice", "Rights",
        "Existentialism", "Phenomenology", "Pragmatism", "Analytic Philosophy",
    ],

    "History": [
        "History", "Historiography", "Primary Sources", "Secondary Sources",
        "Ancient History", "Classical Antiquity", "Medieval History",
        "Renaissance", "Enlightenment", "Industrial Revolution",
        "Modern History", "Contemporary History", "World War I", "World War II",
        "Cold War", "Colonialism", "Imperialism", "Decolonization",
        "American History", "European History", "Asian History", "African History",
        "Economic History", "Social History", "Cultural History", "Military History",
    ],

    "Geography": [
        "Geography", "Physical Geography", "Human Geography", "Cartography",
        "Continents", "Countries", "Capitals", "Borders", "Regions",
        "Climate Zones", "Biomes", "Topography", "Landforms",
        "Demographics", "Population", "Migration", "Urbanization",
        "Geopolitics", "Natural Resources", "Land Use",
    ],

    "Linguistics": [
        "Linguistics", "Phonetics", "Phonology", "Morphology", "Syntax",
        "Semantics", "Pragmatics", "Sociolinguistics", "Psycholinguistics",
        "Historical Linguistics", "Language Families", "Language Acquisition",
        "Bilingualism", "Translation", "Interpretation",
        "Grammar", "Parts of Speech", "Sentence Structure",
    ],

    "Literature": [
        "Literature", "Fiction", "Non-Fiction", "Poetry", "Drama",
        "Novel", "Short Story", "Essay", "Memoir", "Biography",
        "Literary Analysis", "Literary Criticism", "Literary Devices",
        "Metaphor", "Simile", "Symbolism", "Irony", "Foreshadowing",
        "Narrative", "Plot", "Character", "Setting", "Theme",
        "Genre", "Tragedy", "Comedy", "Satire", "Romanticism", "Realism",
    ],

    "Arts": [
        "Visual Arts", "Painting", "Sculpture", "Drawing", "Photography",
        "Architecture", "Design", "Graphic Design", "Industrial Design",
        "Art History", "Renaissance Art", "Modern Art", "Contemporary Art",
        "Music", "Music Theory", "Melody", "Harmony", "Rhythm",
        "Composition", "Orchestration", "Musical Genres",
        "Theater", "Acting", "Directing", "Stagecraft",
        "Film", "Cinematography", "Film Editing", "Screenwriting",
        "Dance", "Choreography", "Ballet", "Contemporary Dance",
    ],

    "Religion & Spirituality": [
        "Religion", "Theology", "Spirituality", "Faith", "Belief Systems",
        "Christianity", "Islam", "Judaism", "Hinduism", "Buddhism",
        "Atheism", "Agnosticism", "Secularism", "Religious Pluralism",
        "Sacred Texts", "Ritual", "Prayer", "Meditation", "Worship",
        "Ethics and Religion", "Theodicy", "Afterlife",
    ],

    # =========================================================================
    # LAW & GOVERNANCE
    # =========================================================================
    "Law": [
        "Law", "Legal System", "Common Law", "Civil Law", "Constitutional Law",
        "Criminal Law", "Civil Law", "Contract Law", "Tort Law", "Property Law",
        "International Law", "Human Rights Law", "Environmental Law",
        "Legislation", "Statute", "Regulation", "Precedent", "Jurisprudence",
        "Courts", "Trial", "Evidence", "Due Process", "Rule of Law",
        "Legal Rights", "Legal Obligations", "Liability",
    ],

    "Governance": [
        "Governance", "Public Administration", "Bureaucracy", "Policy Making",
        "Regulation", "Compliance", "Accountability", "Transparency",
        "Civil Society", "Non-Governmental Organizations", "Lobbying",
        "International Organizations", "United Nations", "Treaties",
    ],

    # =========================================================================
    # BUSINESS & MANAGEMENT
    # =========================================================================
    "Business": [
        "Business", "Entrepreneurship", "Startup", "Business Model",
        "Revenue", "Profit", "Loss", "Cash Flow", "Break-Even",
        "Marketing", "Sales", "Branding", "Advertising", "Market Research",
        "Customer", "Value Proposition", "Competitive Advantage",
        "Supply Chain", "Logistics", "Operations", "Quality Control",
    ],

    "Management": [
        "Management", "Leadership", "Strategy", "Planning", "Organizing",
        "Decision Making", "Problem Solving", "Delegation", "Motivation",
        "Team Building", "Conflict Resolution", "Change Management",
        "Project Management", "Risk Management", "Performance Management",
        "Human Resources", "Recruitment", "Training", "Compensation",
    ],

    "Finance & Accounting": [
        "Finance", "Accounting", "Financial Statements", "Balance Sheet",
        "Income Statement", "Cash Flow Statement", "Assets", "Liabilities",
        "Equity", "Revenue", "Expenses", "Profit", "Loss",
        "Investment", "Portfolio", "Diversification", "Risk",
        "Stocks", "Bonds", "Mutual Funds", "ETFs", "Derivatives",
        "Valuation", "Discounted Cash Flow", "Time Value of Money",
        "Budgeting", "Forecasting", "Financial Planning",
    ],

    # =========================================================================
    # ENGINEERING
    # =========================================================================
    "Engineering Fundamentals": [
        "Engineering", "Engineering Design", "Systems Engineering",
        "Requirements", "Specifications", "Prototyping", "Testing",
        "Reliability", "Maintainability", "Safety Engineering",
        "Engineering Ethics", "Professional Responsibility",
    ],

    "Mechanical Engineering": [
        "Mechanical Engineering", "Mechanics of Materials", "Stress", "Strain",
        "Fluid Mechanics", "Thermodynamics", "Heat Transfer",
        "Machine Design", "Kinematics", "Dynamics", "Vibrations",
        "Manufacturing", "CNC", "3D Printing", "CAD",
    ],

    "Electrical Engineering": [
        "Electrical Engineering", "Circuits", "Electronics", "Semiconductors",
        "Transistors", "Integrated Circuits", "Microprocessors",
        "Power Systems", "Electric Motors", "Generators",
        "Control Systems", "Feedback", "PID Controller",
        "Signal Processing", "Analog", "Digital",
    ],

    "Civil Engineering": [
        "Civil Engineering", "Structural Engineering", "Geotechnical Engineering",
        "Transportation Engineering", "Water Resources", "Environmental Engineering",
        "Construction", "Materials", "Concrete", "Steel", "Bridge", "Dam",
    ],

    "Chemical Engineering": [
        "Chemical Engineering", "Process Engineering", "Reaction Engineering",
        "Separation Processes", "Mass Transfer", "Heat Transfer",
        "Process Control", "Plant Design", "Safety",
    ],

    # =========================================================================
    # COMMUNICATION & MEDIA
    # =========================================================================
    "Communication": [
        "Communication", "Verbal Communication", "Non-Verbal Communication",
        "Written Communication", "Visual Communication", "Interpersonal Communication",
        "Public Speaking", "Presentation Skills", "Persuasion", "Negotiation",
        "Active Listening", "Feedback", "Clarity", "Conciseness",
    ],

    "Media & Journalism": [
        "Media", "Journalism", "News", "Reporting", "Investigative Journalism",
        "Editorial", "Opinion", "Fact-Checking", "Media Literacy",
        "Social Media", "Digital Media", "Broadcasting", "Print Media",
        "Media Ethics", "Freedom of Press", "Censorship", "Propaganda",
    ],

    # =========================================================================
    # EDUCATION & LEARNING
    # =========================================================================
    "Education": [
        "Education", "Pedagogy", "Curriculum", "Assessment", "Evaluation",
        "Learning Objectives", "Instructional Design", "Lesson Planning",
        "Classroom Management", "Differentiated Instruction", "Inclusive Education",
        "Educational Psychology", "Learning Theories", "Constructivism",
        "Bloom's Taxonomy", "Formative Assessment", "Summative Assessment",
        "Distance Learning", "E-Learning", "Blended Learning", "MOOC",
    ],

    "Learning": [
        "Learning", "Memory", "Attention", "Concentration", "Study Skills",
        "Note-Taking", "Spaced Repetition", "Active Recall", "Elaboration",
        "Metacognition", "Self-Regulated Learning", "Growth Mindset",
        "Critical Reading", "Research Skills", "Information Literacy",
    ],

    # =========================================================================
    # PRACTICAL SKILLS
    # =========================================================================
    "Life Skills": [
        "Time Management", "Goal Setting", "Planning", "Prioritization",
        "Decision Making", "Problem Solving", "Critical Thinking",
        "Emotional Intelligence", "Self-Awareness", "Self-Regulation",
        "Stress Management", "Resilience", "Adaptability", "Flexibility",
        "Financial Literacy", "Budgeting", "Saving", "Investing",
    ],

    "Professional Skills": [
        "Professionalism", "Work Ethic", "Accountability", "Integrity",
        "Teamwork", "Collaboration", "Leadership", "Initiative",
        "Networking", "Career Development", "Resume Writing", "Interviewing",
        "Workplace Communication", "Email Etiquette", "Meeting Management",
    ],

    "Technical Skills": [
        "Technical Writing", "Documentation", "Troubleshooting", "Debugging",
        "Data Analysis", "Spreadsheets", "Data Visualization",
        "Research Methods", "Experimental Design", "Statistical Analysis",
    ],
}


def get_all_foundational_concepts() -> Set[str]:
    """Get flat set of all concepts from taxonomy."""
    concepts = set()
    for category, items in FOUNDATIONAL_TAXONOMY.items():
        concepts.add(category)
        concepts.update(items)
    return concepts


def get_concept_count() -> int:
    """Count total concepts in taxonomy."""
    count = len(FOUNDATIONAL_TAXONOMY)  # Categories
    for items in FOUNDATIONAL_TAXONOMY.values():
        count += len(items)
    return count


def extract_concepts_lda(data: List, n_topics: int = 100, n_words: int = 15) -> Set[str]:
    """
    Use LDA topic modeling to discover concepts from training data.
    Returns top words from each topic as potential concepts.
    """
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
    except ImportError:
        print("  Warning: sklearn not available for LDA")
        return set()

    print(f"  Running LDA ({n_topics} topics)...")

    # Combine prompts and responses
    documents = [f"{p} {r}" for p, r in data[:15000]]  # Sample 15k

    # Stopwords
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
        'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
        'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until', 'while',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'it',
        'its', 'you', 'your', 'we', 'our', 'they', 'their', 'he', 'she', 'him',
        'her', 'i', 'me', 'my', 'myself', 'code', 'example', 'use', 'using',
        'following', 'like', 'also', 'one', 'two', 'three', 'write', 'python',
        'function', 'return', 'print', 'def', 'class', 'import', 'from',
    }

    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=5,
        max_features=5000,
        stop_words=list(stopwords),
        ngram_range=(1, 2),
        token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
    )

    try:
        doc_term_matrix = vectorizer.fit_transform(documents)
    except ValueError as e:
        print(f"  Warning: Vectorization failed: {e}")
        return set()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method='online',
        random_state=42,
        n_jobs=-1,
    )

    lda.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    concepts = set()

    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-n_words-1:-1]
        top_words = [feature_names[i] for i in top_indices]

        for word in top_words:
            if len(word) > 3 and word.lower() not in stopwords:
                concept = ' '.join(w.title() for w in word.split())
                concepts.add(concept)

    print(f"  LDA extracted {len(concepts)} raw concepts")
    return concepts


def filter_lda_concepts(lda_concepts: Set[str], taxonomy: dict) -> Set[str]:
    """Filter LDA concepts to meaningful foundational ones."""
    # Get taxonomy terms for matching
    taxonomy_terms = set()
    for category, items in taxonomy.items():
        taxonomy_terms.add(category.lower())
        taxonomy_terms.update(item.lower() for item in items)

    filtered = set()

    # Skip patterns (garbage)
    skip_patterns = ['python', 'java', 'code', 'write', 'print', 'hello',
                     'world', 'string', 'list', 'number', 'value', 'example']

    for concept in lda_concepts:
        concept_lower = concept.lower()

        # Skip garbage
        if any(p in concept_lower for p in skip_patterns):
            continue

        # Keep if matches taxonomy or has technical indicators
        if concept_lower in taxonomy_terms:
            filtered.add(concept)
        elif any(term in concept_lower for term in taxonomy_terms if len(term) > 5):
            filtered.add(concept)
        elif any(suffix in concept_lower for suffix in ['tion', 'ment', 'ity', 'ism', 'ology']):
            filtered.add(concept)

    return filtered


def load_existing_concepts(path: str) -> Dict[str, str]:
    """Load existing concept dict, preserving all values."""
    if not os.path.exists(path):
        return {}

    try:
        with open(path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            return {concept: "" for concept in data}
        elif isinstance(data, dict):
            return data
        return {}
    except Exception as e:
        print(f"  Warning: Could not load existing concepts: {e}")
        return {}


def load_lexicon_sample(path: str) -> Set[str]:
    """Load lexicon sample list/dict if present."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return set(data.keys())
        if isinstance(data, list):
            return set(str(x) for x in data)
        return set()
    except Exception as e:
        print(f"  Warning: Could not load lexicon sample: {e}")
        return set()


def main():
    print(f"\n{'='*60}")
    print(f"  COMPREHENSIVE CONCEPT HARVESTER (TAXONOMY + LDA)")
    print(f"{'='*60}")

    # 1. Load existing concepts (preserve values!)
    print(f"\n  Loading existing concepts from '{OUTPUT_FILE}'...")
    existing = load_existing_concepts(OUTPUT_FILE)
    print(f"  Found {len(existing)} existing concepts")
    with_values = sum(1 for v in existing.values() if v)
    print(f"  With generated values: {with_values}")

    # 2. Get comprehensive taxonomy concepts
    print(f"\n  Loading comprehensive taxonomy...")
    taxonomy_concepts = get_all_foundational_concepts()
    print(f"  Taxonomy concepts: {len(taxonomy_concepts)}")

    # 3. Load training data for LDA (balanced avoids massive multipliers)
    print(f"\n  Loading training data for LDA...")
    try:
        all_data = get_all_training_data(balanced=True)
        print(f"  Training examples: {len(all_data)}")
    except Exception as e:
        print(f"  Warning: Could not load training data ({e}), skipping LDA.")
        all_data = []

    # 4. Run LDA to discover additional concepts
    if all_data:
        print(f"\n  Running LDA topic modeling...")
        lda_raw = extract_concepts_lda(all_data, n_topics=150, n_words=20)
        lda_filtered = filter_lda_concepts(lda_raw, FOUNDATIONAL_TAXONOMY)
        print(f"  LDA filtered: {len(lda_filtered)} concepts")
    else:
        print(f"\n  Skipping LDA (no training data available).")
        lda_filtered = set()

    # 5. Optional lexicon sample
    lexicon_sample = load_lexicon_sample(LEXICON_SAMPLE_FILE)
    if lexicon_sample:
        print(f"  Lexicon sample: {len(lexicon_sample)} concepts")

    # 6. Combine all concepts
    all_concepts = taxonomy_concepts | lda_filtered | lexicon_sample

    # 6. Merge with existing (preserve values!)
    merged = existing.copy()
    added = 0
    for concept in all_concepts:
        if concept not in merged:
            merged[concept] = ""
            added += 1

    print(f"\n  Merge results:")
    print(f"    Preserved existing:  {len(existing)}")
    print(f"    From taxonomy:       {len(taxonomy_concepts)}")
    print(f"    From LDA:            {len(lda_filtered)}")
    print(f"    New concepts added:  {added}")
    print(f"    Total concepts:      {len(merged)}")

    # 7. Save
    sorted_concepts = dict(sorted(merged.items()))
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sorted_concepts, f, indent=2)

    print(f"\n  Saved {len(sorted_concepts)} concepts to '{OUTPUT_FILE}'")

    # Show sample LDA discoveries
    lda_new = lda_filtered - taxonomy_concepts
    if lda_new:
        print(f"\n  Sample LDA-discovered concepts:")
        for c in sorted(lda_new)[:15]:
            print(f"    + {c}")

    # Show categories
    print(f"\n  Taxonomy categories ({len(FOUNDATIONAL_TAXONOMY)}):")
    for cat in sorted(FOUNDATIONAL_TAXONOMY.keys()):
        count = len(FOUNDATIONAL_TAXONOMY[cat])
        print(f"    - {cat}: {count}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
