from knowledge_neurons import KnowledgeNeurons, initialize_model_and_tokenizer, model_type
import random

# first initialize some hyperparameters
MODEL_NAME = "bert-base-uncased"

# to find the knowledge neurons, we need the same 'facts' expressed in multiple different ways, and a ground truth
TEXTS = [
    "Sarah was visiting <mask>, the capital of france",
    "The capital of france is <mask>",
    "<mask> is the capital of france",
    "France's capital <mask> is a hotspot for romantic vacations",
    "The eiffel tower is situated in <mask>",
    "<mask> is the most populous city in france",
    "<mask>, france's capital, is one of the most popular tourist destinations in the world",
]
TEXT = TEXTS[0]
GROUND_TRUTH = "paris"

# these are some hyperparameters for the integrated gradients step
BATCH_SIZE = 20
STEPS = 20 # number of steps in the integrated grad calculation
ADAPTIVE_THRESHOLD = 0.3 # in the paper, they find the threshold value `t` by multiplying the max attribution score by some float - this is that float.
P = 0.5 # the threshold for the sharing percentage

# setup model & tokenizer
model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)

# check out model params
model.state_dict().keys()

# initialize the knowledge neuron wrapper with your model, tokenizer and a string expressing the type of your model ('gpt2' / 'gpt_neo' / 'bert')
kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(MODEL_NAME))

# use the integrated gradients technique to find some refined neurons for your set of prompts
refined_neurons = kn.get_refined_neurons(
    TEXTS,
    GROUND_TRUTH,
    p=P,
    batch_size=BATCH_SIZE,
    steps=STEPS,
    coarse_adaptive_threshold=ADAPTIVE_THRESHOLD,
)

a=1
