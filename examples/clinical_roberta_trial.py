from knowledge_neurons import KnowledgeNeurons, initialize_model_and_tokenizer, model_type
import random

# first initialize some hyperparameters
MODEL_NAME = ("../../pretrained-models/RoBERTa-base-Mimic-half-1_epoch/roberta-base-custom/12-09-2023--15-28/checkpoint-10000")

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


# suppress the activations at the refined neurons + test the effect on a relevant prompt
# 'results_dict' is a dictionary containing the probability of the ground truth being generated before + after modification, as well as other info
# 'unpatch_fn' is a function you can use to undo the activation suppression in the model.
# By default, the suppression is removed at the end of any function that applies a patch, but you can set 'undo_modification=False',
# run your own experiments with the activations / weights still modified, then run 'unpatch_fn' to undo the modifications
results_dict, unpatch_fn = kn.suppress_knowledge(
    TEXT, GROUND_TRUTH, refined_neurons
)

# suppress the activations at the refined neurons + test the effect on an unrelated prompt
results_dict, unpatch_fn = kn.suppress_knowledge(
    "<mask> is the official language of the solomon islands",
    "english",
    refined_neurons,
)

# enhance the activations at the refined neurons + test the effect on a relevant prompt
results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, refined_neurons)

# erase the weights of the output ff layer at the refined neurons (replacing them with zeros) + test the effect
results_dict, unpatch_fn = kn.erase_knowledge(
    TEXT, refined_neurons, target=GROUND_TRUTH, erase_value="zero"
)

# erase the weights of the output ff layer at the refined neurons (replacing them with an unk token) + test the effect
results_dict, unpatch_fn = kn.erase_knowledge(
    TEXT, refined_neurons, target=GROUND_TRUTH, erase_value="unk"
)

# edit the weights of the output ff layer at the refined neurons (replacing them with the word embedding of 'target') + test the effect
# we can make the model think the capital of france is London!
results_dict, unpatch_fn = kn.edit_knowledge(
    TEXT, target="london", neurons=refined_neurons
)

a = 1
