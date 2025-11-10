# 1. Introduction

It's been just over two years since large language models (LLMs) reached the general public and, for some, it still feels like magic how precise (or at least seemingly precise) their answers can be. But far from being any kind of magic, LLMs are simply models capable of processing and generating text after going through exhaustive training stages as Transformer layers, Backpropagation, Fine-Tuning and Reinforcement Learning.

However, for all this training to happen, a crucial yet often overlooked step must take place: the **Tokenization**. This is the first step of the process of converting text inputs into numbers so that machines can handle them. To achieve this, tokenizer models are trained to break text into smaller units called tokens, following a couple of specific rules and building their own dictionaries of these **tokens**, also called vocabularies. Once trained, the tokenizer can then map any new text into tokens and their IDs (indexes) by looking up matching pieces in its vocabulary. An example of the use of a pretrained tokenizer (in this case, the GPT tokenizer) is presented in Figure 1.[1,2]

### ***Figure 1: Illustrating the Tokenization Process***
<p align="center">
<img src="figures/fig1a.png" alt="Figure 1a: Visualization of tokenized text showing how each word or phrase is split into colored tokens. The sentence “I thought about the past, thought about the present, and wondered about the future” is segmented by color to represent different tokens.o" width="450">
<img src="figures/fig1b.png" alt="Figure 1b: Table showing the top 5 most frequent tokens, including their token IDs and frequencies. The tokens “the” and “about” appear three times each, while “thought” and “,” appear twice, and “wondered” appears once." width="450">
</p>


>***Caption.** Example of tokenization of an English sentence, showing both the tokenized text and the most frequent tokens with their IDs. Notice that the total number of tokens in the sentence is not the same as the number of unique dictionary entries, since repeated tokens share the same ID. The image was generated using gptforwork.com/tools/tokenizer. Try running your own experiments there to better understand how tokenization works!*
>
>***Source.** GPT for Work\'s Online Tokenizer, available at
<https://gptforwork.com/tools/tokenizer>.*

To make this idea more intuitive, we can think of Tokenization as the **chewing process in the digestive system**: just as chewing breaks food into smaller pieces so it can be swallowed and properly digested by stomach and intestines, tokenization breaks text into manageable units so that the subsequent training steps of LLMs - such as Transformers - can learn from the information, or in other words, extract nutrients from it.

There are a couple of different tokenization types and methods that we will go over in this article and to further understand their strengths and weaknesses, we propose four key metrics to evaluate them:

-   **Coverage** → How well can the tokenizer represent any possible input? Can it handle rare words, numbers, emojis, and multilingual text without gaps?

-   **Efficiency** → How compactly does the tokenizer encode text? Does it minimize the number of tokens per sentence while balancing vocabulary size and sequence length?

-   **Consistency** → How uniform are the segmentation patterns? Does the same word always get segmented the same way? Are there redundant tokens?

-   **Meaning Conservation** → Do the tokens generated have meaning unit? Does the tokenizer capture semantic relationships between similar words?

In short, achieving a consistent and efficient language model requires a well-designed tokenizer. This, in turn, demands training on a sufficiently large and diverse dataset to ensure a comprehensive vocabulary, so that no relevant text fragment is left unrepresented. At the same time, the tokenizer must segment the input into a reasonable number of tokens, avoiding both excessive fragmentation and redundancy.

## Setup

This section installs and configures all the dependencies required to run the notebook. The libraries listed below are essential for Natural Language Processing (NLP) tasks, data visualization, and text tokenization.

``` python
# If this is your first time running the notebook,
# you can uncomment and run the commands below to install the required package

# !pip -q install transformers==4.* sentencepiece spacy tiktoken matplotlib pandas
# !pip install numpy pandas matplotlib
# !python -m spacy download en_core_web_sm
```

### Utilities

``` python
# Import required libraries
from typing import List
import string
import io, re, unicodedata
import numpy as np
import pandas as pd
import token
import matplotlib.pyplot as plt
from matplotlib import colors
import tokenize, textwrap as _tw
from typing import Callable, Dict, List, Iterable
from textwrap import dedent

# Function to avoid wrong lenghts in the function below
def safe_len(fn, text):
    try:
        toks = fn(text)
        return len(toks)
    except Exception:
        return None

#Function to compute the number of tokens produced by multiple tokenizers for each sentence.
def compute_token_counts(
    tokenizers: Dict[str, Callable[[str], List[str]]],
    sentences: Dict[str, str]
) -> Dict[str, List[int]]:
    counts = {name: [] for name in tokenizers}
    for _, sent in sentences.items():
        for name, fn in tokenizers.items():
            n = safe_len(fn, sent)
            counts[name].append(n if n is not None else np.nan)
    return counts

#Function to plot a grouped bar chart comparing token counts across tokenizers and sentences.
def plot_token_counts(
    counts: Dict[str, List[int]],
    labels: Iterable[str],
    title: str = "Comparing Tokenization Levels"
):
    labels = list(labels)
    x = np.arange(len(labels))
    series = list(counts.items())
    width = 0.8 / max(1, len(series))

    plt.figure(figsize=(4, 3), dpi=150)
    for i, (name, values) in enumerate(series):
        plt.bar(x + i * width, values, width, label=name)

    plt.title(title)
    plt.ylabel("Number of Tokens")
    plt.xlabel("Sentence Size")
    plt.xticks(x + (len(series) - 1) * width / 2, labels)
    plt.grid(True, linestyle="--", alpha=0.6, axis="y")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.show()

# Function to show the tokens of a text, for a specific tokenizer model
def subword_tokenizer(texts, tokenizer):
  for text in texts:
    # Return string tokens (subwords) before mapping to IDs
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Display outputs to inspect token strings, token IDs, and decoder round-trips.
    print(f"Text: {text}")
    print("Tokens and their vocabulary indices:")

    # Create a pandas DataFrame for table formatting
    df = pd.DataFrame({'Token': tokens, 'Token ID': token_ids})
    display(df)
    print("\n")
```

### Load common tokenizers (first run may download models)


``` python
from transformers import AutoTokenizer

tok_bpe = AutoTokenizer.from_pretrained("gpt2")                 # BPE
tok_wp  = AutoTokenizer.from_pretrained("bert-base-uncased")    # WordPiece
tok_uni = AutoTokenizer.from_pretrained("google/mt5-small")     # Unigram (SentencePiece)

import spacy
nlp = spacy.load("en_core_web_sm")
```


#    
# 2. Word-Level Tokenization: the most intuitive alternative 

It is understandable that the tokenization concept almost automatically evokes the idea of breaking the text into **words**, after all, that is the way we are already used to (at least speakers of English and Romance Languages).

The most intuitive and simple idea is to split sentences at every whitespace, a process known as **Whitespace-Based Tokenization (WBT)**. This is a rule-based tokenization technique, meaning it follows a fixed rule without any learnable parameters to optimize how text is segmented. As a result, the training phase of these tokenizers focuses only on building a vocabulary of known words and assigning them unique IDs. This is typically done by setting a maximum vocabulary size *k* and selecting the *k* most frequent words from the training corpus [3].

Although being very straightforward, it has several notable limitations [1], beginning with its **language dependent coverage**. As long as words are separated by whitespaces, as in English, Portuguese or French, they can be tokenized by this method. However whitespaces are not the paradigm in all writing systems. Some languages such as Japanese, Chinese, Hebrew, Thai or Burmese do not have explicit word boundary markers as whitespaces, making WBT incapable of tokenizing its texts [4]. In section 6, tokenization of other languages is discussed.

Another issue with this approach is related to **consistency**. Since token boundaries are defined solely by whitespaces, words followed by punctuation are treated as completely new tokens (as shown in Example 1 of our Colab Notebook, resulting in multiple entries of the same word, one for each punctuation variation, being added to the vocabulary [2].

Furthermore, this method fails to **conserve meaning relationships** between compound words and their components. For example, it fails to connect "sunflower" with "sun", or "grandfather" with "father". These relationships must be learned in later training stages, which is more expensive. In addition, WBT cannot reuse tokens from the component words and instead creates entirely new tokens for each compound word.

These factors inflate the vocabulary size required to represent what is essentially the same set of words. And because the vocabulary size is predetermined, it actually means that more words are likely to fall outside of it - these are known as **Out-Of-Vocabulary words (OOVs)** - as a result of the presence of these duplicates. It either reduces the **efficiency** of the tokenizer, due to the larger dictionary, or limits its coverage by increasing the number of OOVs [2].

An alternative to the WBT is the **Punctuation-Based Tokenization (PBT)**. Beyond whitespaces, this model also uses the punctuation of sentences to separate words into tokens. In this way, each punctuation mark will be stored in a separate token, preventing words followed by punctuation from being treated as new tokens, as occurred in whitespace-based tokenization [1].

Although it might solve the problem of punctuation-copies, this tokenization model still retains most WBT issues and also presents a new flaw [2]: there are situations where punctuation is inside words - as in the case of e-mails, phone numbers, URLs or decimal points - and we don\'t want these units to be split apart as the PBT will do, because it loses its semantic value, as we can see in Example 1.

In short, neither of these two raw tokenization methods performs well enough to serve as a final tokenizer in large-scale language models today [5]. Nevertheless, several modern tokenization pipelines still rely on an initial segmentation of text into words before further subword processing.

In such cases, it is common to employ tokenizers such as **SpaCy** [6], that uses carefully crafted rules and lexical exceptions rather than simple whitespace or punctuation splitting, providing a more robust pre-processing stage for downstream subword tokenization methods.

It is interesting to add that while it is common to normalize text by converting all words to lowercase, simplifying vocabulary by treating "Cat" and "cat" as the same token, SpaCy takes a different approach: it preserves the original casing of tokens, despite its increased memory usage and duplicate entries. This choice enhances linguistic accuracy in downstream tasks such as Named Entity Recognition (NER), where capitalization can carry important semantic meaning (e.g., Apple vs. apple)[6].

By doing so, SpaCy retains contextual information that would otherwise be lost through normalization, while it still allows normalization for later processing stages that may require case-insensitive input [6].

### Example 1 - Comparison between Whitespace-Based, Punctuation-Based Tokenization and SpaCy Outputs 

> ***Caption.** This code compares three tokenization strategies: Whitespace-Based Tokenization (WBT), Punctuation-Based Tokenization (PBT), and SpaCy's rule-based tokenizer.*
>
> -   **WBT** simply splits text at whitespaces, leading to redundant tokens such as "programming." and "programming".
> -   **PBT** separates punctuation into independent tokens, solving the duplication issue but incorrectly fragmenting elements like email addresses or decimal numbers.
> -   **SpaCy** applies a linguistic, rule-based approach that accounts for punctuation, contractions, and special cases, preserving meaningful units like \"did\" + \"n\'t\", \"<tokenization.guy@mail.com>\" and \"\$1.234,56\".
>
> *The examples highlight the trade-offs between simplicity and linguistic precision across these tokenization methods.*

``` python
# Function for WBT tokenization
def manual_WBT(text):
    return text.split()

# Function for PBT tokenization
def manual_PBT(text):
    tokens = []
    current_token = ""
    for char in text:
        if char.isspace() or char in string.punctuation:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            if char in string.punctuation:
                tokens.append(char)
        else:
            current_token += char
    if current_token:
        tokens.append(current_token)
    return tokens
```


``` python
CASES = [
    "I love programming. Programming challenges are fun!",
    "She said she'll call when she gets home.",
    "tokenization.guy@mail.com",
    "Didn't it cost $1.234,56, did it?"
]

# Display outputs to inspect token strings
for s in CASES:
    print("Text Input:", s)
    print("WBT Output:  ", manual_WBT(s))
    print("PBT Output:  ", manual_PBT(s))
    print("SpaCy Output:",[token.text for token in nlp(s)])

    print("-" * 100)
```

    Text Input: I love programming. Programming challenges are fun!
    WBT Output:   ['I', 'love', 'programming.', 'Programming', 'challenges', 'are', 'fun!']
    PBT Output:   ['I', 'love', 'programming', '.', 'Programming', 'challenges', 'are', 'fun', '!']
    SpaCy Output: ['I', 'love', 'programming', '.', 'Programming', 'challenges', 'are', 'fun', '!']
    ----------------------------------------------------------------------------------------------------
    Text Input: She said she'll call when she gets home.
    WBT Output:   ['She', 'said', "she'll", 'call', 'when', 'she', 'gets', 'home.']
    PBT Output:   ['She', 'said', 'she', "'", 'll', 'call', 'when', 'she', 'gets', 'home', '.']
    SpaCy Output: ['She', 'said', 'she', "'ll", 'call', 'when', 'she', 'gets', 'home', '.']
    ----------------------------------------------------------------------------------------------------
    Text Input: tokenization.guy@mail.com
    WBT Output:   ['tokenization.guy@mail.com']
    PBT Output:   ['tokenization', '.', 'guy', '@', 'mail', '.', 'com']
    SpaCy Output: ['tokenization.guy@mail.com']
    ----------------------------------------------------------------------------------------------------
    Text Input: Didn't it cost $1.234,56, did it?
    WBT Output:   ["Didn't", 'it', 'cost', '$1.234,56,', 'did', 'it?']
    PBT Output:   ['Didn', "'", 't', 'it', 'cost', '$', '1', '.', '234', ',', '56', ',', 'did', 'it', '?']
    SpaCy Output: ['Did', "n't", 'it', 'cost', '$', '1.234,56', ',', 'did', 'it', '?']
    ----------------------------------------------------------------------------------------------------

  \\
# 
# 3. Character-Level Tokenization: Breaking it all the way down
The underlying logic of this type of tokenizer is straightforward: while a language may contain thousands of words, it is composed of a limited set of characters.

For instance, although there are approximately 170,000 distinct words in English, according to the Oxford English Dictionary [7], any text in this language can be represented with just a little over one hundred characters (not only the 26 letters, but also digits, whitespace, punctuation, and common symbols) [8].

To process texts in languages that use accent marks, different alphabets, emojis, or mathematical symbols, more than a hundred characters are needed. That is why Character-Level Tokenization relies on Unicode. Unicode is a standardized system that assigns a unique code point to every character across the world's writing systems, as well as symbols, punctuation marks, and emojis [9].

This tokenization maps each code point to a single token, meaning that the total number of tokens in the sequence corresponds exactly to the number of characters in the input, as illustrated in Example 2.

### Example 2 - Character-Level: One Token per Unicode Code Point

> ***Caption.** Each character (Unicode code point) becomes a token, guaranteeing coverage but producing longer sequences.*


``` python
samples = ["house", "language", "internationalization"]

for sample in samples:
    chars = list(sample)
    print(f"Word: {sample}")
    print("Characters:", chars, "| length:", len(chars))
    print("-" * 130)
```

    Word: house
    Characters: ['h', 'o', 'u', 's', 'e'] | length: 5
    ----------------------------------------------------------------------------------------------------------------------------------
    Word: language
    Characters: ['l', 'a', 'n', 'g', 'u', 'a', 'g', 'e'] | length: 8
    ----------------------------------------------------------------------------------------------------------------------------------
    Word: internationalization
    Characters: ['i', 'n', 't', 'e', 'r', 'n', 'a', 't', 'i', 'o', 'n', 'a', 'l', 'i', 'z', 'a', 't', 'i', 'o', 'n'] | length: 20
    ----------------------------------------------------------------------------------------------------------------------------------

The trade-off of this approach is that, while it keeps the dictionary relatively small compared to other tokenization methods -- which can exceed 800,000 tokens -- it generates much longer token sequences, as illustrated in Example 3. This leads to an efficiency issue, since it forces the model to use significantly more layers and parameters to capture dependencies across characters. As a result, the LLMs training becomes slower, more memory-intensive, and computationally expensive [10].

### Example 3 - Sequence Length Growth at Character Level 
> ***Caption.** This example compares character counts for short, medium, and long samples. In short sentences, using character-level tokenization might not seem so disadvantageous, but as sentence length increases, its token sequences become extremely bigger than the sequences of word-level and subword-level tokenization- which is going to be discussed next.*

``` python
# Models used as examples
tokenizers = {
    "Word-Level (SpaCy)": lambda s: [token.text for token in nlp(s)],
    "Subword-Level (WordPiece - BERT)": lambda s: tok_wp.tokenize(s),
    "Character-Level": lambda s: list(s)
}

# Sentences used as examples
sentences = {
  "Small": "Artificial intelligence is transforming the way we use language.",
  "Medium": "Natural Language Processing, also known as NLP, is an interdisciplinary field that combines techniques from linguistics, statistics, and machine learning to enable computers to understand, interpret, and generate text in a way that increasingly resembles human language.",
  "Large": "Over the past decades, advances in language models have revolutionized the field of artificial intelligence. Models based on deep architectures, such as BERT, RoBERTa, and T5, have enabled remarkable achievements in tasks such as machine translation, sentiment analysis, question answering, text summarization, and even creative language generation. These models have become fundamental tools for research, business, and education, paving the way for innovative applications in virtually every area that depends on human communication. However, ethical and social challenges also arise, such as bias in models, the privacy of data used in training, and accountability for the use of these technologies, which must be carefully considered in the future of NLP."
}
```
``` python
results = {tok_name: [] for tok_name in tokenizers.keys()}
for size, sentence in sentences.items():
    for name, fn in tokenizers.items():
        n = fn(sentence)
        if n is not None:
            results[name].append(n)

counts = compute_token_counts(tokenizers, sentences)
plot_token_counts(counts, sentences.keys(), title="Comparing Tokenization Levels")
```

<p align="center">
<img src="figures/graph1.png" alt="Bar chart comparing tokenization levels across sentence sizes. The x-axis shows sentence size categories (Small, Medium, Large), and the y-axis represents the number of tokens. Three tokenization methods are compared: Word-Level (SpaCy) in blue, Subword-Level (WordPiece – BERT) in orange, and Character-Level in green. For all sentence sizes, Character-Level produces significantly more tokens, especially in large sentences, while Word- and Subword-Level tokenizations yield fewer and similar numbers of tokens." width="450">
</p>

One of the main advantages of this method is its ***coverage***. Since the vocabulary is defined at the character level, there are no unknown words (Out of Vocabulary - OOVs). Any new word can always be represented as a combination of characters already present in the model's vocabulary.

An additional important aspect is that ***meaning conservation*** of character-level tokenization varies across languages. In languages based on the Latin alphabet, such as Portuguese or English, splitting the text into individual characters can lead to a loss of meaning, since a single character carries little to no semantic information on its own. In contrast, in languages such as Chinese or Japanese, each character inherently carries semantic meaning, as we can see in Example 4. In these cases, character-level tokenization tends to be more natural and effective [11].

### Example 4 - Alphabetic vs Logographic Scripts

> ***Caption.** Contrast an English alphabetic word with a Chinese logographic word to highlight differences in meaning preservation at character level.*
> 
> -   **English: "house"** → character-level tokens: [h, o, u, s, e]
>     Here, each individual character has no semantic meaning on its own; the concept of "house" only emerges from their combination.
> -   **Chinese: "家"** (jiā - house) → character-level tokens: [家]
>     In this case, the character itself already represents the concept of "house/home," preserving semantic meaning at the character level.

``` python
eng = "house"
zh  = "家"  # 'home/house'
print("English characters:", list(eng))
print("Chinese character:", list(zh))
```

    English characters: ['h', 'o', 'u', 's', 'e']
    Chinese character: ['家']

Another alternative is to tokenize by converting every character into bytes using UTF-8 encoding, a method known as **Byte-Level Tokenization**. UTF-8 (Unicode Transformation Format -- 8 bit) is the dominant standard for representing text digitally; it encodes each Unicode character as one to four bytes, allowing all writing systems and symbols to be represented in a compact, backward-compatible form [12]. This approach further reduces the dictionary size, since it only needs to store the 256 possible byte values (plus a few special tokens), rather than maintaining an entry for every possible character or symbol, as illustrated in Example 5.

### Example 5 - Byte-Level UTF‑8: Accents Become Multiple Bytes 

> ***Caption.** In this example of usage of byte-level tokenization, the character \"é\" is divided into its bytes (C3 + A9). With this approach, the vocabulary will be smaller, since it won\'t be necessary to create more entries for each character with accentuation, and the model will only use the byte combination of the letter and the accent.*


``` python
# Function to converting the characters to bytes
def to_hex_bytes(s: str):
    return " ".join([f"{b:02X}" for b in s.encode('utf-8')])

for w in ["cafe", "café", "fiancée", "coração"]:
    print(f"{w:10s} -> {to_hex_bytes(w)}")
```

    cafe       -> 63 61 66 65
    café       -> 63 61 66 C3 A9
    fiancée    -> 66 69 61 6E 63 C3 A9 65
    coração    -> 63 6F 72 61 C3 A7 C3 A3 6F

Even so, the Byte-Level tokenization still carries the ***efficiency*** and ***meaning conservation*** problems of the Character-Level, which make their use far from ideal. If splitting text into whole words does not work, and splitting into individual characters also fails, the answer may lie in an intermediate approach we will explore next: **subword tokenization**.

#
# 4. Subword-Level Tokenization: The sweet spot of modern NLP 

Subword-level tokenization has become the standard for most state-of-the-art language models. A **subword** is a unit of text that is larger than a single character but smaller than a full word, for example, "grass" and "hopper" in grasshopper.

It strikes a balance: word-level methods risk an unmanageably large vocabulary, while character-level approaches are overly granular and slow to train. On top of that, subword tokenization does a much better job at ***preserving meaning***, since many of the chunks it produces correspond to morphemes or recognizable parts of words, helping models keep track of both structure and sense.

These methods are **data-driven**, meaning that the segmentation rules are not predefined but instead learned from large corpora based on statistical patterns of character co-occurrence. To better understand the strengths and weaknesses of word-, character-, and subword-level tokenization, we can contrast these three approaches side by side in table 1:

**Table 1 - Comparison of Tokenization Levels** 
| **Metric** | **Word** | **Character** | **Subword** |
|--------|--------|--------|---------|
| **Coverage** | **Language-Dependent** - Works well for languages with clear word boundaries; fails for those without spaces. | **Excellent** - No OOV words; any word can be formed from known characters. | **Excellent** - Builds words from a mix of characters and frequent subwords. |
| **Efficiency** | **Low** - Large vocabulary sizes. | **Very Low** - Long token sequences; deeper networks and more training needed. | **High** - Balanced vocab size and sequence length. |
| **Consistency** | **Low** - Same word may be segmented differently with punctuation. | **Excellent** - Every character treated uniformly. | **High** - Reduces redundancy across similar words. |
| **Meaning Preservation** | **Moderate** - Breaks semantic units, missing relationships in compounds. | **Language-Dependent** - Poor for alphabetic but good for logographic languages (e.g., Chinese). | **High** - Some subwords align with morphemes (smallest meaning units). |


>***Caption.** This table contrasts word-, character-, and subword-level tokenization according to four key metrics: coverage, efficiency, consistency, and meaning preservation. It highlights how subword tokenization offers a balanced compromise between the vocabulary size and sequence length of word- and character-level approaches.*
>
>***Source.** Authors*

Although subword tokenization methods outperform word- and character-level approaches, they still present limitations. These methods may introduce inconsistencies and occasionally break apart meaningful semantic units. The next sections provide a closer examination of the three most widely used subword algorithms – **Byte Pair Encoding (BPE), WordPiece, and Unigram** – which differ in how tokens are selected and vocabularies are constructed.

##
## 4.1. Byte Pair Encoding (BPE)

This is the most widely used tokenization algorithm [13], adopted in models such as the GPT family (for instance, the Tiktoken tokenizer). The intuition of this tokenization type is simple: BPE builds a vocabulary by iteratively looking for most frequent adjacent symbol pairs and then merging them to store as a new token [2]. 

The process works as follows:

1.  **Initialization:** the algorithm starts with an initial vocabulary that includes all the basic characters and special symbols found in the training data.

2.  **Pair counting:** it scans the corpus and counts all adjacent pairs of symbols.

3.  **Merging:** the most frequent pair of adjacent units is combined into a new token and added to the dictionary.

4.  **Iteration:** steps 2 and 3 are repeated until the target vocabulary size is reached.

This iterative merging produces a dictionary that adapts naturally to the data: frequent words as "people\", are eventually represented as single tokens, while rare words are split into smaller units, as is shown in Example 6 [14, 15].

At the moment of using, also called inference time, the tokenizer applies the learned merge rules greedily – from longest valid merge to shortest – so each span in the input is replaced by the largest token present in the vocabulary. Unseen or rare words naturally back off to smaller subwords and, if necessary, to the base characters. This behavior yields broad ***coverage*** with relatively compact sequences for frequent forms, improving both ***efficiency*** and ***consistency*** of segmentation across similar words.

Implementations differ in how they encode boundaries. In GPT-style BPE vocabularies, a leading "Ġ" on a token denotes that it is preceded by a space (e.g., "Ġlove"), allowing the model to learn word boundaries without a separate pre-tokenizer. This convention appears in the outputs of Example 6 and explains why otherwise identical strings may occur with and without the marker.

A widely used variant, **byte-level BPE** (e.g., GPT-2 and successors), runs BPE over UTF-8 bytes instead of Unicode code points. This guarantees language-agnostic coverage, keeps preprocessing reversible and simple, and reduces the base alphabet to 256 values. The trade-off is that some words split into slightly more pieces and space/byte markers become visible in the vocabulary – harmless in practice, but occasionally surprising to readers unfamiliar with the convention.

### Example 6 - Subword BPE: Merge Frequent Pairs

> ***Caption.** In this example, the output of the GPT-2 tokenizer is shown for two different sentences:*
>
>-   \"The old man wrote a short note to his friend.\" (only common   words)
>-   \"The **fey** creature spoke with **floccinaucinihilipilification** of human worries\" (some rare word)
>
> *The words from the first sentence are treated as unique tokens, since they are more common. However, both of the rare words from the second sentence (the short and the long one) are divided into more than one token, because their sequences of characteres are uncommon. The character \"Ġ\" in some tokens is a marker that indicates a whitespace before the token.*
>
> *You can see the full implementation of Byte-Pair Encoding tokenization, step by step, in the [Hugging Face article](https://huggingface.co/learn/llm-course/chapter6/5) and [codes](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter6/section5.ipynb).*

``` python
# Texts used as examples
texts = ["The old man wrote a short note to his friend.","The fey creature spoke with floccinaucinihilipilification of human worries"]

# Print the tokens
subword_tokenizer(texts, tok_bpe)
```

    Text: The old man wrote a short note to his friend.
    Tokens and their vocabulary indices:

| **#** | **Token** | **Token ID** |
|-------|------------|--------------|
| 0 | The | 464 |
| 1 | Ġold | 1468 |
| 2 | Ġman | 582 |
| 3 | Ġwrote | 2630 |
| 4 | Ġa | 257 |
| 5 | Ġshort | 1790 |
| 6 | Ġnote | 3465 |
| 7 | Ġto | 284 |
| 8 | Ġhis | 465 |
| 9 | Ġfriend | 1545 |
| 10 | . | 13 |

    Text: The fey creature spoke with floccinaucinihilipilification of human worries
    Tokens and their vocabulary indices:

| **#** | **Token** | **Token ID** |
|-------|------------|--------------|
| 0 | The | 464 |
| 1 | Ġfe | 730 |
| 2 | y | 88 |
| 3 | Ġcreature | 7185 |
| 4 | Ġspoke | 5158 |
| 5 | Ġwith | 351 |
| 6 | Ġfl | 781 |
| 7 | occ | 13966 |
| 8 | in | 259 |
| 9 | auc | 14272 |
| 10 | in | 259 |
| 11 | ihil | 20898 |
| 12 | ip | 541 |
| 13 | il | 346 |
| 14 | ification | 2649 |
| 15 | Ġof | 286 |
| 16 | Ġhuman | 1692 |
| 17 | Ġworries | 18572 |


##
## 4.2. WordPiece

The WordPiece model is the basis for the tokenizers used in famous Transformer models, such as **BERT, DistilBERT, and Electra** [2], and it originated in work on Japanese and Korean voice search [16]. The operation of WordPiece can be summarized in the following steps:

1.  **Initialization:** start from a vocabulary with basic characters and special symbols.

2.  **Pair Scoring:** scan the corpus and compute, for every adjacent pair (A, B), a likelihood score, using the following formula:

$$
   Score(A, B) = \frac{freq(AB)}{freq(A) \cdot freq(B)}
$$

3.  **Merging:** merge the highest-scoring pair and add the new token to the dictionary.
4.  **Iteration:** repeat scoring and merging until the dictionary reaches a preset size [17].

The objective here is to prioritize merging pairs of tokens that appear together much more frequently than they would by chance. Then, the main difference between WordPiece and BPE (Byte-Pair Encoding), is precisely the use of this probability-based formula, instead of just using raw frequency counts to decide which pairs to merge.

In statistics, likelihood quantifies how probable the observed data are under a specified model with given parameter values; it is a function of the parameters given the data, not a probability of the parameters themselves, and maximizing it is the core idea behind maximum likelihood estimation [18].

In the context of WordPiece, the "model and parameters" correspond to a candidate subword vocabulary (and the segmentations it enables). At each iteration, WordPiece asks whether merging an adjacent pair **A B** would increase the corpus likelihood under this tokenization model. In practice, this is approximated by a score that favors pairs whose joint frequency is larger than expected from their individual frequencies.

It's also worth pointing out that in this scoring, **order matters:** the score for "AB" will generally differ from the score for "BA," just as "me" has a different chance of appearing in text compared with "em."

This likelihood-driven construction yields a dictionary that adapts naturally to the data: common multi-character sequences (e.g., stems and affixes) tend to become single tokens, while rare or idiosyncratic words break off to smaller pieces – as illustrated in Example 7.

At inference time, WordPiece segments each word into the most probable sequence of subwords under the learned vocabulary; implementations in the BERT family typically mark non-initial subwords with a continuation prefix (e.g., "##ing"), while relying on a word pre-tokenizer to handle spaces and punctuation, which preserves consistency across contexts.

### Example 7 - WordPiece: Likelihood‑Driven Merges 

> **Caption.** In this example, the output of the BERT tokenizers shown for the same two sentences of last example:
>-   \"The old man wrote a short note to his friend.\" (only common words)
>-   \"The **fey** creature spoke with **floccinaucinihilipilification** of human worries\" (some rare word)
> Here, the relation between rarity of words and their number of tokens can be seen. The \"##\" that appears in some tokens marks that the token began in the middle of a word that was segmented.
> You can see the full implementation of WordPiece tokenization, step by step, in the [Hugging Face article](https://huggingface.co/learn/llm-course/chapter6/6) and [codes](https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter6/section6.ipynb).

``` python
# Texts used as examples
texts = ["The old man wrote a short note to his friend.","The fey creature spoke with floccinaucinihilipilification of human worries"]

# Prints the tokens
subword_tokenizer(texts, tok_wp)
```

    Text: The old man wrote a short note to his friend.
    Tokens and their vocabulary indices:

| **#** | **Token** | **Token ID** |
|-------|------------|--------------|
| 0 | the | 1996 |
| 1 | old | 2214 |
| 2 | man | 2158 |
| 3 | wrote | 2626 |
| 4 | a | 1037 |
| 5 | short | 2460 |
| 6 | note | 3602 |
| 7 | to | 2000 |
| 8 | his | 2010 |
| 9 | friend | 2767 |
| 10 | . | 1012 |

    Text: The fey creature spoke with floccinaucinihilipilification of human worries
    Tokens and their vocabulary indices:
    
| **#** | **Token** | **Token ID** |
|-------|------------|--------------|
| 0 | the | 1996 |
| 1 | fey | 23864 |
| 2 | creature | 6492 |
| 3 | spoke | 3764 |
| 4 | with | 2007 |
| 5 | fl | 13109 |
| 6 | ##oc | 10085 |
| 7 | ##cina | 28748 |
| 8 | ##uc | 14194 |
| 9 | ##ini | 5498 |
| 10 | ##hil | 19466 |
| 11 | ##ip | 11514 |
| 12 | ##ili | 18622 |
| 13 | ##fication | 10803 |
| 14 | of | 1997 |
| 15 | human | 2529 |
| 16 | worries | 15508 |


##
## 4.3. Unigram 

The Unigram algorithm is a subword-based tokenization method that, unlike purely heuristic approaches such as BPE, the dictionary constructed in this method is derived on probabilistic grounds, where each subword is assumed to be independent of the others. The primary objective of the algorithm of building the dictionary is to **identify a set of subwords that minimizes the overall loss function** [19] and it works as follow:

1.  **Initialization:** start from a large candidate vocabulary $V_0$ that includes all basic characters and plausible subwords.

2.  **Probability Estimation:** estimate $P(t)$ for each token $t \in V$.
    
$$
   P(t) = \frac{\text{number of times } t \text{ appears in the tokenizations}}{\text{total number of tokens in the tokenizations}}
$$

4.  **Segmentation:** For every word, compute all possible segmentations using $V$, and select the one with the most probable segmentation under the model, typically

$$
   P(\mathbf{w}) = P(t_1) \cdot P(t_2) \cdot \ldots \cdot P(t_n)
$$

4.  **Loss Computation:** Compute the negative log-likelihood (NLL) of the corpus. For each token, estimate the increase in NLL if ( t ) is removed.

$$
   𝓛 = -\sum_{w \in \text{corpus}} \text{freq}(w) \cdot \log(P(w))
$$

5.  **Pruning:** Remove the bottom ( p% ) of tokens (those whose removal least harms likelihood), while preserving basic characters.

6.  **Iteration:** Re-estimate probabilities with the pruned dictionary. Repeat steps 3 - 7 until the dictionary reaches the target size.
