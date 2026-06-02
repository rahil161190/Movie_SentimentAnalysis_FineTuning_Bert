# Fine-Tuning BERT for IMDB Sentiment Analysis

A quick, resource-efficient pipeline to fine-tune `bert-base-uncased` on IMDB movie reviews to classify them as either **Positive** or **Negative**. By combining Hugging Face tools with a downsampled dataset, this setup cuts training time down to minutes while maintaining high accuracy.

## How it Works (`bert-base-uncased`)
* **Bidirectional Context:** Unlike older models that read left-to-right, BERT looks at the whole sentence at once to catch context and sarcasm.
* **Base Size:** Uses the standard 12-layer, 110M parameter model to balance accuracy and processing speed.
* **Uncased:** Automatically converts all text to lowercase, meaning `Great`, `GREAT`, and `great` are treated identically to save vocabulary space.

## The Challenge
Training a text classifier from scratch usually hits three walls:
1. **Unknown Words (OOV):** Typos or slang crash traditional tokenizers or get wiped out as an `[UNK]` tag.
2. **Heavy Compute:** Processing thousands of long movie reviews takes hours of expensive GPU time.
3. **Data Requirements:** Models need millions of sentences to learn English syntax from scratch; the 25k IMDB dataset isn't enough.

## The Workaround (Steps Taken)
Instead of starting from zero, this project builds on Google's pre-trained weights:
* **WordPiece Tokenization:** Used BERT's native tokenizer to split unseen words into subwords (e.g., `"scubadiving"` becomes `['scuba', '##di', '##ving']`), completely bypassing OOV errors.
* **Swapping the Head:** Loaded `BertForSequenceClassification`, which safely drops BERT's original language-modeling head and attaches a fresh, uninitialized layer mapped specifically to **2 outputs (Positive and Negative)**.
* **Data Downsampling:** Configured the Hugging Face `Trainer` API to run on a random slice of just **2,000 training** and **500 evaluation** samples for a single epoch.

## Results
* **Evaluation Accuracy:** **89.6%**

Taking a shortcut by training on just 2,000 samples for one epoch still yielded a near-90% accurate sentiment classifier because BERT already understands English grammar—it just needed a quick cheat-sheet on sorting good reviews from bad ones.

