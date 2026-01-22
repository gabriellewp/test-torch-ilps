# BERT Similarity Calculator

A simple PyTorch-based information retrieval program using BERT for document/sentence similarity calculation. This project demonstrates how to use pre-trained BERT models for semantic similarity tasks.

## Features

- 🤖 Uses DistilBERT (a smaller, faster variant of BERT) for efficient similarity calculations
- 📊 Calculates cosine similarity between texts using BERT embeddings
- 🔍 Finds the most similar documents from a collection given a query
- 🚀 GPU support for faster inference (automatically uses CUDA if available)
- 📝 Well-commented, easy-to-understand code
- 💡 Practical examples included

## Requirements

- Python 3.8 or higher
- PyTorch 2.0+
- Transformers 4.30+
- NumPy 1.24+
- scikit-learn 1.3+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gabriellewp/test-torch-ilps.git
cd test-torch-ilps
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

The first time you run the program, it will automatically download the DistilBERT model (~250MB).

## Usage

### Basic Usage

Run the example script to see the BERT similarity calculator in action:

```bash
python bert_similarity.py
```

### Using in Your Code

#### 1. Calculate similarity between two texts

```python
from bert_similarity import BERTSimilarityCalculator

# Initialize the calculator
calculator = BERTSimilarityCalculator()

# Calculate similarity
text1 = "The cat sat on the mat"
text2 = "A feline rested on the rug"
similarity = calculator.calculate_similarity(text1, text2)

print(f"Similarity: {similarity:.4f}")
```

#### 2. Find most similar documents from a collection

```python
from bert_similarity import BERTSimilarityCalculator

# Initialize the calculator
calculator = BERTSimilarityCalculator()

# Define your query and document collection
query = "machine learning and artificial intelligence"
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing helps computers understand human language",
    "I love cooking pasta and pizza",
    "Reinforcement learning is inspired by behavioral psychology",
]

# Find top 3 most similar documents
results = calculator.find_most_similar(query, documents, top_k=3)

# Display results
for rank, (idx, doc, score) in enumerate(results, 1):
    print(f"{rank}. [Score: {score:.4f}] {doc}")
```

#### 3. Encode text to embeddings

```python
from bert_similarity import BERTSimilarityCalculator

calculator = BERTSimilarityCalculator()

# Encode a single text
embedding = calculator.encode_text("This is a sample sentence")
print(f"Embedding shape: {embedding.shape}")  # (1, 768)

# Encode multiple texts
texts = ["First sentence", "Second sentence", "Third sentence"]
embeddings = calculator.encode_texts(texts)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 768)
```

## Example Output

When you run `python bert_similarity.py`, you'll see output similar to:

```
================================================================================
BERT Similarity Calculator - Example Usage
================================================================================

Loading model: distilbert-base-uncased...
Model loaded successfully on cpu

--------------------------------------------------------------------------------
Example 1: Calculating similarity between two sentences
--------------------------------------------------------------------------------

Text 1: 'The cat sat on the mat'
Text 2: 'A feline rested on the rug'
Similarity: 0.8245

Text 1: 'The cat sat on the mat'
Text 3: 'Python is a programming language'
Similarity: 0.3421

--------------------------------------------------------------------------------
Example 2: Finding most similar documents to a query
--------------------------------------------------------------------------------

Query: 'machine learning and artificial intelligence'

Searching through 10 documents...
Encoding 10 documents...

Top 5 most similar documents:

1. [Score: 0.9134] Machine learning is a subset of artificial intelligence
2. [Score: 0.8567] Reinforcement learning is inspired by behavioral psychology
3. [Score: 0.8421] Deep learning uses neural networks with multiple layers
4. [Score: 0.7923] Natural language processing helps computers understand human language
5. [Score: 0.7645] Data science involves statistical analysis and visualization

--------------------------------------------------------------------------------
Example 3: Semantic search with different query phrasings
--------------------------------------------------------------------------------

Query: 'simple coding language'
Top 3 matches:
  1. [Score: 0.7856] Python is a high-level programming language known for its simplicity
  2. [Score: 0.7234] Ruby is known for its elegant syntax and the Rails framework
  3. [Score: 0.6891] JavaScript is widely used for web development and runs in browsers

Query: 'browser scripting'
Top 3 matches:
  1. [Score: 0.8123] JavaScript is widely used for web development and runs in browsers
  2. [Score: 0.6234] Python is a high-level programming language known for its simplicity
  3. [Score: 0.5987] Go is a statically typed language designed for concurrent programming

Query: 'fast system programming'
Top 3 matches:
  1. [Score: 0.7945] C++ provides low-level memory manipulation and high performance
  2. [Score: 0.7123] Go is a statically typed language designed for concurrent programming
  3. [Score: 0.6345] Java is an object-oriented language used for enterprise applications

================================================================================
Examples completed successfully!
================================================================================
```

## How It Works

1. **Model**: Uses DistilBERT, a distilled version of BERT that is 40% smaller and 60% faster while retaining 97% of BERT's language understanding capabilities.

2. **Embeddings**: Text is tokenized and passed through the BERT model. The output token embeddings are pooled (averaged) to create a fixed-size sentence representation.

3. **Similarity**: Cosine similarity is calculated between embedding vectors to determine how semantically similar two texts are (score ranges from -1 to 1, typically 0 to 1).

4. **Search**: For document retrieval, the query is encoded and compared against all document embeddings to find the most similar matches.

## Technical Details

- **Mean Pooling**: The implementation uses mean pooling over token embeddings (excluding padding tokens) to create sentence embeddings. This is more effective than using just the CLS token for similarity tasks.

- **GPU Acceleration**: Automatically detects and uses CUDA-enabled GPU if available for faster inference.

- **Batch Processing**: The `encode_texts` method processes documents in batches (default batch size of 8) for better performance on large collections.

## Customization

You can use different pre-trained models by specifying the model name:

```python
# Use standard BERT
calculator = BERTSimilarityCalculator(model_name="bert-base-uncased")

# Use a smaller model for faster inference
calculator = BERTSimilarityCalculator(model_name="distilbert-base-uncased")

# Use a multilingual model
calculator = BERTSimilarityCalculator(model_name="distilbert-base-multilingual-cased")
```

## Use Cases

- **Semantic search**: Find relevant documents based on meaning rather than just keywords
- **Question answering**: Match questions to similar FAQs
- **Document clustering**: Group similar documents together
- **Duplicate detection**: Find near-duplicate content
- **Recommendation systems**: Suggest similar items based on descriptions

## Limitations

- **Context length**: Limited to 512 tokens (approximately 400-500 words)
- **Performance**: Encoding is slower than traditional methods (e.g., TF-IDF) but provides better semantic understanding
- **Domain-specific**: May require fine-tuning for specialized domains
- **Language**: DistilBERT base model is English-only (use multilingual models for other languages)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Uses [HuggingFace Transformers](https://huggingface.co/transformers/)
- Model: [DistilBERT](https://huggingface.co/distilbert-base-uncased) by HuggingFace