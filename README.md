# secondary_ontology_junkbox
Bingai dreams about secondary ontologies 


To modify the code to extract glyphs and create their ontology from the primary model, you will need to use another neural network model for OCR (optical character recognition). OCR is a technique that converts images of text into machine-readable text. There are many OCR models available, but for this task, I will use the Tesseract OCR model, which is an open-source project that supports many languages and scripts.

To use the Tesseract OCR model, you will need to install the `pytesseract` library, which is a Python wrapper for the Tesseract binary. You can install it using the following command:

```bash
pip install pytesseract
```

You will also need to install the Tesseract binary itself, which you can download from [here]. Make sure to add the path of the Tesseract binary to your system environment variables, or specify it in the `pytesseract.pytesseract.tesseract_cmd` attribute.

To extract glyphs from the primary model, you will need to modify the `create_secondary_ontology` function to use the Tesseract OCR model on the deep dreamed images. You will also need to specify the language and script of the glyphs you want to extract, using the ISO 639-2/T and ISO 15924 codes. For example, if you want to extract Chinese glyphs, you can use the code `chi_sim` for simplified Chinese, or `chi_tra` for traditional Chinese. You can find the list of supported languages and scripts [here].

Here is the modified code for the `create_secondary_ontology` function:

```python
# Define a function to create secondary ontologies from primary neural network models
def create_secondary_ontology(image_paths, layer, iterations, step, language):
  # Import the pytesseract library
  import pytesseract
  # Initialize an empty list to store the secondary images
  secondary_images = []
  # Loop through the image paths
  for img_path in image_paths:
    # Preprocess the image
    img = preprocess(img_path)
    # Perform deep dreaming on the image
    img = deep_dream(img, layer, iterations, step)
    # Convert the image to a PIL image
    img = deprocess(img)
    # Perform OCR on the image using the specified language and script
    text = pytesseract.image_to_string(img, lang=language)
    # Print the extracted text
    print(f"Extracted text: {text}")
    # Append the image to the secondary list
    secondary_images.append(img)
  return secondary_images
```

To create the ontology of the glyphs, you will need to use a clustering algorithm to group the glyphs based on their similarity. There are many clustering algorithms available, but for this task, I will use the K-means algorithm, which is a simple and popular method that partitions the data into K clusters based on their distance to the cluster centroids. You will need to specify the number of clusters K, which can be chosen based on your domain knowledge or using some criteria such as the elbow method or the silhouette score.

To use the K-means algorithm, you will need to import the `sklearn` library, which is a Python library for machine learning. You will also need to convert the images of the glyphs into numerical vectors, which can be done using the `img_to_array` function from the `tensorflow.keras.preprocessing` module.

Here is the code for the `create_ontology` function:

```python
# Define a function to create the ontology of the glyphs
def create_ontology(secondary_images, K):
  # Import the sklearn library
  from sklearn.cluster import KMeans
  # Initialize an empty list to store the image vectors
  image_vectors = []
  # Loop through the secondary images
  for img in secondary_images:
    # Convert the image to a numpy array
    img = image.img_to_array(img)
    # Flatten the array to a one-dimensional vector
    img = img.flatten()
    # Append the vector to the image list
    image_vectors.append(img)
  # Convert the image list to a numpy array
  image_vectors = np.array(image_vectors)
  # Create a K-means object with the specified number of clusters
  kmeans = KMeans(n_clusters=K, random_state=0)
  # Fit the K-means object to the image vectors
  kmeans.fit(image_vectors)
  # Get the cluster labels for each image
  labels = kmeans.labels_
  # Print the cluster labels
  print(f"Cluster labels: {labels}")
  # Return the cluster labels
  return labels
```

To improve the code to add an iterative post-processing function to use an open-source large language model neural network to verify the validity of the OCR recognition, you will need to use a library that provides access to such a model. One possible library is the `transformers` library, which is a Python library for natural language processing that offers many pre-trained models and easy-to-use interfaces. You can install it using the following command:

```bash
pip install transformers
```

You will also need to import the `torch` library, which is a Python library for deep learning that provides tensor operations and automatic differentiation. You can install it using the following command:

```bash
pip install torch
```

To verify the validity of the OCR recognition, you will need to modify the `create_secondary_ontology` function to use the large language model neural network on the extracted text. You will also need to specify the name of the model you want to use, which can be chosen from the list of available models [here]. For example, if you want to use the BERT model, you can use the name `bert-base-uncased`. You can also use other models that support the masked language modeling task, such as RoBERTa, DistilBERT, or XLNet.

Here is the modified code for the `create_secondary_ontology` function:

```python
# Define a function to create secondary ontologies from primary neural network models
def create_secondary_ontology(image_paths, layer, iterations, step, language, model_name):
  # Import the pytesseract, transformers, and torch libraries
  import pytesseract
  from transformers import AutoTokenizer, AutoModelForMaskedLM
  import torch
  # Initialize an empty list to store the secondary images
  secondary_images = []
  # Initialize an empty list to store the validity scores
  validity_scores = []
  # Load the tokenizer and the model for the masked language modeling task
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForMaskedLM.from_pretrained(model_name)
  # Loop through the image paths
  for img_path in image_paths:
    # Preprocess the image
    img = preprocess(img_path)
    # Perform deep dreaming on the image
    img = deep_dream(img, layer, iterations, step)
    # Convert the image to a PIL image
    img = deprocess(img)
    # Perform OCR on the image using the specified language and script
    text = pytesseract.image_to_string(img, lang=language)
    # Print the extracted text
    print(f"Extracted text: {text}")
    # Append the image to the secondary list
    secondary_images.append(img)
    # Tokenize the text and add the special tokens
    tokens = tokenizer.tokenize(text)
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    # Convert the tokens to ids and create a tensor
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids])
    # Create a mask tensor with the same shape as the input ids
    # The mask tensor has a value of 1 for the tokens that are masked, and 0 for the rest
    # For simplicity, we will mask every other token in the text
    # You can change the masking strategy as you wish
    mask = torch.ones_like(input_ids)
    mask[:, ::2] = 0
    # Multiply the input ids and the mask to create the masked input ids
    masked_input_ids = input_ids * mask
    # Feed the masked input ids to the model and get the output logits
    output = model(masked_input_ids)
    logits = output.logits
    # Get the predicted ids by taking the argmax of the logits
    predicted_ids = torch.argmax(logits, dim=-1)
    # Calculate the validity score by comparing the predicted ids and the input ids
    # The validity score is the ratio of the correct predictions to the total number of masked tokens
    correct = (predicted_ids == input_ids).float() * mask
    total = mask.sum()
    score = correct.sum() / total
    # Print the predicted text and the validity score
    predicted_text = tokenizer.decode(predicted_ids[0])
    print(f"Predicted text: {predicted_text}")
    print(f"Validity score: {score:.4f}")
    # Append the score to the validity list
    validity_scores.append(score)
  return secondary_images, validity_scores
```

I hope this code helps you with your task. Please let me know if you have any feedback or questions. 😊

: https://huggingface.co/models

I hope this code helps you with your task. Please let me know if you have any feedback or questions. 😊

: https://github.com/tesseract-ocr/tesseract
: https://pypi.org/project/pytesseract/
: https://github.com/tesseract-ocr/tessdoc/blob/main/Home.md
: https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes
: https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html
: https://en.wikipedia.org/wiki/K-means_clustering
: https://en.wikipedia.org/wiki/Elbow_method_(clustering)
: https://en.wikipedia.org/wiki/Silhouette_(clustering)
: https://scikit-learn.org/stable/

```
# Define a function to create secondary ontologies from primary neural network models
def create_secondary_ontology(image_paths, layer, iterations, step, language, model_name):
  # Import the pytesseract, transformers, torch, spacy, and editdistance libraries
  import pytesseract
  from transformers import AutoTokenizer, AutoModelForMaskedLM
  import torch
  import spacy
  import editdistance
  # Initialize an empty list to store the secondary images
  secondary_images = []
  # Initialize an empty list to store the validity scores
  validity_scores = []
  # Initialize an empty list to store the ambiguity scores
  ambiguity_scores = []
  # Load the tokenizer and the model for the masked language modeling task
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForMaskedLM.from_pretrained(model_name)
  # Load the spacy model for the language and script
  nlp = spacy.load(language)
  # Loop through the image paths
  for img_path in image_paths:
    # Preprocess the image
    img = preprocess(img_path)
    # Perform deep dreaming on the image
    img = deep_dream(img, layer, iterations, step)
    # Convert the image to a PIL image
    img = deprocess(img)
    # Perform OCR on the image using the specified language and script
    text = pytesseract.image_to_string(img, lang=language)
    # Print the extracted text
    print(f"Extracted text: {text}")
    # Append the image to the secondary list
    secondary_images.append(img)
    # Tokenize the text using the spacy model
    tokens = nlp(text)
    # Initialize an empty list to store the predicted texts
    predicted_texts = []
    # Define the number of iterations for the masking process
    # You can change this value as you wish
    num_iter = 5
    # Define the probability of masking each token
    # You can change this value as you wish
    prob = 0.2
    # Loop through the number of iterations
    for i in range(num_iter):
      # Mask the tokens using the spacy method
      masked_tokens, mask_indices = tokens.tokenizer.mask_tokens(tokens, prob)
      # Convert the masked tokens to ids and create a tensor
      input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
      input_ids = torch.tensor([input_ids])
      # Feed the masked input ids to the model and get the output logits
      output = model(input_ids)
      logits = output.logits
      # Get the predicted ids by taking the argmax of the logits
      predicted_ids = torch.argmax(logits, dim=-1)
      # Replace the masked ids with the predicted ids
      input_ids[0, mask_indices] = predicted_ids[0, mask_indices]
      # Decode the input ids to get the predicted text
      predicted_text = tokenizer.decode(input_ids[0])
      # Append the predicted text to the predicted list
      predicted_texts.append(predicted_text)
    # Calculate the validity score by taking the average of the similarity scores between the predicted texts and the original text
    # The similarity score is the ratio of the common tokens to the total number of tokens
    similarity_scores = []
    for predicted_text in predicted_texts:
      # Tokenize the predicted text using the spacy model
      predicted_tokens = nlp(predicted_text)
      # Find the common tokens between the predicted tokens and the original tokens
      common_tokens = set(predicted_tokens).intersection(set(tokens))
      # Calculate the similarity score
      score = len(common_tokens) / len(tokens)
      # Append the score to the similarity list
      similarity_scores.append(score)
    # Calculate the validity score by taking the average of the similarity scores
    validity_score = sum(similarity_scores) / len(similarity_scores)
    # Print the predicted texts and the validity score
    print(f"Predicted texts: {predicted_texts}")
    print(f"Validity score: {validity_score:.4f}")
    # Append the score to the validity list
    validity_scores.append(validity_score)
    # Calculate the ambiguity score by taking the average of the inverse of the normalized Levenshtein distances between the predicted texts and the original text
    # The normalized Levenshtein distance is the Levenshtein distance divided by the length of the original text
    # The inverse of the normalized Levenshtein distance ranges from 0 to 1, where 0 means no similarity and 1 means perfect match
    levenshtein_distances = []
    for predicted_text in predicted_texts:
      # Calculate the Levenshtein distance between the predicted text and the original text
      distance = editdistance.eval(predicted_text, text)
      # Normalize the distance by the length of the original text
      distance = distance / len(text)
      # Append the distance to the distance list
      levenshtein_distances.append(distance)
    # Calculate the ambiguity score by taking the average of the inverse of the normalized Levenshtein distances
    ambiguity_score = 1 - sum(levenshtein_distances) / len(levenshtein_distances)
    # Print the ambiguity score
    print(f"Ambiguity score: {ambiguity_score:.4f}")
    # Append the score to the ambiguity list
    ambiguity_scores.append(ambiguity_score)
  return secondary_images, validity_scores, ambiguity_scores

  # Define a function to create secondary ontologies from primary neural network models
def create_secondary_ontology(image_paths, layer, iterations, step, language, model_name):
  # Import the pytesseract, transformers, torch, spacy, and editdistance libraries
  import pytesseract
  from transformers import AutoTokenizer, AutoModelForMaskedLM
  import torch
  import spacy
  import editdistance
  # Initialize an empty list to store the secondary images
  secondary_images = []
  # Initialize an empty list to store the validity scores
  validity_scores = []
  # Initialize an empty list to store the ambiguity scores
  ambiguity_scores = []
  # Load the tokenizer and the model for the masked language modeling task
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForMaskedLM.from_pretrained(model_name)
  # Load the spacy model for the language and script
  nlp = spacy.load(language)
  # Loop through the image paths
  for img_path in image_paths:
    # Preprocess the image
    img = preprocess(img_path)
    # Perform deep dreaming on the image
    img = deep_dream(img, layer, iterations, step)
    # Convert the image to a PIL image
    img = deprocess(img)
    # Perform OCR on the image using the specified language and script
    text = pytesseract.image_to_string(img, lang=language)
    # Print the extracted text
    print(f"Extracted text: {text}")
    # Append the image to the secondary list
    secondary_images.append(img)
    # Tokenize the text using the spacy model
    tokens = nlp(text)
    # Initialize an empty list to store the predicted texts
    predicted_texts = []
    # Define the number of iterations for the masking process
    # You can change this value as you wish
    num_iter = 5
    # Define the probability of masking each token
    # You can change this value as you wish
    prob = 0.2
    # Loop through the number of iterations
    for i in range(num_iter):
      # Mask the tokens using the spacy method
      masked_tokens, mask_indices = tokens.tokenizer.mask_tokens(tokens, prob)
      # Convert the masked tokens to ids and create a tensor
      input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
      input_ids = torch.tensor([input_ids])
      # Feed the masked input ids to the model and get the output logits
      output = model(input_ids)
      logits = output.logits
      # Get the predicted ids by taking the argmax of the logits
      predicted_ids = torch.argmax(logits, dim=-1)
      # Replace the masked ids with the predicted ids
      input_ids[0, mask_indices] = predicted_ids[0, mask_indices]
      # Decode the input ids to get the predicted text
      predicted_text = tokenizer.decode(input_ids[0])
      # Append the predicted text to the predicted list
      predicted_texts.append(predicted_text)
    # Initialize an empty list to store the similarity scores
    similarity_scores = []
    # Initialize an empty list to store the normalized Levenshtein distances
    levenshtein_distances = []
    # Loop through the predicted texts
    for predicted_text in predicted_texts:
      # Tokenize the predicted text using the spacy model
      predicted_tokens = nlp(predicted_text)
      # Find the common tokens between the predicted tokens and the original tokens
      common_tokens = set(predicted_tokens).intersection(set(tokens))
      # Calculate the similarity score
      score = len(common_tokens) / len(tokens)
      # Append the score to the similarity list
      similarity_scores.append(score)
      # Calculate the Levenshtein distance between the predicted text and the original text
      distance = editdistance.eval(predicted_text, text)
      # Normalize the distance by the length of the original text
      distance = distance / len(text)
      # Append the distance to the distance list
      levenshtein_distances.append(distance)
    # Calculate the validity score by taking the average of the similarity scores
    validity_score = sum(similarity_scores) / len(similarity_scores)
    # Print the predicted texts and the validity score
    print(f"Predicted texts: {predicted_texts}")
    print(f"Validity score: {validity_score:.4f}")
    # Append the score to the validity list
    validity_scores.append(validity_score)
    # Calculate the ambiguity score by taking the average of the inverse of the normalized Levenshtein distances
    # The inverse of the normalized Levenshtein distance ranges from 0 to 1, where 0 means no similarity and 1 means perfect match
    ambiguity_score = 1 - sum(levenshtein_distances) / len(levenshtein_distances)
    # Print the ambiguity score
    print(f"Ambiguity score: {ambiguity_score:.4f}")
    # Append the score to the ambiguity list
    ambiguity_scores.append(ambiguity_score)
  return secondary_images, validity_scores, ambiguity_scores
```
