import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.svm import SVC

# --- INITIALIZATION ---git push -u origin main
nlp = spacy.load("en_core_web_sm")
seg_model = SentenceTransformer('all-mpnet-base-v2')

# --- PHASE 1: LINGUISTIC FEATURE FUNCTION ---
def get_features(sentence):
    text = sentence.lower()
    doc = nlp(text)
    return [
        1 if 'paki' in text else 0,              # Taglish Politeness
        1 if 'sige' in text else 0,              # Taglish Agreement
        1 if doc[0].pos_ == "VERB" else 0,       # Imperative
        1 if any(m in text for m in ['need', 'must', 'will']) else 0 # Modals
    ]

# --- PHASE 3: PRE-TRAINED MOCK SVM ---
# (In a real thesis, you train this on your i-TANONG/AMI CSV first)
clf = SVC(kernel='linear')
# Training on small sample to "prime" the model
train_X = [[1,0,0,0], [0,0,0,0], [0,1,0,1], [0,0,0,0]]
train_y = [1, 0, 1, 0] 
clf.fit(train_X, train_y)

# --- THE PROCESS ---
raw_paragraph = "Okay let's start the meeting by discussing the budget for the third quarter because we have about fifty thousand pesos left for marketing and the financial report shows we overspent on social media ads so paki-send naman the minutes tomorrow for this part. Moving on to the next item which is the office renovation we need to buy new chairs and paint the walls blue and the contractor said the renovation will take two weeks so sige ako na bahala sa slides for that update. Lastly let's talk about the company Christmas party where we are thinking of holding it at a beach resort in Batangas so baka pwedeng paki-check yung price of the rooms."

# Step 1: Sentence Splitting (Basic)
sentences = [s.strip() for s in raw_paragraph.replace('so ', '. ').split('.') if len(s) > 5]

# --- PHASE 2: TOPIC SEGMENTATION (With Process Output) ---
print("--- PHASE 2: SEGMENTATION PROCESS ---")
embeddings = seg_model.encode(sentences, convert_to_tensor=True)
topic_chunks = []
temp_chunk = [sentences[0]]

for i in range(len(sentences) - 1):
    # This is the "Process" - calculating how related sentence A is to sentence B
    similarity = util.cos_sim(embeddings[i], embeddings[i+1]).item()
    
    print(f"Comparing Sentence {i} & {i+1}:")
    print(f"  > Similarity Score: {similarity:.4f}")
    
    # If the score is low (e.g., < 0.45), the topic changed
    if similarity < 0.45:
        print(f"  [!] SHIFT DETECTED: Score is below threshold. Creating new topic.\n")
        topic_chunks.append(temp_chunk)
        temp_chunk = [sentences[i+1]]
    else:
        print(f"  [+] CONTINUING: High similarity. Staying in same topic.\n")
        temp_chunk.append(sentences[i+1])

topic_chunks.append(temp_chunk)
print(f"Total Topics Identified: {len(topic_chunks)}")

# PHASE 3: ACTION ITEM DETECTION (Linear SVM)
print("\n--- PHASE 3: ACTION ITEM DETECTION ---")
for i, chunk in enumerate(topic_chunks):
    print(f"\n[TOPIC {i+1}]")
    for sent in chunk:
        feat = [get_features(sent)]
        is_action = clf.predict(feat)[0]
        
        status = " [!] ACTION ITEM" if is_action == 1 else " [ ] Info"
        print(f"{status}: {sent}")