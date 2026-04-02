import os
import pandas as pd
import xml.etree.ElementTree as ET
import re

def parse_and_clean_ami(ami_root_path, output_csv="ami_final_accurate_dataset.csv"):
    extracted_data = []
    words_path = os.path.join(ami_root_path, "words")
    da_folder = os.path.join(ami_root_path, "dialogueActs")

    # 1. Scientific Mapping: AMI Codes to Action Items
    # 11: Suggest, 29: Action-Directive, 15: Elicit-Offer, 12: Assessment/Decision
    ACTION_CODES = ['ami_da_11', 'ami_da_29', 'ami_da_15', 'ami_da_12']
    
    # 2. Heuristic Filter: Action items usually contain these "intent" words
    ACTION_KEYWORDS = [
        'should', 'could', 'will', 'let\'s', 'can', 'must', 'need', 
        'task', 'draw', 'write', 'decide', 'make', 'want to', 'possible',
        'how about', 'what if', 'agree', 'ready'
    ]

    print(f"\n[System] Starting High-Accuracy Pipeline...")
    da_files = [f for f in os.listdir(da_folder) if "dialog-act" in f and f.endswith(".xml")]
    total_files = len(da_files)

    for idx, da_file_name in enumerate(da_files, 1):
        if idx % 50 == 0 or idx == total_files:
            print(f"[Progress] {idx}/{total_files} files processed...")

        meeting_info = ".".join(da_file_name.split('.')[:2]) 
        word_file = os.path.join(words_path, f"{meeting_info}.words.xml")
        if not os.path.exists(word_file): continue

        try:
            # Load Words
            word_tree = ET.parse(word_file)
            word_map = {w.get('{http://nite.sourceforge.net/}id'): w.text 
                        for w in word_tree.getroot() if w.get('{http://nite.sourceforge.net/}id')}

            # Parse Dialogue Acts
            da_tree = ET.parse(os.path.join(da_folder, da_file_name))
            for dact in da_tree.getroot():
                pointer = dact.find('{http://nite.sourceforge.net/}pointer')
                da_code = pointer.get('href', '').split('id(')[-1].replace(')', '') if pointer is not None else ""

                child = dact.find('{http://nite.sourceforge.net/}child')
                if child is not None:
                    ids = re.findall(r'id\((.*?)\)', child.get('href', ''))
                    if ids:
                        # Reconstruct sentence
                        prefix = ids[0].rsplit('words', 1)[0] + 'words'
                        start_num = int(ids[0].split('words')[-1])
                        end_num = int(ids[-1].split('words')[-1])
                        
                        sentence = " ".join([word_map.get(f"{prefix}{i}", "") 
                                           for i in range(start_num, end_num + 1)]).strip()

                        # --- ACCURACY FILTERS ---
                        word_count = len(sentence.split())
                        if word_count > 3: # Ignore very short fragments
                            
                            is_action_type = any(ac in da_code for ac in ACTION_CODES)
                            has_action_word = any(kw in sentence.lower() for kw in ACTION_KEYWORDS)
                            
                            # Final Labeling Logic
                            if is_action_type and has_action_word:
                                label = "action_item"
                            else:
                                label = "information_item"
                                
                            extracted_data.append({"text": sentence, "label": label})
        except Exception:
            continue

    # Save and Report
    if extracted_data:
        df = pd.DataFrame(extracted_data)
        # Drop duplicates to ensure quality
        df = df.drop_duplicates(subset=['text'])
        
        save_path = os.path.join(os.path.dirname(ami_root_path), output_csv)
        df.to_csv(save_path, index=False)
        
        print("\n" + "="*40)
        print(f"FINAL DATASET QUALITY REPORT")
        print(f"Total Rows: {len(df)}")
        print(f"Action Items: {len(df[df['label'] == 'action_item'])}")
        print(f"Information Items: {len(df[df['label'] == 'information_item'])}")
        print(f"Saved to: {save_path}")
        print("="*40)

if __name__ == "__main__":
    ami_path = r"C:\Users\John Keith Mercado\ThesisModel\data\ami_public_manual_1.6.2"
    parse_and_clean_ami(ami_path)