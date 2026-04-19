#!/usr/bin/env python
"""Test script to verify ActionItemClassifier mode constants are accessible."""

import sys
import os

print("[Test] Python executable:", sys.executable)
print("[Test] Python version:", sys.version)
print("[Test] Current working directory:", os.getcwd())

try:
    print("\n[Test] Attempting to import ActionItemClassifier...")
    # Import directly to bypass core/__init__.py which requires faster_whisper
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("classifier", "core/classifier.py")
    classifier_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classifier_module)
    ActionItemClassifier = classifier_module.ActionItemClassifier
    
    print("[Test] Import successful!")
    print("[Test] ActionItemClassifier class:", ActionItemClassifier)
    
    print("\n[Test] Checking class attributes...")
    print("[Test] dir(ActionItemClassifier):", [x for x in dir(ActionItemClassifier) if 'MODE' in x])
    
    print("\n[Test] Checking MODE constants...")
    if hasattr(ActionItemClassifier, 'MODE_BALANCED'):
        print("[SUCCESS] MODE_BALANCED exists:", ActionItemClassifier.MODE_BALANCED)
    else:
        print("[ERROR] MODE_BALANCED does NOT exist")
    
    if hasattr(ActionItemClassifier, 'MODE_HIGH_RECALL'):
        print("[SUCCESS] MODE_HIGH_RECALL exists:", ActionItemClassifier.MODE_HIGH_RECALL)
    else:
        print("[ERROR] MODE_HIGH_RECALL does NOT exist")
    
    print("\n[Test] Creating classifier instance...")
    c = ActionItemClassifier()
    print("[Test] Classifier instance created successfully")
    print("[Test] Instance operating_mode:", c.operating_mode)
    print("[Test] Instance get_operating_threshold():", c.get_operating_threshold())
    
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
