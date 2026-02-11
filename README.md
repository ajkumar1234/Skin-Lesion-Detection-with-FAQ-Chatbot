Medical AI is different from typical computer vision problems. A wrong prediction isn’t just a number — it’s a missed diagnosis. With that mindset, I built a Skin Lesion Detection system using deep learning, focusing not only on model performance but also on real-world clinical constraints.

This project involved designing an end-to-end medical imaging pipeline, starting from noisy dermoscopic images to reliable classification outputs that prioritize patient safety.

🔍 What I Built

A CNN-based image classification system to identify different types of skin lesions from dermoscopic images

Used ResNet18 and EfficientNetB0 with ImageNet pre-trained weights to leverage transfer learning in a data-limited medical domain

Implemented a robust preprocessing pipeline including normalization and data augmentation to improve generalization

⚙️ Engineering Decisions That Mattered

Chose transfer learning over training from scratch due to limited labeled medical data

Focused on recall and F1-score, not just accuracy, to reduce false negatives in critical classes

Used confusion matrix analysis to understand model behavior beyond aggregate metrics

🧠 What This Project Taught Me

Why medical AI models must be evaluated differently than standard ML systems

How class imbalance can silently bias models if not handled properly

The importance of explainable evaluation when building trust in AI-assisted healthcare systems

🚀 Outcome

The final system demonstrates how deep learning can assist clinicians by acting as a decision-support tool, highlighting suspicious lesions while reducing diagnostic subjectivity.
