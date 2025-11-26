"""
SIMPLE TRAINING SCRIPT FOR PATIENT GENERATOR GAN

Just run this file to train the model!

Usage:
    python train_gan.py
"""

from numpy import test
from patient_generator_gan import PatientGenerator

print("\n" + "="*70)
print("SYNTHETIC PATIENT GENERATOR - TRAINING")
print("="*70)

# Step 1: Initialize generator with your data
gen = PatientGenerator(data_path='preprocessed_nhanes/nhanes_final_complete.csv')

# Step 2: Train the model
gen.train(
    epochs=400,              # How many times to go through data
    batch_size=128,          # Process 128 patients at a time
    lr=0.0002,               # Learning rate (0.0002 is good for GANs)
    save_every=50,           # Save checkpoint every 50 epochs
    show_progress_every=10   # Print progress every 10 epochs
)

# Step 3: Validate the generator
print("\n" + "="*70)
print("TESTING GENERATOR")
print("="*70)
print("\nTesting with sample inputs (45yo male, 175cm, 80kg)...")

gen.validate_generation(
    age=45,
    sex='M',
    height=175,
    weight=80,
    n=100,
    seed=42
)

# Step 4: Generate a few test patients
print("="*70)
print("GENERATING TEST PATIENTS")
print("="*70)
print("\nGenerating 5 test patients...")

test_patients = gen.generate(
    age=45,
    sex='M',
    height=175,
    weight=80,
    n=5,
    seed=42
)

print("\nGenerated patients (first 5):")
print("\nBasic info:")
print(test_patients[['AGE', 'SEX', 'BMXHT', 'BMXWT', 'BMXBMI']].to_string(index=False))

print("\nSample body measurements:")
body_cols = ['BMXARMC', 'BMXARML', 'BMXLEG', 'BMXWAIST']
print(test_patients[body_cols].to_string(index=False))

print("\nSample lab results:")
lab_cols = ['LBXTC', 'LBXSGL', 'LBXCRP', 'LBXSNA']
print(test_patients[lab_cols].to_string(index=False))

print("\nSample questionnaire:")
quest_cols = ['MCQ160B', 'MCQ160C', 'MCQ160E']
print(test_patients[quest_cols].to_string(index=False))

print("\n" + "="*70)
print("[OK] Training complete! Model ready to use.")
print("="*70)
print("\nNext steps:")
print("  1. Use use_generator.py for more examples")
print("  2. Generate patients: gen.generate(age, sex, height, weight, n)")
print("  3. Model saved at: models/patient_generator_gan.pt")
print("="*70 + "\n")
