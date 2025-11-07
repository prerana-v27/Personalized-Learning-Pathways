import pandas as pd

df = pd.read_csv('Data/processed/all_courses_cleaned.csv')
musical = df[df['subject'].str.contains('musical', case=False, na=False)]

print(f'Total musical instrument courses: {len(musical)}')
print('\nSample courses:')
for idx, row in musical.head(10).iterrows():
    print(f"\n{row['title'][:80]}")
    print(f"  Subject: {row['subject']}")
    print(f"  Difficulty: {row['difficulty']}")
    print(f"  Description: {row['description'][:150]}...")
    
print('\n\nChecking if "drums" appears in any course text:')
drums = df[df['text'].str.contains('drums', case=False, na=False)]
print(f'Courses mentioning "drums": {len(drums)}')
if len(drums) > 0:
    print('\nFirst 5 courses with "drums":')
    for idx, row in drums.head(5).iterrows():
        print(f"  - {row['title'][:80]} ({row['subject']})")
