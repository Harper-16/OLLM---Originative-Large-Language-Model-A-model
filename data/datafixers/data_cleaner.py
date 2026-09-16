import csv

path = "../kaggle/dolly_15k.csv"
output = "../fixedata.txt"

seen = set()
added = 0
skipped = 0

# Read existing User: questions so we don't add duplicates
try:
    with open(output, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("User:"):
                seen.add(line[5:].strip().lower())
except FileNotFoundError:
    pass

with open(path, "r", encoding="utf-8", newline="") as infile, \
     open(output, "a", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)

    for row in reader:
        question = row.get("instruction", "").strip()
        answer = row.get("response", "").strip()

        # Skip only if missing question or answer
        if not question or not answer:
            skipped += 1
            continue

        # Skip duplicates
        key = question.lower().strip()

        if key in seen:
            skipped += 1
            continue

        seen.add(key)

        # No context
        outfile.write(f"User: {question}\n")
        outfile.write(f"Assistant: {answer}\n\n")

        added += 1

print()
print("DOLLY DATASET IMPORTED")
print("----------------------")
print(f"Added   : {added:,}")
print(f"Skipped : {skipped:,}")
print(f"Total   : {added + skipped:,}")
print(f"Output  : {output}")
