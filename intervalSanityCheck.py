def verify_ranges(min_file, og_file, max_file):
    count = 0
    with open(min_file, 'r') as fmin, open(og_file, 'r') as fog, open(max_file, 'r') as fmax:
        for line_number, (min_val, og_val, max_val) in enumerate(zip(fmin, fog, fmax), start=0):
            min_val, og_val, max_val = int(min_val.strip()), int(og_val.strip()), int(max_val.strip())

            if not (min_val <= og_val <= max_val):
                print(f"Line {line_number}: {og_val} is out of range ({min_val}, {max_val})")
                # return False
                count += 1
    if count == 0:
        print("All values are within range.")
    else:
        print(f"Total {count} values are out of range.")
    return True

# Example usage
min_file = "globalMin.txt"
og_file = "abc.txt"
max_file = "globalMax.txt"

verify_ranges(min_file, og_file, max_file)
