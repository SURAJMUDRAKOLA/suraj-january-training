# Q1.py
# Demonstrates behavior of immutable default parameters

def count_message(msg, count=0):
    count += 1
    print(f"Message received: {msg}")
    print(f"Message count: {count}")
    return count


# 🔽 FUNCTION CALLS (THIS WAS MISSING)
count_message("heya")
count_message("hello")
count_message("welcome")
