# Q3.py
# Corrected version avoiding default mutable argument pitfall

def add_item(item, items=None):
    """
    Safely adds items to a list without sharing data across calls.
    """
    if items is None:
        items = []

    items.append(item)
    return items


# Testing multiple calls
print(add_item("apple"))
print(add_item("banana"))
print(add_item("orange"))
